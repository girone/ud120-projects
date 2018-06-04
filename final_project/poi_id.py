#!/usr/bin/python

import sys
import argparse
import pickle
import copy
import random
import numpy as np
from sklearn.pipeline import make_pipeline
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# TODO(Jonas): Set reasonable defaults here when cleaning up. Whatever performs best.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--algorithm",
    choices=["naive_bayes", "decision_tree", "linear_svc"],
    default="naive_bayes")
parser.add_argument("--remove-outliers", action="store_true")
parser.add_argument(
    "--feature-scaling", choices=["normal", "robust"], default=None)
parser.add_argument(
    "--feature-selection",
    choices=[None, "kbest", "p68.5", "RFECV"],
    default=None)
args = parser.parse_args()


def plot_two_features(data_dict, feature1, feature2, annotate=False):
    """Creates a scatterplot with optional labels on data points."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for name, entry in data_dict.iteritems():
        x = entry[feature1]
        y = entry[feature2]
        ax.scatter(x, y, color="red" if entry["poi"] else "black")
        if annotate:
            ax.annotate(name, (x, y))
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()


# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary']  # You will need to use more features

# Available features in the original data:
FEATURES_PAYMENT = [
    "salary", "bonus", "long_term_incentive", "deferred_income",
    "deferral_payments", "loan_advances", "other", "director_fees", "expenses",
    "total_payments"
]
FEATURES_STOCK = [
    "exercised_stock_options", "restricted_stock", "restricted_stock_deferred",
    "total_stock_value"
]
FEATURES_EMAIL = [
    "to_messages", "from_poi_to_this_person", "from_messages",
    "from_this_person_to_poi"
]  # "email_address" not included yet, could add a "has enron email address" feature
FEATURES_OTHER = ["shared_receipt_with_poi"]
FEATURES_ALL = FEATURES_PAYMENT + FEATURES_STOCK + FEATURES_EMAIL + FEATURES_OTHER
FEATURES_ONE_OF_EACH_FOR_TESTING = [
    "bonus", "exercised_stock_options", "to_messages"
]
FEATURES_TRY = ["shared_receipt_with_poi"]

features_list = ["poi"] + FEATURES_ALL
pipeline_steps = []

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Scale the features (with robustness too outliers, which we remove later on).
from sklearn.preprocessing import StandardScaler, RobustScaler
if not args.feature_scaling:
    scaler = None
elif args.feature_scaling == "normal":
    scaler = StandardScaler()
elif args.feature_scaling == "robust":
    scaler = RobustScaler()
if scaler:
    pipeline_steps.append(scaler)

# Validation:
# from custom_validation import validate_scaling
# original_data_dict = copy.deepcopy(data_dict)
# data_dict = scaler.fit_transform(data_dict)  # needs to work on ndarray instead of list
# validate_scaling(original_data_dict, data_dict)

# Task 2: Remove outliers

# Remove summary entry "TOTAL" (done outside of pipeline processing because
# its bias on the data would be too strong).
data_dict.pop("TOTAL", 0)

# Setting featureFormat()'s `sort_keys` parameter to true will break this!
data = featureFormat(
    data_dict,
    features_list,
    sort_keys=False,
    remove_NaN=True,
    remove_all_zeroes=False)  # need to keep these entries
labels, features = targetFeatureSplit(data)

if args.remove_outliers:
    # Remove outliers from the original data.

    # Use Local Outlier Factor LOF to detect ourliers. This is a nearest neighbor
    # based method: For each point, compute the density of its k (say 10) nearest
    # neighbors. Points with a the lowest 5% density are most likely outliers
    # (parameter `contamination`).
    from sklearn.neighbors import LocalOutlierFactor
    # TODO(Jonas): Check the alternative sklearn.svm.OneClassSVM
    outlier_detector = LocalOutlierFactor(
        n_neighbors=10, contamination=0.05, n_jobs=2)
    outlier_labels = outlier_detector.fit_predict(features)
    keys = np.array(data_dict.keys())
    # -1 marks outliers
    outlier_keys = keys[outlier_labels == -1]
    print "Removing {} outliers...".format(len(outlier_keys))
    for outlier in outlier_keys:
        print " -> '{}' {}".format(outlier,
                                   ("who is POI"
                                    if data_dict[outlier]["poi"] else ""))
        data_dict.pop(outlier, 0)
    # NOTE(Jonas): Including this into the pipeline is not yet possible
    # (without extending the API, which is beyond the scope of this project).
    # Will try to use this reasonably, probably after adding the new features.

# Task 3: Create new feature(s)
from additional_features import EmailShares, PaymentsStockRatio
for new_features in [EmailShares(), PaymentsStockRatio()]:
    data_dict = new_features.extend(data_dict)
    features_list.extend(new_features.new_feature_names())

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Select and initialize algorithm
algorithm = args.algorithm
if algorithm == "naive_bayes":
    from sklearn.naive_bayes import GaussianNB
    main_algorithm = GaussianNB()
elif algorithm == "decision_tree":
    from sklearn.tree import DecisionTreeClassifier
    main_algorithm = DecisionTreeClassifier()
elif algorithm == "linear_svc":
    from sklearn.svm import LinearSVC
    main_algorithm = LinearSVC()
else:
    print "Unknown algorithm", algorithm
    exit(1)

# Setup PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=len(features_list) // 2)

# Visualize
# for feature1 in features_list[1:]:
#     for feature2 in features_list[1:]:
#         plot_two_features(data_dict, feature1, feature2, annotate=False)

# Setup feature selection
from sklearn.feature_selection import SelectKBest, SelectPercentile, RFECV
feature_selector = args.feature_selection
if args.feature_selection == "kbest":
    feature_selector = SelectKBest(k=len(features_list) // 2)
elif args.feature_selection == "p68.5":
    feature_selector = SelectPercentile(percentile=68.5)
elif args.feature_selection == "RFECV":
    from sklearn.tree import DecisionTreeClassifier
    estimator = DecisionTreeClassifier()
    feature_selector = RFECV(estimator, n_jobs=2)
if feature_selector:
    pipeline_steps.append(feature_selector)

# Create the pipeline of steps
pipeline_steps.append(main_algorithm)
clf = make_pipeline(*pipeline_steps)

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(20, test_size=0.3, random_state=0)
all_test_predictions = []
all_test_labels = []
for indices_train, indices_test in sss.split(features, labels):
    training_features = []
    testing_features = []
    training_labels = []
    testing_labels = []
    for i in indices_train:
        training_features.append(features[i])
        training_labels.append(labels[i])
    for i in indices_test:
        testing_features.append(features[i])
        testing_labels.append(labels[i])
    clf.fit(training_features, training_labels)
    predictions = clf.predict(testing_features)
    all_test_predictions.extend(list(predictions))
    all_test_labels.extend(list(testing_labels))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print clf
print "Accuracy:", accuracy_score(all_test_labels, all_test_predictions)
print "Precision:", precision_score(all_test_labels, all_test_predictions)
print "Recall:", recall_score(all_test_labels, all_test_predictions)
print "f1_score:", f1_score(all_test_labels, all_test_predictions)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
