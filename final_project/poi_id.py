#!/usr/bin/python

import sys
import argparse
import pickle
import copy
import random
import numpy as np
from itertools import tee, izip
from sklearn.pipeline import make_pipeline
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# TODO(Jonas): Randomize the seed.
SEED = 1


def plot_two_features(data_dict, feature1, feature2, annotate=False):
    """Creates a scatterplot with optional labels on data points."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for name, entry in data_dict.iteritems():
        x = np.nan_to_num(entry[feature1])
        y = np.nan_to_num(entry[feature2])
        ax.scatter(x, y, color="red" if entry["poi"] else "black")
        if annotate:
            ax.annotate(name, (x, y))
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


# TODO(Jonas): Set reasonable defaults here when cleaning up. Whatever performs best.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--algorithm",
    choices=[
        "naive_bayes", "decision_tree", "linear_svc", "rbf_svc",
        "logistic_regression", "ada_boost", "gradient_boosting",
        "random_forest", "gaussian_process", "stochastic_gradient_descent",
        "multi_layer_perceptron"
    ])
parser.add_argument("--remove-outliers", action="store_true")
parser.add_argument(
    "--feature-scaling", choices=["normal", "robust"], default=None)
parser.add_argument(
    "--feature-selection",
    choices=[None, "kbest", "p68.5", "RFECV", "linear_model"],
    default=None)
args = parser.parse_args()

# Task 1: Select what features you'll use.

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
FEATURES_EMAIL_COUNTS = [
    "to_messages",
    "from_poi_to_this_person",
    "from_messages",
    "from_this_person_to_poi",
]
FEATURES_OTHER = ["shared_receipt_with_poi"]
FEATURES_ALL = FEATURES_PAYMENT + FEATURES_STOCK + FEATURES_EMAIL_COUNTS + FEATURES_OTHER
FEATURES_ONE_OF_EACH_FOR_TESTING = [
    "bonus", "exercised_stock_options", "to_messages"
]
FEATURES_TRY = ["shared_receipt_with_poi"]
FEATURE_EMAIL = "email_address"  # cannot use this non-numerical feature  directly, will convert later

features_list = ["poi"] + FEATURES_ALL
pipeline_steps = []

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Scale the features (with robustness too outliers, which we remove later on).
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = None
if args.feature_scaling == "normal":
    scaler = StandardScaler()
elif args.feature_scaling == "robust":
    scaler = RobustScaler()
if scaler:
    pipeline_steps.append(scaler)

# Validation of scaling:
# from custom_validation import validate_scaling
# original_data_dict = copy.deepcopy(data_dict)
# data_dict = scaler.fit_transform(data_dict)  # needs to work on ndarray instead of list
# validate_scaling(original_data_dict, data_dict)

# Task 2: Remove outliers

# Remove summary entry "TOTAL" and the travel agency (done outside of pipeline
# processing because its bias on the data would be too strong).
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LAY KENNETH L", 0)

# Task 3: Create new feature(s)

print "Adding features..."
from additional_features import EmailShares, PaymentsStockRatio, RelativeFeature, HasEnronEmailAddress
for new_features in [
        EmailShares(),
        PaymentsStockRatio(),
        HasEnronEmailAddress()
]:
    data_dict = new_features.extend(data_dict)
    features_list.extend(new_features.new_feature_names())
    for name in new_features.new_feature_names():
        print " * added feature", name

for feature_list in FEATURES_PAYMENT, FEATURES_STOCK:
    # Assumes that the "total_" feature is the last in the list.
    for feature in feature_list[:-1]:
        new_feature = RelativeFeature(feature, feature_list[-1])
        data_dict = new_feature.extend(data_dict)
        features_list.extend(new_feature.new_feature_names())
        print " * added feature", features_list[-1]

# Remove outliers automatically.

if args.remove_outliers:
    # Setting featureFormat()'s `sort_keys` parameter to true will break this!
    data = featureFormat(
        data_dict,
        features_list,
        sort_keys=False,
        remove_NaN=True,
        remove_all_zeroes=False)  # need to keep these entries
    labels, features = targetFeatureSplit(data)
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

# Visualize features
# for feature1, feature2 in pairwise(features_list[1:]):
#     plot_two_features(data_dict, feature1, feature2, annotate=False)

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
# Select and initialize algorithm
algorithm = args.algorithm
if algorithm == "naive_bayes":
    main_algorithm = GaussianNB()
elif algorithm == "decision_tree":
    main_algorithm = DecisionTreeClassifier()
elif algorithm == "linear_svc":
    main_algorithm = LinearSVC(dual=False, random_state=SEED)
elif algorithm == "rbf_svc":
    main_algorithm = SVC()
elif algorithm == "logistic_regression":
    main_algorithm = LogisticRegression()
elif algorithm == "ada_boost":
    main_algorithm = AdaBoostClassifier(random_state=SEED)
elif algorithm == "gradient_boosting":
    main_algorithm = GradientBoostingClassifier(random_state=SEED)
elif algorithm == "random_forest":
    main_algorithm = RandomForestClassifier()
elif algorithm == "gaussian_process":
    main_algorithm = GaussianProcessClassifier()
elif algorithm == "stochastic_gradient_descent":
    main_algorithm = SGDClassifier()
elif algorithm == "multi_layer_perceptron":
    main_algorithm = MLPClassifier()
else:
    print "Unknown algorithm", algorithm
    exit(1)

# Setup PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components=len(features_list) // 2)

# Use StratifiedShuffleSplit instead of default StratifiedKFold for cross validation:
# See notes.md for a summary of the article at scikit-learn.org on cross validaiton.
# The rationale of stratification is that the relative frequencies of class labels
# (POI or not POI) should be the same in training and test set as in the whole data.
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=SEED)

# Setup feature selection
from sklearn.feature_selection import SelectKBest, SelectPercentile, RFECV, SelectFromModel
feature_selector = args.feature_selection
if args.feature_selection == "kbest":
    feature_selector = SelectKBest(k=len(features_list) // 2)
elif args.feature_selection == "p68.5":
    feature_selector = SelectPercentile(percentile=68.5)
elif args.feature_selection == "RFECV":
    from sklearn.tree import DecisionTreeClassifier
    # estimator = DecisionTreeClassifier(criterion="entropy", random_state=SEED)
    # estimator = LinearSVC()
    estimator = main_algorithm
    feature_selector = RFECV(estimator, cv=sss)
elif args.feature_selection == "linear_model":
    feature_selector = SelectFromModel(
        LinearSVC(penalty="l1", dual=False, random_state=SEED))
if feature_selector:
    # NOTE(Jonas): Disabled because running RFE-CV does not make sense in the
    # tester.py, only slows it down. We rather restrict the features of the
    # input to what we found best.
    # pipeline_steps.append(feature_selector)
    print "Feature selection..."
    data = featureFormat(data_dict, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    feature_selector.fit(features, labels)
    mask = feature_selector._get_support_mask()
    print "\n".join(
        [" * {}: {}".format(a, b) for a, b in zip(features_list[1:], mask)])
    selected_features = [a for a, b in zip(features_list[1:], mask) if b]
    # Filter data_dict by changing the feature list:
    features_list = ["poi"] + selected_features
print "Selected features:", features_list

# Recreate from new data set
data = featureFormat(data_dict, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Create the pipeline of steps
pipeline_steps.append(main_algorithm)
clf = make_pipeline(*pipeline_steps)

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function.

# Search best parameters.
from sklearn.model_selection import GridSearchCV
parameter_grid = {
    "gradientboostingclassifier__criterion": ["friedman_mse", "mae"],
    "gradientboostingclassifier__loss": ["deviance", "exponential"],
    "gradientboostingclassifier__n_estimators":
    [10, 50, 90, 100, 120, 150, 200],
    "gradientboostingclassifier__max_depth": [1, 3, 6, 8, 9, 10, 12, 15, 20],
    "gradientboostingclassifier__max_features": [None, "sqrt", "log2", 0.9],
    "gradientboostingclassifier__subsample": [0.5, 0.8, 0.9, 1.]
}
# Validate the parameter grid:
for key in parameter_grid.keys():
    if key not in clf.get_params():
        raise Exception(
            "'{}' is not among the parameters of your algorithm, which are {}".
            format(key, ",".join(clf.get_params())))
# Optimize for f1 score brings good combination of precision and recall.
scoring_function = "f1"
scoring_function = "recall"
param_search = GridSearchCV(
    estimator=clf,
    param_grid=parameter_grid,
    cv=sss,
    scoring=scoring_function,
    n_jobs=2)
param_search.fit(features, labels)
# print param_search.cv_results_
print "Best estimator:", param_search.best_estimator_
print "Best parameters:", param_search.best_params_
print "Score({}):".format(scoring_function), param_search.best_score_

with open("cv_results.pkl", "wb") as file:
    pickle.dump(param_search.cv_results_, file)

# The result could be used as classifier right away. However, the code in
# `tester.py` fits it to the data several times. This means evaluation
# takes very long when presented with the GridSearch+StratifiedShuffleSplit(n=5)
# classifier. Thus, we extract the best found parameter and store it as classifier.
clf = param_search.best_estimator_

# Apply it on the whole data to get an idea of the other metrics (using multiple
# scoring functions in the GridSearch + refit parameter will repeat the search
# for all scoring functions, but not yield the other scores for the one selected
# for fitting).
predictions = clf.predict(features)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print "Accuracy:", accuracy_score(labels, predictions)
print "Precision:", precision_score(labels, predictions)
print "Recall:", recall_score(labels, predictions)
print "f1_score:", f1_score(labels, predictions)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
