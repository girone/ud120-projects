#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### code from previous lecture
from sklearn.model_selection import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(features, labels, random_state=42, test_size=0.3)


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(data_train, labels_train)

predicted_labels = clf.predict(data_test)

print sum(predicted_labels), len(predicted_labels)

for i, v in enumerate(predicted_labels):
    if v == 1. and v == labels_test[i]:
        print 1


### your code goes here 
from sklearn.metrics import precision_score, recall_score
print "precision_score:", precision_score(predicted_labels, labels_test)
print "recall_score:", recall_score(predicted_labels, labels_test)