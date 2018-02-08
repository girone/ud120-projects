#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print("Number of features:", features_train.shape[1])

#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier

t0 = time()
classifier = DecisionTreeClassifier(criterion="entropy", min_samples_split=40)
classifier.fit(features_train, labels_train)
print("Training time:", time() - t0)
print("This can be reduced by a large amount by changing the percentile " +
      "of features used in email_preprocess.py to a smaller value (see " +
      "`SelectPercentile()`).")

t0 = time()
labels = classifier.predict(features_test)
print("Prediction time:", time() - t0)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(labels, labels_test))


#########################################################


