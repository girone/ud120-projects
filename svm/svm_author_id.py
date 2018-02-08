#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

#########################################################
### your code goes here ###

# optional decreasing the training set
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

from sklearn.svm import SVC

t0 = time()
clf = SVC(kernel="linear")
clf = SVC(kernel="rbf", C=10000)  # C in [10,100,1000,10000]
clf.fit(features_train, labels_train)
print(time() - t0)

t0 = time()
labels = clf.predict(features_test)
print(time() - t0)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels, labels_test)
print(accuracy)
#########################################################

print(labels[10], labels[26], labels[50])
print(sum(labels))  # No. of emails predicted to Chris