#!/usr/bin/python

import matplotlib.pyplot as plt
from time import time
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
internal_clf = RandomForestClassifier(criterion="entropy", min_samples_leaf=0.1)
clf = AdaBoostClassifier(base_estimator=internal_clf, 
                         n_estimators=15,
                         algorithm="SAMME.R")
t0 = time()
clf.fit(features_train, labels_train)
print("Training time:", time() - t0)

t0 = time()
labels = clf.predict(features_test)
print("Prediction time:", time() - t0)


from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(labels_test, labels))

prettyPicture(clf, features_test, labels_test)
# try:
    
# except NameError:
#     pass
