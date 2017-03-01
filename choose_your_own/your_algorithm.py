#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

print "Extracting Terrain Data"
features_train, labels_train, features_test, labels_test = makeTerrainData()
print "Data extracted"

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

#### initial visualization
plt.show(block=False)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
print "Plotting done."
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

from time import time
from sklearn.neighbors import KNeighborsClassifier
n = 100
print "Creating classifier..."
clf = KNeighborsClassifier(n_neighbors=n)
print "Done"

#cutting down training sample size to save time
#features_train = features_train[:len(features_train)/10]
#labels_train = labels_train[:len(labels_train)/10]

print "Fitting K-nearest neighbors. n=",n
t = time()
clf.fit(features_train,labels_train)
print time()-t , "s"

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test,pred)

print acc



try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    print "prettyPicture didnt work :("
    pass
