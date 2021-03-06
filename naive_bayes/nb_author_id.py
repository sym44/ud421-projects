#!/usr/bin/python

"""
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project

    use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1

"""

import sys
from sklearn.naive_bayes import GaussianNB
from time import time

sys.path.append("C:\\Users\\dgb_us\\ud421-projects\\tools")

from email_preprocess import preprocess
print sys.path

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
t0 = time()
features_train, features_test, labels_train, labels_test = preprocess()
print "preprocess time.",round(time()-t0,3),"s"

#########################################################
### your code goes here ###

clf = GaussianNB()
t0 = time()
clf.fit(features_test, labels_test)
print "training time.",round(time()-t0,3),"s"
t0 = time()
pred_test = clf.predict(features_test)
print "prediction time.",round(time()-t0,3),"s"

count = 0
n_points=len(labels_test)
for i in range(0,n_points-1):
    if(labels_test[i] == pred_test[i]):
        count+=1

print float(count)/float(n_points)

print clf.score(features_test,labels_test)

#########################################################


