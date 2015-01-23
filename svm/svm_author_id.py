#!/usr/bin/python

"""
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors

    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
from sklearn.svm import SVC
sys.path.append("C:\\Users\\dgb_us\\ud421-projects\\tools")
from email_preprocess import preprocess
print sys.path

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#########################################################
### your code goes here ###
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf = SVC(C=10000.0,kernel="rbf", gamma=0.0)

t0 = time()
clf.fit(features_train, labels_train)
print "training time.",round(time()-t0,3),"s"
t0 = time()
pred = clf.predict(features_test)
print "prediction time.",round(time()-t0,3),"s"

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)



#########################################################
print "Accuracy: ",acc
print pred[10],":",pred[26],":",pred[50],":"
print sum(pred)




