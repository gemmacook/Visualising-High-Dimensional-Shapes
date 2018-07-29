# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 01:42:06 2018

@author: Cameron Hargreaves
"""
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.datasets import fetch_olivetti_faces
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

def printFaces(images, target, top_n):
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)   
        # label the image with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))

def evaluateCrossValidation(clf, X, y, K):
    # create a k-fold croos validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # This is the average accuracy of each fold
    scores = cross_val_score(clf, X, y, cv=cv)
    print("Scores of five fold validation: \n" + str(scores))
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))
    
def trainAndEvaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))
    
    y_pred = clf.predict(X_test)
    
    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))
    
faces = fetch_olivetti_faces()
printFaces(faces.images, faces.target, 20)
svc = SVC(kernel='linear')
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=0)
evaluateCrossValidation(svc, X_train, y_train, 5)
trainAndEvaluate(svc, X_train, X_test, y_train, y_test) 




