# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:55:46 2021

@author: jjschued
"""
# ==============================================================================
# File: ML2 HW2.py
# Project: Machine Learning 2 Homework 2 Death to Grid Search
# Author: Joe Schueder
# File Created: Jan 13, 2021
# ==============================================================================

import numpy as np
from sklearn.metrics import accuracy_score # other metrics too pls!
from sklearn.ensemble import RandomForestClassifier # more!
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold
# importing module  
import csv 
import urllib.request
import requests 
import io

# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
names = ["Nearest Neighbors", "Random Forest",
         "Ada Boost"]

classifiers = [
    KNeighborsClassifier,
    #(3),
#class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)    
    RandomForestClassifier,
    #(max_depth=5, n_estimators=10, max_features=1),   
#class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)    
    AdaBoostClassifier,
    #(n_estimators=100, random_state=0),
#class sklearn.ensemble.AdaBoostClassifier(base_estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)    
   ]

knn_param_grid = [
    {
         'weights': ['uniform','distance'],
         'leaf_size': [5,10],
         'metric': ['minkowski','euclidean'],
         'n_neighbors':[2,3,5],
         'random_state': [101]
         
    }
]


rf_param_grid = [
    {
         'n_estimators': [200, 500], 
         'max_depth': [20,30,35],
         'random_state':[101]
     }
]


ada_param_grid = { 'n_estimators': [50, 100, 200, 500]
              ,'random_state': [101]
              ,'learning_rate':[0]
              
             }
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data

url = 'https://raw.githubusercontent.com/jjschueder/DS7335MachineLearning2/main/Iris.csv'

def get_dict_wo_key(dictionary, keyp):
    """Returns a **shallow** copy of the dictionary without a key."""
    _dict = dictionary.copy()
    for key in keyp:
        #_dict.pop(key)
        del _dict[key]
    return _dict


#Get data set from github
mylist = []
r = requests.get(url)
buff = io.StringIO(r.text)
dr = csv.DictReader(buff)
for row in dr:
    print(row)
    mylist.append(dict(row))
    
# for d in mylist:
#     print(d['Species'])


yremovals = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
xremovals = ['Species', 'Id']

for key in xremovals:
    print("Value", key)


xvalues = []
for row in range(len(mylist)):
    mylistx = get_dict_wo_key(mylist[row], xremovals)
    xvalues.append(mylistx)
  

yvalues = []
for row in range(len(mylist)):
    mylisty = get_dict_wo_key(mylist[row], yremovals)
    yvalues.append(mylisty)
    
X = []    
for row in range(len(xvalues)):    
    X.append(list(xvalues[row].values()))
    
Y = []    
for row in range(len(yvalues)):    
    Y.append(list(yvalues[row].values()))    

XM = np.array(X)
YL = np.array(Y) 

   
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

#M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
#L = np.ones(M.shape[0])
n_folds = 5

#data = (M, L, n_folds)

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data container
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explication of results

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack parameters into clf if they exist
    clf.fit(M[train_index], L[train_index])
    pred = clf.predict(M[test_index])
    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret

#results = run(RandomForestClassifier, data, clf_hyper={})
#LongLongLiveGridS#LongLon#LLongLiveGridSearch!gLiveGridSearch!
data_iris = (XM, YL.ravel(), n_folds)
results_iris_all = []
for name, clf in zip(names, classifiers):
    results_iris = run(clf, data_iris, clf_hyper={})
    results_iris_all.append(results_iris)
