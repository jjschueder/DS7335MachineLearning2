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
from sklearn.model_selection import KFold

# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
# 2. Expand to include larger number of classifiers and hyperparameter settings
# 3. Find some simple data


# importing module  
import csv 
import urllib2

url = 'https://raw.githubusercontent.com/jjschueder/DS7335MachineLearning2/main/Iris.csv'
response = urllib2.urlopen(url)
cr = csv.reader(response)  
# opening the file using "with" 
# statement 
with open(open(filename, 'r') as data:      
    for line in csv.DictReader(data): 
        print(line) 

# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function

M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
L = np.ones(M.shape[0])
n_folds = 5

data = (M, L, n_folds)

def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data container
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explication of results

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack parameters into clf is they exist
    clf.fit(M[train_index], L[train_index])
    pred = clf.predict(M[test_index])
    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret

results = run(RandomForestClassifier, data, clf_hyper={})
#LongLongLiveGridS#LongLon#LLongLiveGridSearch!gLiveGridSearch!
