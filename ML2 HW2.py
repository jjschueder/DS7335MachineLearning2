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
from collections import Counter
from itertools import *
from sklearn.model_selection import KFold
# importing module  
import csv 
import urllib.request
import requests 
import io

bclf = clf(**param)

def run(a_clf, data, clf_hyper):
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

def get_dict_wo_key(dictionary, keyp):
    """Returns a **shallow** copy of the dictionary without a key."""
    _dict = dictionary.copy()
    for key in keyp:
        #_dict.pop(key)
        del _dict[key]
    return _dict

def permute_grid(grid):
    result = []
    for p in grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                result = {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    result.append(params)
    return result

def do_grid_search():
    #loop at permutations
    
    #call run
    
    #append results to dictionary
    result = 'blah'
    return result
    
def evaluate_result_dictionary():
    result = 'blah'
    return result
    #compare accuracy/precision/recall on all algorithms/parmeters
    
    #pick best option
    
# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

def main():
# ==============================================================================
# Intitialize the classifier parameters   
# ==============================================================================    
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
             'n_neighbors':[2,3,5]
             
        }]
    
    
    
    rf_param_grid = [
        {
             'n_estimators': [200, 500], 
             'max_depth': [20,30,35],
             'random_state':[101]
         }
    ]
    
    
    ada_param_grid = [{ 'n_estimators': [50, 100, 200, 500]
                  ,'random_state': [101]
                  ,'learning_rate':[1]
                  
                 }]
    # 2. Expand to include larger number of classifiers and hyperparameter settings
    # 3. Find some simple data

    
    
    knncombos = permute_grid(knn_param_grid)
    rfcombos = permute_grid(rf_param_grid)
    adacombos = permute_grid(ada_param_grid)
# ==============================================================================
# pull in a data set  
# ==============================================================================      
        
    url = 'https://raw.githubusercontent.com/jjschueder/DS7335MachineLearning2/main/Iris.csv'
    
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
    XMFL = XM.astype(np.float64)
    YL = np.array(Y) 
    # Import LabelEncoder
    from sklearn import preprocessing
    #creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    YLE = le.fit_transform(YL)
       

    
    #M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
    #L = np.ones(M.shape[0])
    
# ==============================================================================
# Call grid search with data and classifier parameters 
# 6. Investigate grid search function
# ==============================================================================      
    results = run(RandomForestClassifier, data, clf_hyper={})
    results = run(clf, data_iris, clf_hyper=param)
    sumacc = 0
    for k, v in results.items():
        print(v['accuracy'])
        sumacc += v['accuracy']
        avg_acc = sumacc/n_folds
    #LongLongLiveGridS#LongLon#LLongLiveGridSearch!gLiveGridSearch!
    n_folds = 5
    data_iris = (XMFL, YLE.ravel(), n_folds)
    results_iris_all = []
    for name, clf in zip(names, classifiers):
        print (name, clf)
        if name == 'Nearest Neighbors':
            for param in knncombos:
                results_iris = run(clf, data_iris, clf_hyper=param)
                modelname = {'name': 'knn'}
                sumacc, avg_acc = 0,0
                for k, v in results_iris.items():
                    print(v['accuracy'])
                    sumacc += v['accuracy']
                    avg_acc = sumacc/n_folds
                    acc = {}
                    acc.update({'avg_acc' : avg_acc})
                res = {**results_iris, **param, **modelname, **acc}
                results_iris_all.append(res)
        elif name == 'Random Forest':
           for param in rfcombos: 
               results_iris = run(clf, data_iris, clf_hyper=param)
               modelname = {'name': 'rf'}
               sumacc, avg_acc = 0,0
               for k, v in results_iris.items():
                    print(v['accuracy'])
                    sumacc += v['accuracy']
                    avg_acc = sumacc/n_folds
                    acc = {}
                    acc.update({'avg_acc' : avg_acc})
               res = {**results_iris, **param, **modelname, **acc}
               results_iris_all.append(res)
        elif name == 'Ada Boost':
           for param in adacombos:
                results_iris = run(clf, data_iris, clf_hyper=param)
                modelname = {'name': 'ada'}
                sumacc, avg_acc = 0,0
                for k, v in results_iris.items():
                    print(v['accuracy'])
                    sumacc += v['accuracy']
                    avg_acc = sumacc/n_folds
                    acc = {}
                    acc.update({'avg_acc' : avg_acc})
                res = {**results_iris, **param, **modelname, **acc}
                results_iris_all.append(res)
        else:
            print('Sorry that classifier is not defined')
        
        #find the best one
        
        results_iris_all
        seq = [x['avg_acc'] for x in results_iris_all]
        min(seq)
        max(seq)
        
        maxaccItem = max(results_iris_all, key=lambda x:x['avg_acc'])
        minaccItem = min(results_iris_all, key=lambda x:x['avg_acc'])
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
# 5. Please set up your code to be run and save the results to the directory that its executed from

    
if __name__ == "__main__":
    # output_dir = '/usr/lfs/v1/data/ServiceTech/temp'
    # pn='822-2332-100'
    # sap = SapDataCollection(pn, output_dir)
    # _, _, sn_set = sap.collect_data()
    # tdms = TdmsDataCollection(pn, output_dir, sn_set=sn_set)
    # tdms_data = tdms.collect_data()
    main()
