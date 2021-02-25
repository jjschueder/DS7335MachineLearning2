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


# ==============================================================================
# packages
# ==============================================================================

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score # other metrics too pls!
from sklearn.metrics import classification_report
#from sklearn.metrics import cross_validate
from sklearn.ensemble import RandomForestClassifier # more!
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from itertools import *
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
# importing module  
import csv 
import urllib.request
import requests 
import io
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
import os
import json

# Import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize


os.environ['HTTP_PROXY'] = 'http://proxy.rockwellcollins.com:9090'
os.environ['http_proxy'] = 'http://proxy.rockwellcollins.com:9090'
os.environ['HTTPS_PROXY'] = 'http://proxy.rockwellcollins.com:9090'

#bclf = clf(**param)

# ==============================================================================
# functions
# ==============================================================================



# ==============================================================================
# runs specified classifier with data and hyper parmaters provided
# ==============================================================================
def run(a_clf, data, clf_hyper):
  M, L, n_folds = data # unpack data container
  #kf = KFold(n_splits=n_folds) # Establish the cross validation
  skf = StratifiedKFold(n_splits=n_folds)
  ret = {} # classic explication of results

  for ids, (train_index, test_index) in enumerate(skf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack parameters into clf if they exist
    clf.fit(M[train_index], L[train_index])
    pred = clf.predict(M[test_index])
    predproba = clf.predict_proba(M[test_index])
    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred),
               'precision': precision_score(L[test_index], pred,  average='weighted'),
               'recall' : recall_score(L[test_index], pred,  average='weighted'),
               'roc_auc' : roc_auc_score(L[test_index], predproba, average='weighted', multi_class='ovo')}
  return ret

def get_dict_wo_key(dictionary, keyp):
    """Returns a **shallow** copy of the dictionary without a key."""
    _dict = dictionary.copy()
    for key in keyp:
        #_dict.pop(key)
        del _dict[key]
    return _dict

# ==============================================================================
# gets permutations of classifiers and hyperparameters
# ==============================================================================
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


# ==============================================================================
# gets average evaluation metrics n_folds
# ==============================================================================
def get_avg_results(results, n_folds):
                sumacc, avg_acc, sumrec, avg_rec, sumprec, avg_prec, sumrocauc, avg_rocauc = 0,0,0,0,0,0,0,0
                for k, v in results.items():
                    print(v['accuracy'])
                    sumacc+= v['accuracy']
                    sumrec += v['recall']
                    sumprec += v['precision']
                    sumrocauc += v['roc_auc']
                avg_acc = sumacc/n_folds
                avg_rec = sumrec/n_folds
                avg_prec = sumprec/n_folds
                avg_rocauc = sumrocauc/n_folds
                acc = {}
                acc.update({'avg_acc' : avg_acc, 'avg_rec' : avg_rec, 'avg_prec' : avg_prec, 'avg_rocauc': avg_rocauc})
                return acc    
    #pick best option
    
# adapt this code below to run your analysis
# 1. Write a function to take a list or dictionary of clfs and hypers(i.e. use logistic regression), each with 3 different sets of hyper parameters for each
#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


# ==============================================================================
# main
# ==============================================================================
def main(XMFL, YLE):
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

# ==============================================================================
# call functions to get permutations of classifiers and hyperparameters
# ==============================================================================    
    
    knncombos = permute_grid(knn_param_grid)
    rfcombos = permute_grid(rf_param_grid)
    adacombos = permute_grid(ada_param_grid)

       

    
    #M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
    #L = np.ones(M.shape[0])
    
# ==============================================================================
# Call grid search with data and classifier parameters 
# 6. Investigate grid search function
# ==============================================================================      

    #LongLongLiveGridS#LongLon#LLongLiveGridSearch!gLiveGridSearch!
    n_folds = 5
    data_iris = (XMFL, YLE, n_folds)
    results_iris_all = []
    print("names:", names)
    print("classifiers", classifiers)
    for name, clf in zip(names, classifiers):
        print (name, clf)
        if name == 'Nearest Neighbors':
            for param in knncombos:
                results_iris = run(clf, data_iris, clf_hyper=param)
                modelname = {'name': 'knn'}
                acc = get_avg_results(results_iris, n_folds)
                res = {**param, **modelname, **acc}
                results_iris_all.append(res)
        elif name == 'Random Forest':
           for param in rfcombos: 
               results_iris = run(clf, data_iris, clf_hyper=param)
               modelname = {'name': 'rf'}
               acc = get_avg_results(results_iris, n_folds)
               res = {**param, **modelname, **acc}
               results_iris_all.append(res)
        elif name == 'Ada Boost':
           for param in adacombos:
                results_iris = run(clf, data_iris, clf_hyper=param)
                modelname = {'name': 'ada'}
                acc = get_avg_results(results_iris, n_folds)
                res = {**param, **modelname, **acc}
                results_iris_all.append(res)
        else:
            print('Sorry that classifier is not defined')
        
       
            
    return results_iris_all

def scorer(results, metric):
    maxout =  max(results, key=lambda x:x[metric])
    return maxout


# ==============================================================================
# plot curves of best result
# ==============================================================================
def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6)):

    roc_list = []
    lw = 2
    X_train, X_test, y_train, y_test = train_test_split(X,YLBIN, test_size=0.2)
# Learn to predict each class against the other
    rfclassifiercv = OneVsRestClassifier(clf)
    rfbinarymodel = rfclassifiercv.fit(X_train, y_train)
    rfbinaryscore = rfclassifiercv.predict(X_test)
    y_score = cross_val_predict(rfclassifiercv, X, YLBIN, cv=10 ,method='predict_proba')

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(YLBIN[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(YLBIN.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    from scipy import interp
    from itertools import cycle
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    roc_list.append(("Model 2", "RF", roc_auc["macro"],fpr["macro"],tpr["macro"]))
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.rcParams["figure.figsize"] = (10,10)
    #plt.savefig('bestresultgraph.png')
    #plt.show()
    plt.savefig('bestresultgraph.png')
    
if __name__ == "__main__":

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

    #creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    YLE = le.fit_transform(YL)
    YLBIN = label_binarize(YLE, classes=[0, 1, 2])
    mainout = main(XMFL, YLE.ravel())
    
     



#find the best one        
    maxscore = scorer(mainout, 'avg_rocauc')
    #could use avg_acc, etc
    


    keyValList = ['knn']
    knnout = [d for d in mainout if d['name'] in keyValList]
    keyValList = ['rf']
    rfout = [d for d in mainout if d['name'] in keyValList]
    keyValList = ['ada']
    adaout = [d for d in mainout if d['name'] in keyValList]
    
    maxknn = scorer(knnout, 'avg_rocauc')
    maxada = scorer(adaout, 'avg_rocauc')
    

    # classifier winner 
    clf = RandomForestClassifier(
                        max_depth=20,
                        n_estimators=200,
                       random_state=101)
    numclasses = 3
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings    
    plot_multiclass_roc(clf, X, YLBIN, n_classes=numclasses, figsize=(16, 10))
    

    
# 5. Please set up your code to be run and save the results to the directory that its executed from
    
    with open('gridresults.json', 'w') as fout:
            json.dump(mainout, fout)        