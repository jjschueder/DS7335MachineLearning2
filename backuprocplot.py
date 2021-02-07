# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 13:31:42 2021

@author: jjsch
"""

    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_predict
    roc_list = []
    lw = 2
    X_train, X_test, y_train, y_test = train_test_split(X,YLBIN, test_size=0.2)
# Learn to predict each class against the other
    rfclassifiercv = OneVsRestClassifier(RandomForestClassifier(
                        max_depth=20,
                        n_estimators=200,
                       random_state=101))
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
    plt.show()
    
    numclasses = len(np.unique(YLE))