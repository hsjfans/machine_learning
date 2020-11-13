import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(y_pred,y_true):
    """confusion_matrix 
    
    one_hot matrix 

    Parameter:
        y_pred: the prediction of input, shape [n_samples, n_class]
        y_true: the label, shape [n_samples, n_class]
    Return:
        confusion, shape,[n_class, n_class]
        
    """

    nums, class_num = y_pred.shape
    cm = np.zeros((class_num,class_num))
    for i in range(class_num):
        for j in range(class_num):
            c1 = np.argmax(y_true,axis = 1) == i
            c2 = np.argmax(y_pred, axis = 1) == j
            cm[i][j] = (c1 & c2).sum()                   
    return cm


def precision(y_pred, y_true):
    """
    precision

    """
    assert y_pred.shape[1] == 2
    cm = confusion_matrix(y_pred, y_true)
    return cm[0][0]*1.0/(cm[0][0] + cm[1][0])


def recall(y_pred, y_true):
    """
    recall

    """
    assert y_pred.shape[1] == 2
    cm = confusion_matrix(y_pred, y_true)
    return cm[0][0]*1.0/(cm[0][0] + cm[0][1])


def f1_score(y_pred, y_true):
    """f1_score for binary 
    
    P = TP/(TP + FP)
    R = TP/(TP + FN)
    F_1 = PxR/(P+R)

    """

    assert y_pred.shape[1] == 2
    cm = confusion_matrix(y_pred, y_true)
    p = cm[0][0]*1.0/(cm[0][0] + cm[1][0])
    r = cm[0][0]*1.0/(cm[0][0] + cm[0][1])
    return p*r/(p+r)



def accuracy_score(y_pred, y_true):
    """accuracy score for classifiction
    
    """
    
    nums = y_true.shape[0]
    acc  = (np.argmax(y_true, axis = 1) == np.argmax(y_pred, axis = 1)).sum()
    return acc * 1.0/ nums


def ROC(y_pred, y_true, positive_column = 0,draw = True):
    """
    ROC 
    """
    
    y_pred = y_pred[:,0]
    y_true = y_true[:,0]
    
    # sort by y_pred 
    sort_index = np.argsort(-y_pred)
    y_pred = y_pred[sort_index]
    y_true = y_true[sort_index]

    tprs = []
    fprs = []
    positive_num = (y_true == 1.0).sum() 
    negivate_num = len(y_true) - positive_num 
    for threshold in np.arange(0,1+0.1,0.1):   
        t = ((y_true == 1.0)& (y_pred >= threshold)).sum()
        f = ((y_true == 0.0) & (y_pred >= threshold)).sum()
        tprs.append(t*1.0/positive_num)
        fprs.append(f*1.0/negivate_num)
    if draw:
        plt.plot(fprs,tprs,c='r')
        plt.show()
    return tprs, fprs


def AUC(y_pred, y_true, positive_num = 0,draw = True):
    """

    """
    tprs,fprs = ROC(y_pred,y_true, positive_num, draw)
    auc = 0.0
    for i in np.arange(0,len(tprs)-1,1):
        auc += 0.5*(tprs[i+1]+tprs[i])*(fprs[i]-fprs[i+1])
    return auc
