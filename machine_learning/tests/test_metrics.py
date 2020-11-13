from machine_learning.metrics import accuracy_score, f1_score,ROC,AUC
import pytest
import numpy as np

def test_f1_score():

    y_true = [1,1,0,0,1,0]
    y_pred = [1,0,1,0,0,1]
    n_class = 2

    y_true = np.eye(n_class)[y_true]
    y_pred = np.eye(n_class)[y_pred]
    
    print(y_true, 'y_true')
    score = f1_score(y_pred, y_true)

    print('f1_score = {}'.format(score))


def test_accuracy_score():
    
    y_true = [1,1,0,0,1,0]
    y_pred = [1,0,1,0,0,1]
    n_class = 2

    y_true = np.eye(n_class)[y_true]
    y_pred = np.eye(n_class)[y_pred]

    print(' accuracy_score = {}'.format(accuracy_score(y_pred,y_true)))



def test_AUC():

    y_pred = np.random.rand(10000)
    y_true = np.random.rand(10000).round()
    
    n_class = 2

    y_pred = np.stack((y_pred,1- y_pred),axis = 1)
    y_true = np.stack((y_true,1- y_true),axis = 1)
    
    score =  AUC(y_pred,y_true)
    print('AUC = ',score)
