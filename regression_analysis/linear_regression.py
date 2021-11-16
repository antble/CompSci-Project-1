import numpy as np
from numpy import random as npr
import findStat
from sklearn.model_selection import train_test_split

def poly_powers2D(order):
    x1pow = [0]
    x2pow = [0]
    for i in np.arange(1,order+1):
        for j in np.arange(0,i+1):
            x1pow.append(j)
            x2pow.append(i-j)
    return x1pow, x2pow

def design_mat2D(x1, x2, order):
    x1pow, x2pow = poly_powers2D(order)
    n = np.size(x1)
    design_mat = np.zeros((n, len(x1pow)))
    for term in range(len(x1pow)):
        design_mat[:, term] = (x1**x1pow[term])*(x2**x2pow[term])
    return design_mat

def find_params(X, y_train):
    X = np.asmatrix(X)
    XT = np.asmatrix(np.transpose(X))
    XTX = XT*X
    C = np.linalg.inv(XTX)
    beta = (C*XT)*y_train #model parameters
    return beta

class linear_regression2D():
    def __init__(self, x, y, **kwargs):
        self.x = x #input 2Ddata
        self.y = y #output
        self.n_points = len(y)
        self.trainMSE = np.nan
        self.trainR2 = np.nan
        self.testMSE = np.nan
        self.testR2 = np.nan

    def apply_ols(self, order=3, test_ratio=0.0):
        if(test_ratio!=0.0):
            x_train, x_test, y_train, y_test = train_test_split(np.hstack([x1, x2]), y, test_size=test_ratio)
            
            x1_train = x_train[:, 0]
            x2_train = x_train[:, 1]

            x1_test = x_test[:, 0]
            x2_test = x_test[:, 1]
        else:
            x1_train = x1.flatten()
            x2_train = x2.flatten()
            x1_test = np.array([])
            x2_test = np.array([])
            y_train = y
            y_test = np.array([])

        #find train design matrix
        X = design_mat2D(x1_train, x2_train, order)

        #finding model parameters
        beta = find_params(X, y_train)
        
        y_model_train = np.array(design_mat2D(x1_train, x2_train, order) @ beta) #fitting training data
        self.trainMSE = findStat.findMSE(y_train, y_model_train)
        self.trainR2 = findStat.findR2(y_train, y_model_train)
        if(test_ratio!=0.0):
            y_model_test = np.array(design_mat2D(x1_test, x2_test, order) @ beta) #fitting testing data
            self.testMSE = findStat.findMSE(y_test, y_model_test)
            self.testR2 = findStat.findR2(y_test, y_model_test)
