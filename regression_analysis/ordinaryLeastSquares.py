import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import findStat

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

def ols2D(x1, x2, y, order, test_ratio):
    """performs ordinary least squares for a 2D function using polynomials of order "order".
        The test ratio is the fraction of data that is used as testing data. 
        x1, x2 are 1D arrays of input and y is the 1d array of output


    """
    points = np.size(x1)

    #splitting data
    #np.hstack([x1, x2])
    x = np.zeros((points, 2))
    x_train, x_test, y_train, y_test = train_test_split(np.hstack([x1, x2]), y, test_size=test_ratio)
    
    x1_train = x_train[:, 0]
    x2_train = x_train[:, 1]

    x1_test = x_test[:, 0]
    x2_test = x_test[:, 1]

    #find train design matrix
    X = design_mat2D(x1_train, x2_train, order)

    #finding model parameters
    X = np.asmatrix(X)
    XT = np.asmatrix(np.transpose(X))
    XTX = XT*X
    C = np.linalg.inv(XTX)
    beta = (C*XT)*y_train #model parameters

    y_model_test = np.array(design_mat2D(x1_test, x2_test, order) @ beta)
    y_model_train = np.array(design_mat2D(x1_train, x2_train, order) @ beta)

    return y_train, y_test, y_model_train, y_model_test
"""
    trainMSE = findStat.findMSE(y_train, y_model_train)
    testMSE = findStat.findMSE(y_test, y_model_test)
    trainR2 = findStat.findR2(y_train, y_model_train)
    testR2 = findStat.findR2(y_test, y_model_test)

    return trainMSE, testMSE, trainR2, testR2
"""