import numpy as np
import matplotlib.pyplot as plt
import ordinaryLeastSquares
import findStat
import franke
import linear_regression
from numpy import random as npr

if __name__ == '__main__':
    
    n = 103
    x1 = np.linspace(0,1,n)
    x2 = np.linspace(0,1,n)
    xx1, xx2 = np.meshgrid(x1, x2)
    xx1 = xx1.reshape((n*n),1)
    xx2 = xx2.reshape((n*n),1)

    y = franke.Franke(xx1, xx2, var=0)
    print(y.shape)
    
    """
    n = 100
    x1 = np.linspace(1,n,n)
    x2 = np.linspace(1,n,n)
    xx1, xx2 = np.meshgrid(x1, x2)
    xx1 = xx1.reshape((n*n),1)
    xx2 = xx2.reshape((n*n),1)

    y = xx1+xx2**2    
    """

    linear_reg = linear_regression.linear_regression2D(xx1, xx2, y)
    linear_reg.apply_ols(order=5, test_ratio=0.1)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)
    linear_reg.apply_ols_with_bootstrap(order=5, test_ratio=0.1, n_boots=10)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)
    linear_reg.apply_ols_with_crossvalidation(order=5, kfolds=10)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)

    xx1 = np.array([3,5,7,9])
    xx2 = np.array([2,4,8,10])
    y = np.array([10,3,7,15])

    linear_reg = linear_regression.linear_regression2D(xx1, xx2, y)
    linear_reg.apply_ols(order=1, test_ratio=0.0)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)
    linear_reg.apply_ols_with_bootstrap(order=1, test_ratio=0.1, n_boots=10)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)
    linear_reg.apply_ols_with_crossvalidation(order=1, kfolds=10)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)

