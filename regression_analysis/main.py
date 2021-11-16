import numpy as np
import matplotlib.pyplot as plt
import ordinaryLeastSquares
import findStat
import franke
import linear_regression


if __name__ == '__main__':
    
    n = 103
    x1 = np.linspace(0,1,n)
    x2 = np.linspace(0,1,n)
    xx1, xx2 = np.meshgrid(x1, x2)
    xx1 = xx1.reshape((n*n),1)
    xx2 = xx2.reshape((n*n),1)

    y = franke.Franke(xx1, xx2)
    """
    n = 10
    x1 = np.linspace(1,n,n)
    x2 = np.linspace(1,n,n)
    xx1, xx2 = np.meshgrid(x1, x2)
    xx1 = xx1.reshape((n*n),1)
    xx2 = xx2.reshape((n*n),1)

    y = xx1+xx2    
    """
    linear_reg = linear_regression.linear_regression2D(xx1, xx2, y)
    linear_reg.apply_ols(order=4, test_ratio=0.1)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)
    linear_reg.apply_ols_with_bootstrap(order=4, test_ratio=0.1, n_boots=10)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)
    linear_reg.apply_ols_with_crossvalidation(order=4, test_ratio=0.1, kfolds=10)
    print(linear_reg.trainMSE)
    print(linear_reg.testMSE)