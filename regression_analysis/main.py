import numpy as np
import matplotlib.pyplot as plt
import ordinaryLeastSquares
import findStat
import franke


if __name__ == '__main__':
    n = 100
    x1 = np.linspace(0,1,n)
    x2 = np.linspace(0,1,n)
    xx1, xx2 = np.meshgrid(x1, x2)
    xx1 = xx1.reshape((n*n),1)
    xx2 = xx2.reshape((n*n),1)

    y = franke.Franke(xx1, xx2)

    trainMSE, testMSE, trainR2, testR2 = ordinaryLeastSquares.ols2D(x1=xx1, x2=xx2, y=y, order=5, test_ratio=0.9, bootstrap=True)
    print(trainMSE)
    print(testMSE)