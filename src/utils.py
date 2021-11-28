import numpy as np
from random import random, seed
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import KFold 
from sklearn import linear_model


def skl_standardscaler(X):
    scaler = StandardScaler(with_std=True)
    return scaler.fit(X).transform(X)

def min_max_scaling(X, a, b):
    # to make broadcasting work
    numerator = X - X.min(axis=0)[np.newaxis, :]
    denominator = X.max(axis=0)[np.newaxis, :] - X.min(axis=0)[np.newaxis, :]
    min_max_scaled = (b-a)*(numerator/denominator) - a
    return min_max_scaled


def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2)-0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-((9*x-7)**2)/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2-(9*y-7)**2)
    return term1+term2+term3+term4


def create_data(N=20, var=0.0):
    x1 = np.linspace(0, 1, N)
    y1 = np.linspace(0, 1, N)
    x, y = np.meshgrid(x1, y1)

    if var != 0.0:
        z = FrankeFunction(x.flatten(), y.flatten())
        z = z + var*np.random.normal(size=z.shape)
    else:
        z = FrankeFunction(x.flatten(), y.flatten())
    return x1, y1, x, y, z


def create_design_matrix(x1, y1, degree):
    power_list = [(i, j) for i in range(0, degree + 1) for j in range(0, degree + 1) if i+j <= degree]
    X = np.zeros((len(x1), len(power_list)))
    for index, (x_pow, y_pow) in enumerate(power_list):
        X[:, index] = x1**(x_pow)*y1**(y_pow)
    return X


def statistics(y_test, y_predict, sampling=None):
    def R2(y_data, y_predict):
        return 1 - np.sum((y_data - y_predict)**2)/np.sum((y_data - np.mean(y_data))**2)

    def MSE(y_data, y_predict):
        n = np.size(y_predict)
        return np.sum((y_data-y_predict)**2)/n

    if sampling is not None:
        def error_bias_variance(y_pred, y_test):
            error = np.mean(np.mean((y_pred-y_test)**2, axis=1, keepdims=True))
            bias = np.mean((y_test-np.mean(y_pred, axis=1, keepdims=True))**2)
            variance = np.mean(np.var(y_pred, axis=1, keepdims=True))
            return bias, variance, error
        statistics = {'MSE': MSE(y_test, y_predict), 'R2': R2(y_test, y_predict), 'EBV': error_bias_variance(y_test, y_predict)}
    else:
        statistics = {'MSE': MSE(y_test, y_predict), 'R2': R2(y_test, y_predict)}
    return statistics
