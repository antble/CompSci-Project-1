from sklearn.utils import resample
from sklearn.model_selection import KFold 
from src.utils import statistics

import numpy as np 
import pandas as pd


def bootstrap(num_boots, train_data, X_test, model, lmb=0):
    X_train, y_train = train_data
    z_pred = np.empty((X_test.shape[0], num_boots))
    for boot_num in range(num_boots):
        X_, y_ = resample(X_train, y_train)
        y_predict, _ = model((X_, y_), X_test, lmb)
        z_pred[:, boot_num] = np.squeeze(y_predict)
    return z_pred

def cross_validation(kfold_num, X, y, model, lmb=0):
    kfold = KFold(n_splits=kfold_num)
    mse_ = []
    r2_ = []
    for train_inds, test_inds in kfold.split(X):
        X_train = X[train_inds]
        y_train = y[train_inds]
        X_test = X[test_inds]
        y_test = y[test_inds]

        # using the equation
        y_predict, _ = model((X_train, y_train), X_test, lmb)
        stats = statistics(y_test, y_predict, sampling='cv')
        mse_ += [stats['MSE']]
        r2_ += [stats['R2']]

    stat_record = pd.DataFrame({'MSE': mse_, 'R2': r2_})
    return stat_record
