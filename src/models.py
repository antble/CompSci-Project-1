import numpy as np 
from src.utils import statistics
from sklearn import linear_model

'''
OLS Regression
'''

def ols_model(train_data, X_test, *args):
    X_train, y_train = train_data
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    y_predict = X_test @ beta
    return y_predict, beta


def ols_model_skl(train_data, X_test, *args):
    X_train, y_train = train_data
    ols = linear_model.LinearRegression(fit_intercept=False)
    ols.fit(X_train, y_train)
    y_predict = ols.predict(X_test)
    return y_predict, ols.coef_


'''
Ridge regression
'''

def ridge_model(train_data, X_test, lmb=0):
    X_train, y_train = train_data
    p = (X_train.T @ X_train).shape
    identity_matrix = np.eye(p[0], p[1])
    ridge_beta = np.linalg.pinv(X_train.T @ X_train + lmb*identity_matrix) @ X_train.T @ y_train
    y_predict = X_test @ ridge_beta
    return y_predict, ridge_beta


def ridge_model_skl(train_data, X_test, lmb):
    X_train, y_train = train_data
    ridge = linear_model.Ridge(lmb, fit_intercept=False)
    ridge.fit(X_train, y_train)
    return ridge.predict(X_test), ridge.coef_


'''
LASSO Regression
'''

def lasso_model_skl(train_data, X_test, lmb=0):
    X_train, y_train = train_data
    lasso = linear_model.Lasso(lmb, fit_intercept=False, tol=1e-2)
    lasso.fit(X_train, y_train)
    return lasso.predict(X_test), lasso.coef_
