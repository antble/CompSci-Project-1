import unittest 
import numpy as np 
from src.models import ridge_model, ridge_model_skl
from sklearn.model_selection import train_test_split

class TestRidge(unittest.TestCase):
    # special case lmb=0
    def test_ridge_model_case0(self):
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        train_data = (X_vander, y_true)
        y_predict, beta = ridge_model(train_data, X_vander, lmb=0)
        self.assertTrue(np.allclose(y_predict, X_vander @ beta))

    # case lmb !=0
    def test_ridge_model_case1(self):
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        train_data = (X_vander, y_true)
        y_predict, beta = ridge_model(train_data, X_vander, lmb=100)
        self.assertTrue(np.allclose(y_predict, X_vander @ beta))

    # test beta calculation
    def test_ridge_model_beta(self):
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        train_data = (X_vander, y_true)
        _, beta = ridge_model(train_data, X_vander, lmb=0)
        # print(true_beta, beta)
        self.assertTrue(np.allclose(true_beta, beta))

    # compare own ridge with scikit
    def test_ridge_with_scikit(self):
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        train_data = (X_vander, y_true)
        y_predict_model, _ = ridge_model(train_data, X_vander, lmb=0.1)
        y_predict_skl, _ = ridge_model_skl(train_data, X_vander, lmb=0.1)
        self.assertTrue(np.allclose(y_predict_skl, y_predict_model))

    # check model prediction with scikit with data split
    def test_model_predict_split(self):
        np.random.seed(2021)
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X_vander, y_true, test_size=0.20)
        y_predict_own, _ = ridge_model((X_train, y_train), X_test, lmb=0.1)
        y_predict_skl, _ = ridge_model_skl((X_train, y_train), X_test, lmb=0.1)
        print(y_predict_own.shape, y_predict_skl.shape)
        print(y_predict_own, y_predict_skl)
        self.assertTrue(np.allclose(y_predict_own, y_predict_skl))

    # check beta with scikit beta with split
    def test_model_beta_split(self):
        np.random.seed(2021)
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X_vander, y_true, test_size=0.20)
        _, beta_own = ridge_model((X_train, y_train), X_test, lmb=0.1)
        _, beta_skl = ridge_model_skl((X_train, y_train), X_test, lmb=0.1)
        print(beta_own.shape, beta_skl.shape)
        print(beta_own, beta_skl)
        self.assertTrue(np.allclose(beta_own, beta_skl))



