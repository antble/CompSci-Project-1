from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from src.models import ols_model, ols_model_skl
import numpy as np
import unittest 


class TestOLS(unittest.TestCase):
    # test y_predict of OLS
    def test_ols_model_ypredict(self):
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        train_data = (X_vander, y_true)
        y_predict, beta = ols_model(train_data, X_vander)
        self.assertTrue(np.allclose(y_predict, X_vander @ beta))

    # test beta calculation
    def test_ols_model_beta(self):
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        train_data = (X_vander, y_true)
        _, beta = ols_model(train_data, X_vander)
        self.assertTrue(np.allclose(true_beta, beta))

    # check scikit beta calculation
    def test_ols_with_scikit(self):
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        skl = LinearRegression(fit_intercept=False).fit(X_vander, y_true)
        beta = skl.coef_
        self.assertTrue(np.allclose(beta, true_beta))

    # check scikit y preiction
    def test_ols_with_scikit_beta(self):
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        skl = LinearRegression(fit_intercept=True).fit(X_vander, y_true)
        y_predict = skl.predict(X_vander)
        self.assertTrue(np.allclose(y_true, y_predict))

    # check scikit y output to the model y prediction
    def test_model_with_scikit(self):
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        train_data = (X_vander, y_true)
        y_predict_model, _ = ols_model(train_data, X_vander)
        skl = LinearRegression(fit_intercept=True).fit(X_vander, y_true)
        y_predict_skl = skl.predict(X_vander)
        self.assertTrue(np.allclose(y_predict_skl, y_predict_model))

    # check model prediction with scikit with data split
    def test_model_predict_split(self):
        np.random.seed(2021)
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X_vander, y_true, test_size=0.20)
        y_predict_own, _ = ols_model((X_train, y_train), X_test)
        y_predict_skl, _ = ols_model_skl((X_train, y_train), X_test)
        print(y_predict_own.shape, y_predict_skl.shape)
        print(y_predict_own, y_predict_skl)
        self.assertTrue(np.allclose(y_predict_own, y_predict_skl))

    # check model beta with scikit with data split
    def test_model_beta_split(self):
        np.random.seed(2021)
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X_vander, y_true, test_size=0.20)
        _, beta_own = ols_model((X_train, y_train), X_test)
        _, beta_skl = ols_model_skl((X_train, y_train), X_test)
        print(beta_own.shape, beta_skl.shape)
        print(beta_own, beta_skl)
        self.assertTrue(np.allclose(beta_own, beta_skl))


if __name__ == "__main__":
    unittest.main()
