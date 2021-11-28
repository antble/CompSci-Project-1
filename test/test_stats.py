import unittest 
from src.utils import statistics, FrankeFunction
from src.utils import create_design_matrix, create_data
from src.utils import min_max_scaling 
from sklearn.preprocessing import MinMaxScaler
import numpy.polynomial.polynomial as poly
from src.models import ols_model, ols_model_skl
from sklearn.model_selection import train_test_split
import numpy as np 


class TestUtils(unittest.TestCase):
    def test_frankefunction(self):
        # value from akima R library
        frankeR = 0.7664206
        frankeP = FrankeFunction(0, 0)
        self.assertAlmostEqual(frankeR, frankeP)

    '''
    link: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    '''
    def test_min_max_scaling(self):
        data = np.array([[100, 0.001],
                        [8, 0.05],
                        [50, 0.005],
                        [88, 0.07],
                        [4, 0.1]])
        scaler = MinMaxScaler()
        scaled_data_skl = scaler.fit_transform(data)
        scaled_data_own = min_max_scaling(data, 0, 1)
        self.assertTrue(np.allclose(scaled_data_skl, scaled_data_own))

    # def test_create_design_matrix(self):
    #     x = np.array([1, 3, 5, 7])
    #     y = np.array([2, 4, 6, 8])
    #     X_skl = poly.polyvander2d(x, y, [2, 2])
    #     X_own = create_design_matrix(x, y,2) 
    #     print(X_skl)
    #     print(X_own)
    #     self.assertTrue(np.allclose(X_own, X_skl))


    def test_mse_calculation(self):
        np.random.seed(2021)
        x = np.linspace(0, 1, 11)
        X_vander = np.flip(np.vander(x, 3), axis=1)
        true_beta = [2, 0.5, 3.7]
        noise = 0.1 * np.random.normal(size=len(x))
        y_true = np.sum(np.array([x**p*b for p, b in enumerate(true_beta)]), axis=0) + noise
        X_train, X_test, y_train, y_test = (X_vander, X_vander, y_true, y_true)
        assert np.array_equal(X_train, X_test)
        assert np.array_equal(y_train, y_test)
        y_predict_own, beta_own = ols_model((X_train, y_train), X_test)
        y_predict_skl, beta_skl = ols_model_skl((X_train, y_train), X_test)
        print(y_predict_own.shape, y_predict_skl.shape)
        print(y_predict_own, y_predict_skl)
        print(beta_skl, beta_own)
        MSE_calc = 0.00411363461744314140
        stats_skl = statistics(y_test, y_predict_skl)
        stats_own = statistics(y_test, y_predict_own)
        print(stats_skl['MSE'], stats_own['MSE'])
        np.testing.assert_almost_equal(stats_skl['MSE'], stats_own['MSE'])
        self.assertAlmostEqual(MSE_calc, stats_skl['MSE'])