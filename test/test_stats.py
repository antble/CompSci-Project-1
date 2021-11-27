import unittest 
from src.utils import statistics, FrankeFunction
from src.utils import create_design_matrix, create_data
from src.utils import min_max_scaling 
from sklearn.preprocessing import MinMaxScaler
import numpy.polynomial.polynomial as poly
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


