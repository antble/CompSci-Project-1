from sklearn.model_selection import train_test_split
from src.models import lasso_model
import numpy as np
import unittest 


class TestLasso(unittest.TestCase):
    def test_lasso