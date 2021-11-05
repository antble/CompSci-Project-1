"""Collection of functions to prepare data for further analysis."""
import numpy as np
from sklearn.model_selection import train_test_split


def split_test_train(x1, x2, y, train_fraction=0.8):
    """Split data into train and test datasets."""
    # Use scikit learn to split data
    in_train, in_test, y_train, y_test = train_test_split(np.hstack([x1, x2]), y, train_size=train_fraction)

    # Split in_train and in_test in x1 and x2 again
    x1_train = in_train[:, 0:x1.shape[0]]
    x2_train = in_train[:, x1.shape[0]:]

    x1_test = in_test[:, 0:x1.shape[0]]
    x2_test = in_test[:, x1.shape[0]:]

    return x1_train, x2_train, x1_test, x2_test, y_train, y_test
