import numpy as np

def findMSE(y_data, y_fit):
    n = len(y_data)
    return np.sum((y_data-y_fit)**2)/n

def findR2(y_data, y_fit):
    num = np.sum((y_data-y_fit)**2)
    den = np.sum((y_data-np.mean(y_data))**2)
    return 1.0-(num/den)