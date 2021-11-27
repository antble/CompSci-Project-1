from src.utils import create_data, create_design_matrix
from sklearn.model_selection import train_test_split
from src.models import ols_model
from src.utils import statistics
import numpy as np 

'''
param: deg degrees of the polynomial
param: N number of points in linspace
param: noise_var tuner of the added noise
'''
np.random.seed(2021)

deg = 5
N = 100
noise_var = 0.0

x, y, z = create_data(N, noise_var)

X = create_design_matrix(x, y, deg)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.1)
train_data = (X_train, z_train)
test_data = (X_test, z_test)

z_predict, _ = ols_model(train_data, X_test)
stats = statistics(z_test, z_predict)

print(stats)
