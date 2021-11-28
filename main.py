from src.utils import create_data, create_design_matrix
from sklearn.model_selection import train_test_split
from src.models import ols_model, ridge_model, lasso_model_skl
from src.utils import statistics
from src.sampling import bootstrap, cross_validation
import numpy as np 
import pandas as pd

'''
param: deg degrees of the polynomial
param: N number of points in linspace
param: noise_var tuner of the added noise
'''


# --------- PART (a) code starts here --------------------
np.random.seed(2021)

deg = 5
N = 100
noise_var = 0.1
num_degrees = 10


for deg in range(1, num_degrees):
    x1, y1, x, y, z = create_data(N, noise_var)
    z = z[:, np.newaxis]
    X = create_design_matrix(x.flatten(), y.flatten(), deg)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.10, random_state=0)
    train_data = (X_train, z_train)
    test_data = (X_test, z_test)
    z_predict, _ = ols_model(train_data, X_test)
    stats = statistics(z_test, z_predict)

    # print(stats)

# --------- PART (b) code starts here --------------------


x1, y1, x, y, z = create_data(N, noise_var)
z = z[:, np.newaxis]
X = create_design_matrix(x.flatten(), y.flatten(), deg)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.10, random_state=0)
train_data = (X_train, z_train)
test_data = (X_test, z_test)

# OLS + bootstrap sampling
n_bootstrap = 3
zpred_boots_ols = bootstrap(n_bootstrap, train_data, X_test, ols_model)
stats = statistics(z_test, zpred_boots_ols, sampling=True)
print(stats)
df = pd.DataFrame(zpred_boots_ols)
print(df.head())

# --------- PART (c) code starts here --------------------
kfold = 10
stat_record = cross_validation(kfold, X, z, ols_model, lmb=1)
print(stat_record.mean(axis=0))

print(stat_record)


# --------- PART (d) code starts here --------------------
# Ridge regression
zpred_ridge, beta_ridge = ridge_model(train_data, X_test, lmb=0)
stats = statistics(z_test, zpred_ridge, sampling=True)
print(stats)

# Ridge regression + bootstrap sampling
zpred_boots_ridge = bootstrap(n_bootstrap, train_data, X_test, ridge_model)
stats = statistics(z_test, zpred_boots_ridge, sampling=True)
print(stats)
df = pd.DataFrame(zpred_boots_ridge)
print(df.head())


# Ridge regression + cross-validation sampling
kfold = 6
stat_record = cross_validation(kfold, X, z, ridge_model, lmb=5)
print(stat_record)


# --------- PART (e) code starts here --------------------
# Lasso regression + bootstrap sampling
zpred_boots_lasso = bootstrap(n_bootstrap, train_data, X_test, lasso_model_skl, lmb=0.001)
stats = statistics(z_test, zpred_boots_lasso, sampling=True)
print(stats)
df = pd.DataFrame(zpred_boots_lasso)
print(df.head())
