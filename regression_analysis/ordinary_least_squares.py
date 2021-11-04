"""script to calculate ordinary least squares"""

import create_data_franke as data
import basis_functionality as basis
import numpy as np


# To make results comparable set seed()
np.random.seed(2021)

# Get data
input_x1, input_x2, obs_var = data.generate_data(noisy=False, noise_variance=1, uniform=False, points=20)
# Get design matrix
regress_obj = basis.Design_Matrix_2D(input_x1, input_x2, obs_var, 2)
D = regress_obj.make_design_matrix()
A_inv = regress_obj.design_matrix_product_inverse()

print(A_inv.shape)
print(D.shape)
print(obs_var.shape)
# Calculate parameter vector
beta_OLS = np.dot(np.dot(A_inv, D.T), obs_var.flatten())

# Calculate response variable
resp_var_OLS = np.dot(D, beta_OLS)
