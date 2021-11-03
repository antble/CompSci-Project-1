"""script to calculate ordinary least squares"""

import create_data_franke as data
import numpy as np


# To make results comparable set seed()
np.random.seed(2021)

# Get data
input_x1, input_x2 = data.generate_input_values(uniform=False, stepsize=0.05)
obs_var = data.FrankeFunction(input_x1, input_x2, noisy=False)
obs_var = obs_var.reshape(len(obs_var), 1)

# Define design matrix
design_matrix = data.make_design_matrix(input_x1, input_x2, order=5)

# Perform Singular Value Decomposition U*S*V.T to calculate inverse of design matrix product (A*A.T)^-1
A = np.dot(design_matrix.T, design_matrix)
U, S, V_t = np.linalg.svd(A)
S_inverse = np.linalg.inv(np.diag(S))
A_inverse = np.dot(V_t.T, np.dot(S_inverse, U.T))

# Calculate parameter vector
beta = np.dot(np.dot(A_inverse, design_matrix.T), obs_var).flatten()

# Calculate response variable
resp_var = np.dot(design_matrix, beta)