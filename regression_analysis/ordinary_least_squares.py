"""script to calculate ordinary least squares"""

import create_data_franke as data
import basis_functionality as basis
import create_plots as own_plots
import prepare_data 

import numpy as np
import matplotlib.pyplot as plt


# To make results comparable set seed()
np.random.seed(2021)


def calculate_OLS(input_x1, input_x2, obs_var, order):
    """Calculate the response variable for ordinary least suqares."""
    # Get design matrix
    regress_obj = basis.Design_Matrix_2D(input_x1, input_x2, obs_var, order)
    D = regress_obj.make_design_matrix()
    A_inv = regress_obj.design_matrix_product_inverse()

    # Calculate parameter vector
    beta_OLS = np.dot(np.dot(A_inv, D.T), obs_var.flatten())

    # Calculate response variable
    resp_var_OLS = np.dot(D, beta_OLS).reshape(obs_var.shape[0], obs_var.shape[1])

    return beta_OLS, resp_var_OLS


if __name__ == "__main__":

    # Get data
    input_x1, input_x2, obs_var = data.generate_data(noisy=False, noise_variance=1, uniform=False, points=100)

    # Split data in test and train datasets
    x1_train, x2_train, x1_test, x2_test, y_train, y_test = prepare_data.split_test_train(input_x1, input_x2, obs_var,
                                                                                          train_fraction=0.8)

    # Fit model for different polynomial
    max_order = 2
    orders = range(1, max_order+1)
    MSE_train = np.empty([1, max_order])
    R2_train = np.empty([1, max_order])
    MSE_test = np.empty([1, max_order])
    R2_test = np.empty([1, max_order])

    for index, order in enumerate(orders):
        # Get OLS response variable and beta for train dataset
        beta_OLS_train, resp_var_OLS_train = calculate_OLS(x1_train, x2_train, y_train, order)

        # Calculate error evaluaters for the train dataset
        error_class = basis.Error_Measures(y_train, resp_var_OLS_train)
        MSE_train[:, index] = error_class.mean_squared_error()
        R2_train[:, index] = error_class.r2_score()

        # Get OLS response variable for test dataset
        regress_obj_test = basis.Design_Matrix_2D(x1_test, x2_test, y_test, order)
        D_test = regress_obj_test.make_design_matrix()
        resp_var_OLS_test = np.dot(D_test, beta_OLS_train).reshape(y_test.shape[0], y_test.shape[1])

        # Calculate error evaluaters for the test dataset
        error_class = basis.Error_Measures(y_test, resp_var_OLS_test)
        MSE_test[:, index] = error_class.mean_squared_error()
        R2_test[:, index] = error_class.r2_score()

    # Plot errors
    axes_1 = np.array(orders)
    args = (MSE_train, MSE_test, R2_train, R2_test)
    axes_2 = np.concatenate(args, axis=0)

    line_lab = ['MSE train', 'MSE test', 'R2 train', 'R2 test']

    own_plots.make_multi_line_plot(axes_1, axes_2[:2, :], line_lab[:2])
    own_plots.make_multi_line_plot(axes_1, axes_2[2:, :], line_lab[2:])

    """# Plot original data next to fitted data
    own_plots.make_3d_surface_plot(input_x1, input_x2, obs_var, "Plot of Franke function")
    own_plots.make_3d_surface_plot(input_x1, input_x2, resp_var_OLS, "Plot of fitted model") """
    plt.show() 