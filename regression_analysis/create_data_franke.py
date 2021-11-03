"""script to create data file based on the Franke function"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def generate_input_values(uniform=False, stepsize=0.05):
    """Generate values which will be used as input for the Franke function."""
    if uniform:
        x = np.random.uniform(0, 1, int(1/stepsize))
        y = np.random.uniform(0, 1, int(1/stepsize))
    else:
        x = np.arange(0, 1, stepsize)
        y = np.arange(0, 1, stepsize)
    return x, y


def FrankeFunction(x, y, noisy=False):
    """Define Franke function with or without additive noise."""
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)

    output = term1 + term2 + term3 + term4

    if noisy:
        noise = np.random.normal(0, 0.2, len(x)**2).reshape(len(x), len(x))
        output = output + noise

    return output


def make_design_matrix(x, y, order):
    """Define design matrix D with intercept for given input. Order is the order of the polynomial which will be
    fitted."""
    # Set values of first column of design matrix to 1
    D = np.repeat(1, len(x)).reshape(len(x), 1)
    # reshape x and y to correct dimensions
    x = x.reshape(len(x), 1)
    y = y.reshape(len(y), 1)
    # D's columns are defined as [intercept, x, y, x**i, y**i] for i in  1:order
    for power in range(1, order+1):
        D = np.append(D, x**power, axis=-1)
        D = np.append(D, y**power, axis=-1)
    return D


def make_plot(x, y, z):
    """Create 3D surface plot of given input."""
    # Create figure and axes for plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # Display plot
    plt.show()


if __name__ == "__main__":
    # To make results comparable the seed() method is used
    np.random.seed(2021)

    # Generate input for Franke function
    x, y = generate_input_values(uniform=True)
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Call Franke function
    z = FrankeFunction(x_mesh, y_mesh, True)

    # Create plot of Franke function
    make_plot(x_mesh, y_mesh, z)

    # Create design matrix
    design = make_design_matrix(x, y, 2)
