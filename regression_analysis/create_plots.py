"""Collection of functions to make pretty plots."""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def make_multi_line_plot(x, y, line_lab):
    """Make a figure with multiple lines but same values for x."""
    num_lines = y.shape[0]
    for index in range(num_lines):
        plt.plot(x, y[index, :], label=line_lab[index])
    plt.legend()


def make_3d_surface_plot(x, y, z, title=None):
    """Create 3D surface plot of given input."""
    # Create figure and axes for plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if title:
        fig.suptitle(title)
