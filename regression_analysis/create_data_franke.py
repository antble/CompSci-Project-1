"""script to create data file based on the Franke function"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from datetime import datetime
from random import random, seed


def FrankeFunction(x, y, noisy = False):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)

    output = term1 + term2 + term3 + term4

    if noisy:
        noise = np.random.normal(0,0.2,len(x)**2).reshape(len(x), len(x))
        output = output + noise

    return output


def make_plot(x,y,z):
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


def save_data(x,y,z):
    data = pd.DataFrame(data=[x,y,z]).T
    today = datetime.today().strftime('%Y%m%d')
    filename = "data\\_" + str(today) + "FrankeFunction.csv"
    pd.DataFrame(data).to_csv(filename)


if __name__=="__main__":
    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    # Using numpy’s meshgrid
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Call Franke function
    z = FrankeFunction(x_mesh, y_mesh, True)

    # Create plot of Franke function
    make_plot(x_mesh,y_mesh,z)

    # Save data in csv
    #save_data(x_mesh[0,:],y_mesh[:,0],z)
    
