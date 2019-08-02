import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from scipy.stats import multivariate_normal





#Create grid and multivariate normal
x = np.linspace(-10,10,50)
y = np.linspace(-10,10,50)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

mu_list = [[0,0],
           [3,2],
           [1,-5],
           [1,2] ]
sigma_list = [ [[10,0],[0,10]],
               [[2,0], [0,10]],
               [[10,4],[3,2]],
               [[2,1], [4,10]] ]

# Draw 3D multivariate Gaussian plots
fig = plt.figure(figsize=(18,12))
i = 0
for mu, sigma in zip(mu_list, sigma_list):
    rv = multivariate_normal(mu, sigma)
    i = i + 1
    ax = fig.add_subplot(2, 2, i, projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    ax.text2D(0.05, 0.95, "$\mu=$ %s\n$\Sigma=$ %s" % (mu,sigma), transform=ax.transAxes)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

plt.show()

