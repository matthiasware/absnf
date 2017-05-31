import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def f(x1, x2):
	return max(0, x2**2 - max(0, x1))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-10.0, 10.0, 0.05)
X, Y = np.meshgrid(x, y)
Z = np.array([f(a,b) for a,b in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

ax.plot_surface(X, Y, Z)
plt.show()
fig.savefig('test2.png')