import numpy as np
import matplotlib.pyplot as plt
import numerics_gridblock_data as data
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

df = pd.read_csv("data.csv")

d = df.values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(d.T[0], d.T[1], d.T[2], cmap=cm.viridis, linewidth=0)
fig.colorbar(surf)


ax.set_xlabel("blocksize")
ax.set_ylabel("gridsize")
ax.set_zlabel("time/ms")
ax.grid(True)

fig.suptitle("choosing blocksize and gridsize", fontsize=14)
fig.tight_layout()

plt.show()