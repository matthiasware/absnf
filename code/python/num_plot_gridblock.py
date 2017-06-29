import numpy as np
import matplotlib.pyplot as plt
import numerics_gridblock_data as data

x_gs_1, y_gs_1 = zip(*data.d_gs_1)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel("blocksize")
ax1.set_ylabel("time / ms")
ax1.grid(True)

ax1.plot(x_gs_1, y_gs_1, label="blocksize where gridsize = MPC")

fig.suptitle("choosing the blocksize", fontsize=14)
# ax1.legend()

ax1.axvline(32, color="red", linestyle='dotted', label="warpsize")

ax1.legend()
plt.show()