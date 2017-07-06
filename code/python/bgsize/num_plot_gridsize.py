import numpy as np
import matplotlib.pyplot as plt
import numerics_gridblock_data as data

x_gs_1, y_gs_1 = zip(*data.bs_max)
x = np.array(x_gs_1)
y = np.array(y_gs_1)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel("blocksize")
ax1.set_ylabel("time / ms")
ax1.grid(True)

ax1.plot(x[0:150], y[0:150], label="gridsize where blocksize = MaxThreadsPerBlock:")

fig.suptitle("choosing the gridsize", fontsize=14)
# ax1.legend()

ax1.axvline(12, color="red", linestyle='dotted', label="MultiProcessorCount")

ax1.legend()
plt.show()