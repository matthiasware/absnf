import numpy as np
import matplotlib.pyplot as plt
import solve_data as sd

x_p_gtx, y_p_gtx = zip(*sd.gtx_modulus)
x_p_tesla, y_p_tesla = zip(*sd.tesla_modulus)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim(min(x_p_gtx), max(x_p_gtx))
ax1.set_xlabel("m=n=s")
ax1.set_ylabel("execution time / ms")
ax1.grid(True)

# TODO ADD TESLA
ax1.plot(x_p_gtx, y_p_gtx, label="GeForce GTX 780 3GB")
ax1.plot(x_p_tesla, y_p_tesla, label="Tesla P100-PCIE-16GB")

# ax2 = ax1.twiny()
# ax1Xs = ax1.get_xticks()
# x_gb = np.round(np.array([mem_eval(i,i,i, 8) for i in ax1Xs]) * 1e-9,decimals=2)

# ax2.set_xlabel("RAM / GB")
# ax2.set_xticks(ax1Xs)
# ax2.set_xbound(ax1.get_xbound())
# ax2.set_xticklabels(x_gb)
ax1.legend()

fig.suptitle("ABS-NF Solve with Modulus Algorithm", y=1.0, fontsize=12)
fig.tight_layout()
fig.savefig("solve_modulus.png")
plt.show()