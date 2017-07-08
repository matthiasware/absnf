import numpy as np
import matplotlib.pyplot as plt
import gradient_data as gd

def mem_grad(m,n,s,size):
    s_bytes = (s+m+s*n+s*s+m*n+m*s+s+m+m*n+s*s+s*s+m*s)*size
    s_gb = s_bytes * 1e-9
    return s_gb

# Eval single 

x_s_intel, y_s_intel = zip(*gd.single_serial)
x_p_gtx, up_p_gtx, ex_p_gtx = zip(*gd.single_gtx)
x_p_tesla, up_p_tesla, ex_p_tesla = zip(*gd.single_tesla)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim(min(x_s_intel), max(x_s_intel))
ax1.set_xlabel("m=n=s")
ax1.set_ylabel("execution time / ms")
ax1.grid(True)

x_s_intel, y_s_intel = zip(*ed.single_serial)

# TODO ADD TESLA
ax1.plot(x_s_intel, y_s_intel, label="numpy : i5-2500K CPU @ 3.30GHz")
ax1.plot(x_p_gtx, tot_p_gtx, label="parallel: GeForce GTX 780")
ax1.plot(x_p_tesla, tot_p_tesla, label="parallel: Tesla P100-PCIE-16GB")

ax2 = ax1.twiny()
ax1Xs = ax1.get_xticks()
x_gb = np.round(np.array([mem_eval(i,i,i, 8) for i in ax1Xs]) * 1e-9,decimals=2)

ax2.set_xlabel("RAM / GB")
ax2.set_xticks(ax1Xs)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(x_gb)
ax1.legend()

fig.suptitle("ABS-NF Gradient - Serial vs Parallel Implementation", y=1.0, fontsize=12)
fig.tight_layout()
fig.savefig("gradient_single_repetition.png")
plt.show()