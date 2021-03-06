import numpy as np
import matplotlib.pyplot as plt
import eval_data as ed

def mem_eval(s,m,n, size=8):
  return (s+m+(s*n)+(s*s)+(m*n)+(m*s)+m+n+s+s)*size

# Eval single 

x_s_intel, y_s_intel = zip(*ed.multiple_serial_1000)
x_p_gtx, y_p_gtx = zip(*ed.multiple_gtx_1000)
x_p_tesla, y_p_tesla = zip(*ed.multiple_tesla_1000)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim(min(x_s_intel), max(x_s_intel))
ax1.set_xlabel("m=n=s")
ax1.set_ylabel("execution time / ms")
ax1.grid(True)

# TODO ADD TESLA
ax1.plot(x_s_intel, y_s_intel, label="numpy : i5-2500K CPU @ 3.30GHz")
ax1.plot(x_p_gtx, y_p_gtx, label="parallel: GeForce GTX 780 3GB")
ax1.plot(x_p_tesla, y_p_tesla, label="parallel: Tesla P100-PCIE-16GB")

ax2 = ax1.twiny()
ax1Xs = ax1.get_xticks()
x_gb = np.round(np.array([mem_eval(i,i,i, 8) for i in ax1Xs]) * 1e-9,decimals=2)

ax2.set_xlabel("RAM / GB")
ax2.set_xticks(ax1Xs)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(x_gb)
ax1.legend()

fig.suptitle("ABS-NF Evaluation - 1000 Repetitions", y=1.0, fontsize=12)
fig.tight_layout()
fig.savefig("eval_mult_repetition.png")
plt.show()