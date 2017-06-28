import numpy as np
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/10514315/how-to-add-a-second-x-axis-in-matplotlib
def mem_grad(m,n,s,size):
    s_bytes = (s+m+s*n+s*s+m*n+m*s+s+m+m*n+s*s+s*s+m*s)*size
    s_gb = s_bytes * 1e-9
    return s_gb

parallel = [[1000,3182],
            [2000,22617],
            [3000,75455],
            [4000,177386]]

serial = [[1000,18.974418],
          [2000,38.2195062469982],
          [3000,451.2476991170115],
          [4000,1065.3939741420036]]

x_s, y_s = zip(*serial)
y_s = np.array(y_s)*1000
x_p, y_p = zip(*parallel)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim(1000, max(x_s))
ax1.set_xlabel("m=n=s")
ax1.set_ylabel("time / ms")
ax1.grid(True)

ax1.plot(x_s, y_s, label="numpy : i5-2500K CPU @ 3.30GHz")
ax1.plot(x_p, y_p, label="gpu: GeForce GTX 780")

ax2 = ax1.twiny()
ax1Xs = ax1.get_xticks()
x_gb = np.round(np.array([mem_grad(i,i,i,8) for i in ax1Xs]),decimals=3)

ax2.set_xlabel("used global memory / GB")
ax2.set_xticks(ax1Xs)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(x_gb)
ax1.legend()

fig.suptitle("Gradient 100 Executions - Serial vs Parallel", fontsize=14)
fig.tight_layout()
plt.show()