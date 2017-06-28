import numpy as np
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/10514315/how-to-add-a-second-x-axis-in-matplotlib
def mem_grad(m,n,s,size):
    s_bytes = (s+m+s*n+s*s+m*n+m*s+s+m+m*n+s*s+s*s+m*s)*size
    s_gb = s_bytes * 1e-9
    return s_gb

parallel = [[1000,6,36],
            [2000,35,264],
            [3000,50,714],
            [4000,88,1691],
            [5000,137,3338],
            [6000,197,5601],
            [7000,276,1905]]

serial = [[1000,1.0328],
          [2000, 1.3284],
          [3000, 4.6364],
          [4000, 10.2069],
          [5000, 19.5306],
          [6000, 35.0542],
          [7000, 55.0430]]

x_s, y_s = zip(*serial)
y_s = np.array(y_s)*1000
x_p, mem_p, y_p = zip(*parallel)
tot_p = np.array(mem_p) + np.array(y_p)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim(1000, max(x_s))
ax1.set_xlabel("m=n=s")
ax1.set_ylabel("time / ms")
ax1.grid(True)

ax1.plot(x_s, y_s, label="numpy : i5-2500K CPU @ 3.30GHz")
ax1.plot(x_p, y_p, label="gpu execution: GeForce GTX 780")
ax1.plot(x_p, mem_p, label="gpu data upload: GeForce GTX 780")
ax1.plot(x_p, tot_p, label="gpu total: GeForce GTX 780")

ax2 = ax1.twiny()
ax1Xs = ax1.get_xticks()
# x_b, x_gb = 
x_gb = np.round(np.array([mem_grad(i,i,i,8) for i in ax1Xs]),decimals=3)

ax2.set_xlabel("used global memory / GB")
ax2.set_xticks(ax1Xs)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(x_gb)
ax1.legend()

fig.suptitle("Gradient Single Execution - Serial vs Parallel", fontsize=14)
fig.tight_layout()
plt.show()