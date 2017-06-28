import numpy as np
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/10514315/how-to-add-a-second-x-axis-in-matplotlib
def mem_eval(s,m,n, size=8):
  return (s+m+(s*n)+(s*s)+(m*n)+(m*s)+m+n+s+s)*size


# gridblock
parallel = [[1000, 11342],
			[2000, 28694],
			[3000, 52892],
			[4000, 82162],
			[5000, 119479],
			[6000, 161613],
			[7000, 211120],
			[8000, 264746],
			[9000, 331517]]

serial =   [[1000, 4.168],
			[2000, 12.472],
			[3000, 24.3],
			[4000, 40.25],
			[5000, 65.85],
			[6000, 89.130],
			[7000, 118.216],
			[8000, 151,61],
			[9000, 186.05]]

x_s, y_s = zip(*serial)
y_s = np.array(y_s)*1000
x_p, y_p = zip(*parallel)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim(0, max(x_s))
ax1.set_xlabel("m=n=s")
ax1.set_ylabel("time / ms")
ax1.grid(True)

ax1.plot(x_s, y_s, label="numpy : i5-2500K CPU @ 3.30GHz")
ax1.plot(x_p, y_p, label="parallel: GeForce GTX 780")

ax2 = ax1.twiny()
ax1Xs = ax1.get_xticks()
x_gb = np.round(np.array([mem_eval(i,i,i, 8) for i in ax1Xs]) * 1e-9,decimals=2)

ax2.set_xlabel("memory / GB")
ax2.set_xticks(ax1Xs)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(x_gb)
ax1.legend()

fig.suptitle("Eval 1000 Repetitions - Serial and Parallel Implementation", fontsize=16)
fig.tight_layout()
plt.show()