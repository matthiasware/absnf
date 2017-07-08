import numpy as np
import matplotlib.pyplot as plt
import eval_data as ed

def mem_eval(s,m,n, size=8):
  return (s+m+(s*n)+(s*s)+(m*n)+(m*s)+m+n+s+s)*size

x_p_gtx, up_p_gtx, ex_p_gtx, tot_p_gtx = zip(*ed.single_gtx780)
x_p_tesla, up_p_tesla, ex_p_tesla, tot_p_tesla = zip(*ed.single_tesla)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax3 = fig.add_subplot(212, sharex = ax1, sharey=ax1)
ax1.set_xlim(min(x_p_gtx), max(x_p_gtx))
ax3.set_xlabel("m=n=s")
ax1.set_ylabel("time / ms")
ax3.set_ylabel("time / ms")
ax1.grid(True)
ax3.grid(True)


ax1.plot(x_p_gtx, up_p_gtx, label="Dataupload GeForce GTX 780")
ax1.plot(x_p_gtx, ex_p_gtx, label="Runtime GeForce GTX 780")
ax1.plot(x_p_gtx, tot_p_gtx, label="Total GeForce GTX 780")


ax3.plot(x_p_tesla, up_p_tesla, label="Dataupload Tesla P100-PCIE-16GB")
ax3.plot(x_p_tesla, ex_p_tesla, label="Runtime Tesla P100-PCIE-16GB")
ax3.plot(x_p_tesla, tot_p_tesla, label="Total Tesla P100-PCIE-16GB")

# ADD MEMORY USAGE
ax2 = ax1.twiny()
ax1Xs = ax1.get_xticks()
x_gb = np.round(np.array([mem_eval(i,i,i, 8) for i in ax1Xs]) * 1e-9,decimals=2)

ax2.set_xlabel("RAM / GB")
ax2.set_xticks(ax1Xs)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(x_gb)
ax1.legend()
ax3.legend()

fig.suptitle("ABS-NF Evaluation - Memory Transferation Costs", y=1.0, fontsize=12)
fig.tight_layout()
fig.savefig("eval_memory.png")
plt.show()
