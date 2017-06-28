import numpy as np
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/10514315/how-to-add-a-second-x-axis-in-matplotlib
def mem_eval(s,m,n, size=8):
  return (s+m+(s*n)+(s*s)+(m*n)+(m*s)+m+n+s+s)*size

# Abgeschmiert bei 
# 8192768000 bytes = 8.1 GB
serial_eval = [
		[1000, 0.1405057709198445],
		[2000, 0.012646890943869948],
		[3000, 0.02446344494819641],
		[4000, 0.040404943050816655],
		[5000, 0.06675698584876955],
		[6000, 0.0903540609870106],
		[7000, 0.12007018295116723],
		[8000, 0.1526842899620533],
		[9000, 0.18933409894816577],
		[10000, 0.23314318992197514],
		[11000, 0.2677303079981357],
		[12000, 0.31692088884301484],
		[13000, 0.3664737830404192],
		[14000, 0.43748116702772677],
		[15000, 1.5990647720173001]]
# Abgeschmiert bei
# (3200480000 = 3.2 GB)
# Max is 3167092736 = 3.1 GB)
# s, upload, exec, download, total
p_eval_small = [
		[1000,7,13,21],
		[2000,29,34.,63],
		[3000,64,62,127],
		[4000,114,97,212],
		[5000,181,126,303],
		[6000,245,183,424],
		[7000,347,218,566],
		[8000,449,295,744],
		[9000,518,353,872],
		[10000,542,422,965]]


x_serial, y_serial = zip(*serial_eval)
x_serial = np.array(x_serial)
y_serial = np.array(y_serial) * 1000

x_p_bs, up_p_bs, run_p_bs, t_p_bs = zip(*p_eval_small)

# x_gb = np.round(np.array([mem_eval(i,i,i, 8) for i in x_serial]) * 1e-9,decimals=2)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim(0, max(x_serial))
ax1.set_xlabel("m=n=s")
ax1.set_ylabel("time / ms")
ax1.grid(True)

ax1.plot(x_serial, y_serial, label="numpy runtime")
ax1.plot(x_p_bs, t_p_bs, label="parallel total")
ax1.plot(x_p_bs, up_p_bs, label="parallel upload")
ax1.plot(x_p_bs, run_p_bs, label="parallel runtime")

ax2 = ax1.twiny()

ax1Xs = ax1.get_xticks()
x_gb = np.round(np.array([mem_eval(i,i,i, 8) for i in ax1Xs]) * 1e-9,decimals=2)

ax2.set_xlabel("memory / GB")
ax2.set_xticks(ax1Xs)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(x_gb)
# ax2.set_xlim(min(x_gb), max(x_gb))
# ax2.set_xticks([10, 30, 40])
# ax2.set_xticklabels(x_gb)


# ax1.ylabel('time / ms')
# plt.xlabel('m=n=s')
ax1.legend()

plt.show()