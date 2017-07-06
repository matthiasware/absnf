# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.ndimage.filters import convolve

# def pcolor_all(X, Y, C, **kwargs):
#     X = np.concatenate([X[0:1,:], X], axis=0)
#     X = np.concatenate([X[:,0:1], X], axis=1)

#     Y = np.concatenate([Y[0:1,:], Y], axis=0)
#     Y = np.concatenate([Y[:,0:1], Y], axis=1)

#     X = convolve(X, [[1,1],[1,1]])/4
#     Y = convolve(Y, [[1,1],[1,1]])/4

#     plt.pcolor(X, Y, C, **kwargs)

# X, Y = np.meshgrid(
#     [-1,-0.5,0,0.5,1],
#     [-2,-1,0,1,2])

# C = X**2-Y**2

# plt.figure(figsize=(4,4))

# pcolor_all(X, Y, C, cmap='gray')

# plt.savefig('plot.png')
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# def data_gen(t=0):
#     cnt = 0
#     while cnt < 1000:
#         cnt += 1
#         t += 0.1
#         yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)


# def init():
#     ax.set_ylim(-1.1, 1.1)
#     ax.set_xlim(0, 10)
#     del xdata[:]
#     del ydata[:]
#     line.set_data(xdata, ydata)
#     return line,

# fig, ax = plt.subplots()
# line, = ax.plot([], [], lw=2)
# ax.grid()
# xdata, ydata = [], []


# def run(data):
#     # update the data
#     t, y = data
#     xdata.append(t)
#     ydata.append(y)
#     xmin, xmax = ax.get_xlim()

#     if t >= xmax:
#         ax.set_xlim(xmin, 2*xmax)
#         ax.figure.canvas.draw()
#     line.set_data(xdata, ydata)

#     return line,

# ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10,
#                               repeat=False, init_func=init)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# fig = plt.figure()


# def f(x, y):
#     return np.sin(x) + np.cos(y)

# x = np.linspace(0, 2 * np.pi, 120)
# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# im = plt.imshow(f(x, y), animated=True)


# def updatefig(*args):
#     global x, y
#     x += np.pi / 15.
#     y += np.pi / 20.
#     im.set_array(f(x, y))
#     return im,

# ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
# plt.show()


# import sys
# try:
#     subprocess.check_call(['mencoder'])
# except subprocess.CalledProcessError:
#     print("mencoder command was found")
#     pass # mencoder is found, but returns non-zero exit as expected
#     # This is a quick and dirty check; it leaves some spurious output
#     # for the user to puzzle over.
# except OSError:
#     print(not_found_msg)
#     sys.exit("quitting\n")

# WORKS
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# L = 10               # Number of lanes
# C = 10               # Number of cells in road
# fig, ax = plt.subplots()    # Create figure
# a = np.zeros((L,C))     # Create zero array of dimensions L,C
# count = 0           # Used for loop
# x = 0               # Car position

# def animate(c):
#     global count, x, i
#     if x == C-1:        # If in last cell
#         a[L-1,C-1] = 0      # 0 last cell
#         a[L-1,0] = 1        # 1 first cell
#         x = 0           # Start iterating from 0
#     else:
#         if count == 0:      # On 1st iteration, put car @ a[0]
#             a[L-1,0] = 1
#             count += 1
#         else:
#             a[L-1,x] = 0    # 0 previous cell
#             x += 1
#             a[L-1,x] = 1    # 1 in current cell

#     # print(a)
#     ax.imshow(a, cmap=plt.cm.plasma, interpolation='nearest')

# ani = animation.FuncAnimation(fig, animate, interval=100, repeat=False, frames=20)
# # ani.save('ani2.mpeg', writer="ffmpeg")
# ani.save('basic_animation.mp4', fps=10)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

M=np.array([[0,0,100,100,100,100,100,100,300,300,300,300,300,300,500,500,500,500,500,500,1000,1000,1000,1000] for i in range(0,20)]) 

def update(i):
    M[7,i] = 1000
    M[19-i,10] = 500
    matrice.set_array(M)

fig, ax = plt.subplots()
matrice = ax.matshow(M)
plt.colorbar(matrice)

ani = animation.FuncAnimation(fig, update, frames=19, interval=500)
plt.show()