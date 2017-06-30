import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
import random
from matplotlib.offsetbox import AnchoredText

class cuda_thread():
    def __init__(self, thread_id, block_id, matrix, 
                 s, block_dim, grid_dim):
        self.thread_id = thread_id
        self.block_id = block_id
        self.matrix = matrix
        self.s = s
        self.i = thread_id
        self.j = block_id
        self.block_dim = block_dim
        self.grid_dim = grid_dim
        self.id = self.i * self.s + self.j;
        self.done = False
        self.started = False
        self.global_id = thread_id + block_id * block_dim

    def update(self):
        if self.id < self.s*self.s and self.j<self.s:
            self.matrix[self.id] = self.global_id
            self.i += self.block_dim
            if self.i>=self.s:
                self.i = self.i%self.s
                self.j = self.j + self.grid_dim
            self.id = self.i*self.s + self.j
        else:
            self.done = True


class image_generator():
    def __init__(self,block_dim, grid_dim, matrix,s, inplace=True, multicores=None):
        self.threads = []
        self.current_thread = 0
        self.inplace = inplace
        self.matrix = matrix
        self.i = 0
        if multicores:
            self.multicores = multicores
        else:
            self.multicores = block_dim*grid_dim
        for block_id in range(grid_dim):
            for thread_id in range(block_dim):
                thread = cuda_thread(thread_id, block_id, matrix,s, block_dim, grid_dim)
                self.threads.append(thread)
            self.all_done = np.zeros(len(self.threads), dtype=bool)
            self.next = True
        self.matrix_time = []

    # def calculateNextThread(self,current):
    #     thread = self.theads[]

    # def generate(self):
    #     print(self.i)
    #     self.i += 1
    #     if not self.next:
    #         return
    #     if not self.inplace:
    #         self.matrix_time.append(np.copy(self.matrix))
    #     for i in range(self.multicores):
    #         thread = self.threads[self.current_thread]
    #         thread.update()
    #         self.current_thread = 
    #         if thread.done:
    #             self.all_done[self.current_thread] = thread.done
    #             self.current_thread = (self.current_thread + 1) % len(self.threads)
    #     if np.all(self.all_done):
    #         self.next = False

    def generate(self):
        print(self.i)
        self.i += 1
        if not self.next:
            return
        if not self.inplace:
            self.matrix_time.append(np.copy(self.matrix))
        for i in range(self.multicores):
            thread = self.threads[self.current_thread]
            thread.update()
            if thread.done:
                self.all_done[self.current_thread] = thread.done
                self.current_thread = (self.current_thread + 1) % len(self.threads)
        if np.all(self.all_done):
            self.next = False

class image_generator2():
    def __init__(self,block_dim, grid_dim, matrix,s, warpsize):
        self.threads = []
        self.block_dim = block_dim
        self.grid_dim = grid_dim
        self.current_thread = 0
        self.matrix = matrix
        self.warpsize = warpsize
        self.i = 0
        for block_id in range(grid_dim):
            for thread_id in range(block_dim):
                thread = cuda_thread(thread_id, block_id, matrix,s, block_dim, grid_dim)
                self.threads.append(thread)
            self.all_done = np.zeros(len(self.threads), dtype=bool)
            self.next = True
        self.matrix_time = []
    
    def getThread(self,threadlist):
        if len(threadlist) > 0:
            t = threadlist.pop()
            if not t.done:
                return t
            else:
                return None
        else:
            return None

    def generate(self):
        print("RUN:",self.i)
        self.i += 1
        if not self.next:
            return
        # iteratre over blocks
        for i in range(self.grid_dim):
            block_threads = [thread for thread in self.threads if thread.block_id == i]
            block_threads = [thread for thread in block_threads if not thread.done]
            for w in range(self.warpsize):
                # chose thread to execute
                thread = self.getThread(block_threads)
                if thread:
                    thread.update()
                    print(thread.block_id, thread.thread_id, thread.global_id)
                else:
                    break
        #
        self.all_done = np.array([thread.done for thread in self.threads], dtype=bool)
        if np.all(self.all_done):
            self.next = False


def ani2():
    s = 10
    block_dim = 8
    grid_dim = 4 
    warpsize = 2
    num_threads = block_dim * grid_dim
    data = np.ones(s*s) * num_threads
    cmap = LinearSegmentedColormap.from_list('mycmap', ['green', 'white', 'darkgreen', 'black'])
    generator = image_generator2(block_dim, grid_dim, data, s, warpsize)

    def update(i):
        generator.generate()
        matrice.set_array(data.reshape((s,s)))

    fig, ax = plt.subplots()
    matrice = ax.matshow(data.reshape(s,s), cmap="Paired", vmin=0, vmax=num_threads)
    plt.colorbar(matrice)

    frames = (s * s) // grid_dim + 1
    frames = 20
    ########################################################################################
    ani = animation.FuncAnimation(fig, update, frames=10, interval=1000, repeat=False, blit=False)
    ########################################################################################

    ax.set_xlabel("Matrix Columns")
    ax.set_ylabel("Matrix Rows")
    text = "Gridsize: {}\nBlocksize: {}\nWarpsize: {}\nMPU: {}\nWarps\MPU: {}"
    text = text.format(grid_dim, block_dim, warpsize, grid_dim, 1)
    anchored_text = AnchoredText(text, loc=1)
    ax.add_artist(anchored_text)
    fig.suptitle("CUDA threads working on a matrix", fontsize=14)
    plt.tight_layout()

    # ani.save('basic_animation.mp4', fps=1)
    plt.show()

s = 1000
block_dim = 128
grid_dim = 64 
warpsize = 32
num_threads = block_dim * grid_dim
data = np.ones(s*s) * num_threads
cmap = LinearSegmentedColormap.from_list('mycmap', ["darkgreen",'green', 'white', 'darkgreen', "white", "limegreen", 'black'])
generator = image_generator2(block_dim, grid_dim, data, s, warpsize)
# generator = image_generator(block_dim, grid_dim, data, s)

def update(i):
    generator.generate()
    matrice.set_array(data.reshape((s,s)))

def createFigure(i):
    fig, ax = plt.subplots()
    ax.set_xlabel("Matrix Columns")
    ax.set_ylabel("Matrix Rows")
    text = "Gridsize: {}\nBlocksize: {}\nWarpsize: {}\nMPU: {}\nWarps\MPU: {}"
    text = text.format(grid_dim, block_dim, warpsize, grid_dim, 1)
    anchored_text = AnchoredText(text, loc=1)
    ax.add_artist(anchored_text)
    fig.suptitle("CUDA threads working on a matrix", fontsize=14)
    
    im = plt.imshow(data.reshape((s,s)), cmap="prism", vmin=0, vmax=num_threads)
    fig.colorbar(im)
    fig.savefig("/home/matthias/Pictures/ani3/prism{num:05d}.png".format(num=i))
    plt.close('all')

i = 1
while generator.next:
    createFigure(i)
    generator.generate()
    i += 1








def create_cmap(vmin,vmax):
    cmap = LinearSegmentedColormap.from_list('mycmap', [(0 / vmax, 'blue'),
                                             (1 / vmax, 'white'),
                                             (3 / vmax, 'red')])
    return cmap

# def update(i):
#     generator.generate()
#     matrice.set_array(data.reshape((s,s)))

def ani_matrix():
    s = 100
    block_dim = 24
    grid_dim = 8
    num_threads = block_dim * grid_dim
    num_multicores = block_dim
    data = np.ones(s*s) * num_threads
    cmap = LinearSegmentedColormap.from_list('mycmap', ['green', 'white', 'darkgreen', 'black'])
    generator = image_generator(block_dim, grid_dim, data, s, inplace=False, multicores=num_multicores)

    def update(i):
        generator.generate()
        matrice.set_array(data.reshape((s,s)))

    fig, ax = plt.subplots()
    matrice = ax.matshow(data.reshape(s,s), cmap=cmap, vmin=0, vmax=num_threads)
    plt.colorbar(matrice)

    frames = (s * s) // num_multicores + 1
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, repeat=False)
    plt.show()

    data = data.reshape((s,s))

# ani_matrix()

# s = 10
# block_dim = 3
# grid_dim = 4
# num_threads = block_dim * grid_dim
# num_multicores = block_dim
# data = np.ones(s*s) * num_threads
# generator = image_generator(block_dim, grid_dim, data, s, inplace=False, multicores=num_multicores)

# def update(i):
#     generator.generate()
#     matrice.set_array(data.reshape((s,s)))

# fig, ax = plt.subplots()
# matrice = ax.matshow(data.reshape(s,s), cmap="Paired", vmin=0, vmax=num_threads)
# plt.colorbar(matrice)

# frames = (s * s) // num_multicores + 10
# ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, repeat=False)
# plt.show()

# data = data.reshape((s,s))


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