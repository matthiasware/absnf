import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
import random
from matplotlib.offsetbox import AnchoredText


class CudaThread():
    def __init__(self, thread_id, block_id, matrix, 
                 s, block_dim, grid_dim, verbose=False):
        self.thread_id = thread_id
        self.verbose = verbose
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

    def execute(self):
        if self.verbose:
            print("Execute Thread ", self.global_id, " Block: ", self.block_id)
        if self.id < self.s*self.s and self.j<self.s:
            self.matrix[self.id] = self.global_id
            self.i += self.block_dim
            if self.i>=self.s:
                self.i = self.i%self.s
                self.j = self.j + self.grid_dim
            self.id = self.i*self.s + self.j
        else:
            self.done = True

class Warp():
    def __init__(self, warp_id, num_threads_per_warp, verbose=False):
        if verbose:
            print("Created Warp Unit: ", warp_id)
        self.id = warp_id
        self.num_threads_per_warp = num_threads_per_warp
        self.done = True
        self.threads = []
        self.verbose = verbose

    def setThreads(self, threads):
        self.threads = threads
        if self.verbose:
            print("Warp ", self.id, " set threads ", len(threads))
        self.checkWork()

    def execute(self):
        if not self.done:
            print("Warp ", self.id, "executing threads!")
            for thread in self.threads:
                if not thread.done:
                    thread.execute()
            self.checkWork()

        else:
            print("warp ", self.id, " done!")

    def checkWork(self):
        self.done = np.all(np.array([thread.done for thread in self.threads], dtype=bool))


class MPU():
    def __init__(self, mpu_id, num_warps_per_mpu, num_threads_per_warp, verbose=False):
        if verbose:
            print("Created MPU ", mpu_id)
        self.thread_class = None
        self.id = mpu_id
        self.num_warps_per_mpu = num_warps_per_mpu
        self.num_threads_per_warp = num_threads_per_warp
        self.threads = []
        self.done = True
        self.warps = []
        self.verbose = verbose
        for warp_id in range(num_warps_per_mpu):
            warp = Warp(warp_id, num_threads_per_warp, verbose)
            self.warps.append(warp)

    # creates all the threads for the  block block_id
    def create_threads(self, thread_class, block_id, block_dim, grid_dim, matrix, s):
        self.thread_class = thread_class
        self.threads = []
        for thread_id in range(block_dim):
            thread = self.thread_class(thread_id, block_id, matrix, s, block_dim, grid_dim, self.verbose)
            self.threads.append(thread)
        random.shuffle(self.threads)
        if self.verbose:
            print("MPU: ", self.id, "creating threads for block: ", block_id, " : ", len(self.threads))
        self.checkWork()

    def execute(self):
        if not self.done:
            threads = [thread for thread in self.threads if not thread.done]
            for warp in self.warps:
                if warp.done:
                    warp_threads = []
                    for i in range(warp.num_threads_per_warp):
                        if threads:
                            thread = threads.pop()
                            warp_threads.append(thread)
                    if self.verbose:
                        print("MPU ", self.id, "Starting new warp!")
                    warp.setThreads(warp_threads)
                warp.execute()
            self.checkWork()
        else:
            print("MPU ", self.id, " nothing to DO!")

    def checkWork(self):
        self.done = np.all(np.array([thread.done for thread in self.threads], dtype=bool))

class GPUDevice():
    def __init__(self, num_mpu, num_warps_per_mpu, num_threads_per_warp, name="", verbose=False):
        self.num_mpu = num_mpu
        self.num_warps_per_mpu = num_warps_per_mpu
        self.num_threads_per_warp = num_threads_per_warp
        self.concurrent_threads = num_mpu * num_warps_per_mpu * num_threads_per_warp
        self.mpus = [MPU(i, num_warps_per_mpu, num_threads_per_warp, verbose) for i in range(num_mpu)]
        self.blocks = []
        self.grid_dim = None
        self.block_dim = None
        self.done = True
        self.verbose = verbose
        self.name = name
        self.iterations = 0

    def setTask(self, thread_class, grid_dim, block_dim, matrix, s):
        self.matrix = matrix
        self.s = s
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.blocks = [block_id for block_id in range(grid_dim)]
        self.thread_class = thread_class
        self.done = False
        self.iterations = 0

    def execute(self):
        if not self.done:
            self.iterations += 1
            if self.verbose:
                print("-"*20, "GPU ", self.name, " Iteration: ", self.iterations, "-"*20)
            for mpu in self.mpus:
                if mpu.done:
                    if self.blocks:
                        mpu.create_threads(self.thread_class, self.blocks.pop(),
                                           self.block_dim, self.grid_dim,
                                           self.matrix, self.s)
                        mpu.execute()
                else:
                    mpu.execute()
            self.checkWork()
        else:
            print(self.name, " DONE!")

    def checkWork(self):
        self.done = np.all(np.array([mpu.done for mpu in self.mpus], dtype=bool))


def animate(update_creator, frames, data):
    cmap = LinearSegmentedColormap.from_list('mycmap', ['green', 'white', 'darkgreen', 'black'])
    fig, ax = plt.subplots()
    matrice = ax.matshow(data.reshape(s,s), cmap=cmap, vmin=0, vmax=NUM_CONCURRENT_THREADS)
    update = update_creator(matrice, data)
    plt.colorbar(matrice)

    frames = (s * s) // NUM_MPU + 1
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, repeat=False)
    plt.show()

# DEVICE
NUM_MPU = 4
NUM_WARP_PER_MPU = 16
NUM_THREADS_PER_WARP= 8
NUM_CONCURRENT_THREADS = NUM_MPU * NUM_WARP_PER_MPU * NUM_THREADS_PER_WARP

gpu = GPUDevice(NUM_MPU, NUM_WARP_PER_MPU, NUM_THREADS_PER_WARP, "GTX", True)

# Chose dimensions
# BLOCK_DIM = NUM_WARP_PER_MPU * NUM_THREADS_PER_WARP
# IF BLOCKDIM IS NOT EQUALS TO WARPS * THEADS -> INFINATELY MANY CACHE MISSES!!!
BLOCK_DIM = 256
GRID_DIM = NUM_MPU

# Prepare Data
s = 100
matrix = np.zeros(s*s) * 100

gpu.setTask(CudaThread, GRID_DIM, BLOCK_DIM, matrix, s)
gpu.execute()

# while not gpu.done:
#     print(matrix.reshape((s,s)))
#     gpu.execute()

def updata_creator(matrice, data):
    def update(i):
        gpu.execute()
        matrice.set_array(data.reshape((s,s)))
    return update

animate(updata_creator, 20, matrix)

# matrix = matrix.reshape((s,s))
# print(matrix)
# cmap = LinearSegmentedColormap.from_list('mycmap', ['green', 'white', 'darkgreen', 'black'])


# NUM_MPU = 2
# WARP_PER_MPU = 8
# WARPSIZE = 4
# NUM_CONCURRENT_THREADS = NUM_MPU * WARP_PER_MPU * WARPSIZE

# BLOCK_DIM = WARPSIZE
# GRID_DIM = NUM_MPU
# s = 100

# data = np.ones(s*s) * NUM_CONCURRENT_THREADS

# generator = image_generator2(BLOCK_DIM, GRID_DIM, data, s, WARPSIZE)

# def update(i):
#     generator.generate()
#     matrice.set_array(data.reshape((s,s)))

# fig, ax = plt.subplots()
# matrice = ax.matshow(data.reshape(s,s), cmap=cmap, vmin=0, vmax=NUM_CONCURRENT_THREADS)
# plt.colorbar(matrice)

# frames = (s * s) // NUM_MPU + 1
# ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, repeat=False)
# plt.show()