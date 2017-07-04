import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
import random
from matplotlib.offsetbox import AnchoredText

# class cuda_thread():
#     def __init__(self, thread_id, block_id, matrix, 
#                  s, block_dim, grid_dim):
#         self.thread_id = thread_id
#         self.block_id = block_id
#         self.matrix = matrix
#         self.s = s
#         self.i = thread_id
#         self.j = block_id
#         self.block_dim = block_dim
#         self.grid_dim = grid_dim
#         self.id = self.i * self.s + self.j;
#         self.done = False
#         self.started = False
#         self.global_id = thread_id + block_id * block_dim

#     def update(self):
#         if self.id < self.s*self.s and self.j<self.s:
#             self.matrix[self.id] = self.global_id
#             self.i += self.block_dim
#             if self.i>=self.s:
#                 self.i = self.i%self.s
#                 self.j = self.j + self.grid_dim
#             self.id = self.i*self.s + self.j
#         else:
#             self.done = True

# class cuda_thread2():
#     def __init__(self, thread_id, block_id, matrix, 
#                  s, block_dim, grid_dim):
#         self.thread_id = thread_id
#         self.block_id = block_id
#         self.matrix = matrix
#         self.s = s
#         self.i = thread_id
#         self.j = block_id
#         self.block_dim = block_dim
#         self.grid_dim = grid_dim
#         self.id = self.i * self.s + self.j;
#         self.done = False
#         self.started = False
#         self.global_id = thread_id + block_id * block_dim

#     def update(self):
#         if self.id < self.s*self.s and self.j<self.s:
#             if self.i < self.s:
#                 self.matrix[self.id] = self.global_id
#                 self.i += self.block_dim
#             else:
#                 self.i = self.i%self.s
#                 self.j = self.j + self.grid_dim
#             self.id = self.i*self.s + self.j
#         else:
#             self.done = True


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

    def setTask(self, thread_class, grid_dim, block_dim, matrix, s):
        self.matrix = matrix
        self.s = s
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.blocks = [block_id for block_id in range(grid_dim)]
        self.thread_class = thread_class
        self.done = False

    def execute(self):
        if not self.done:
            if self.verbose:
                print("GPU: Next Iteration!")
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
NUM_WARP_PER_MPU = 8
NUM_THREADS_PER_WARP= 8
NUM_CONCURRENT_THREADS = NUM_MPU * NUM_WARP_PER_MPU * NUM_THREADS_PER_WARP

gpu = GPUDevice(NUM_MPU, NUM_WARP_PER_MPU, NUM_THREADS_PER_WARP, "GTX", True)

# Chose dimensions
BLOCK_DIM = NUM_WARP_PER_MPU * NUM_THREADS_PER_WARP
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