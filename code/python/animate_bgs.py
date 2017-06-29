import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

plot_values = True
s = 100
block_dim = 64
grid_dim = 10
num_threads = block_dim * grid_dim
num_multicores = 10


data = np.zeros(s*s)

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


class image_generator2():
	def __init__(self,block_dim, grid_dim,
				 matrix,s,num_multicores,
			     num_warps, num_threads_per_warps,
			     inplace=True, multicores=None):
		self.threads = []
		self.num_multicores = num_multicores
		self.num_warps = num_warps
		self.num_threads_per_warps = num_threads_per_warps
		self.current_thread = 0
		self.inplace = inplace
		self.matrix = matrix
		if multicores:
			self.multicores = multicores
		else:
			self.multicores = block_dim*grid_dim
		for block_id in range(grid_dim):
			for thread_id in range(block_dim):
				thread = cuda_thread(thread_id, block_id, matrix,
							         s, block_dim, grid_dim)
				self.threads.append(thread)
			self.all_done = np.zeros(len(self.threads), dtype=bool)
			self.next = True
		self.matrix_time = []
		self.current_threads = []

	# def getThreadsToExecute():
	# 	threads = []
	# 	for core in range(self.num_multicores):
	# 		for warps in range(self.num_warps):
	# 			for thread in range(self.num_threads_per_warps):

	def generate(self):
		if not self.next:
			return
		if not self.inplace:
			self.matrix_time.append(np.copy(self.matrix))
		for i in range(self.multicores):
			thread = self.threads[self.current_thread]
			thread.update()
			self.all_done[self.current_thread] = thread.done
			self.current_thread = (self.current_thread + 1) % len(self.threads)
		# for i, thread in enumerate(self.threads):
		# 	thread.update()
		# 	self.all_done[i] = thread.done
		if np.all(self.all_done):
			self.next = False

class image_generator():
	def __init__(self,block_dim, grid_dim, matrix,s, inplace=True, multicores=None):
		self.threads = []
		self.current_thread = 0
		self.inplace = inplace
		self.matrix = matrix
		if multicores:
			self.multicores = multicores
		else:
			self.multicores = block_dim*grid_dim
		for block_id in range(grid_dim):
			for thread_id in range(block_dim):
				thread = cuda_thread(thread_id, block_id, matrix,
							         s, block_dim, grid_dim)
				self.threads.append(thread)
			self.all_done = np.zeros(len(self.threads), dtype=bool)
			self.next = True
		self.matrix_time = []

	def generate(self):
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
		# for i, thread in enumerate(self.threads):
		# 	thread.update()
		# 	self.all_done[i] = thread.done
		if np.all(self.all_done):
			self.next = False

# def create_cmap(n,cmap):
# 	cmap = plt.cm.get_cmap(cmap)
# 	colors = cmap(np.arange(cmap.N))
# 	return cmap.from_list(cmap.name , colors, cmap.N)

generator = image_generator(block_dim, grid_dim, data, s, inplace=False, multicores=num_multicores)

def update(i):
	generator.generate()
	matrice.set_array(data.reshape((s,s)))

fig, ax = plt.subplots()
matrice = ax.matshow(data.reshape(s,s), vmin=0, vmax=num_threads)
plt.colorbar(matrice)

ani = animation.FuncAnimation(fig, update, frames=4, interval=100)
plt.show()

data = data.reshape((s,s))
print(s)



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