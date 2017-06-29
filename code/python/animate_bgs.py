import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

plot_values = True
s = 4
block_dim = 2
grid_dim = 2

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
	
	def update(self):
		if self.id < self.s*self.s and self.j<self.s:
			self.matrix[self.id] = self.id
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
		if self.
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

generator = image_generator(block_dim, grid_dim, data, s, multicores=2)
while generator.next:
	generator.generate()

data = data.reshape((s,s))
print(data)

# calculate
# done = np.zeros(len(threads), dtype=bool)
# while(True):
# 	print("round")
# 	for i, thread in enumerate(threads):
# 		thread.update()
# 		done[i] = thread.done
# 	if np.all(done):
# 		break
# data = data.reshape((s,s))
# print(data)


# M=np.array([[0,0,100,100,100,100,100,100,300,300,300,300,300,300,500,500,500,500,500,500,1000,1000,1000,1000] for i in range(0,20)]) 
# M=np.zeros(cols*rows)

# def update(i):
#     M[7,i] = 1000
#     M[19-i,10] = 500
#     matrice.set_array(M)

# fig, ax = plt.subplots()
# matrice = ax.matshow(M)
# plt.colorbar(matrice)

# ani = animation.FuncAnimation(fig, update, frames=19, interval=500)
# plt.show()








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