import numpy as np

def initTss(blockId, threadId, blockDim, gridDim, L, Tss, dz, s, size):
	i = threadId # ROW IDX
	j = blockId # COL IDX
	id = j*s + i # ḾEM IDX
	while id < size:
		if i < s:
			if i==j:
				Tss[i,j] = 1
			elif j>i:
				Tss[i,j] = 0
			else:	
				Tss[i,j] = 0 - L[i,j]*dz[j]
			i += blockDim
		else:
			i = i%s
			j = j + gridDim
		id = j*s + i

def initTss_flat(blockId, threadId, blockDim, gridDim, L, Tss, dz, s, size):
	i = threadId # ROW IDX
	j = blockId # COL IDX
	id = i*s + j # ḾEM IDX
	while id < size and j<s:
		Tss[id] = id
		print("(", i, ",", j,") - ", id)
		i += blockDim
		if i>=s:
			print("i>=s")
			i = i%s
			j = j + gridDim
		id = i*s + j

				
	print("abort with id: ", id, "i ",i, "j ", j)


def test_Tss(blockId, threadId, blockDim, gridDim, data, s):
	i = blockId
	j = threadId
	global_id = blockId * blockDim + threadId
	id = i*s + j
	while id < s*s and i < s:
		print(id)
		data[id] = global_id
		j += blockDim
		if j >= s:
			j = j % s
			i = i + gridDim
		id = i*s + j

s = 4
matrix = np.zeros((s,s)).flatten()
print(matrix)
BLOCK_DIM = 3
GRID_DIM = 2
for block_id in range(GRID_DIM):
	for thread_id in range(BLOCK_DIM):
		test_Tss(block_id, thread_id, BLOCK_DIM, GRID_DIM, matrix, s)
		# print(matrix.reshape((s,s)))
		pass

# test_Tss(1,1, BLOCK_DIM, GRID_DIM, matrix, s)
# print(matrix.reshape((s,s)))