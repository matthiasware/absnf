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

s = 4
# BLOCK
blockDim = 3
gridDim = 6
L = np.array([[0, 0, 0, 0],
			  [0, 0, 0, 0],
			  [0, 0, 0, 0],
			  [0, 0, 0, 0]])
dz = np.array([-1, 0, 1, -1])
eTss = np.diag(np.ones(s)) - L.dot(np.diag(np.sign(dz)))
Tss = np.zeros(s*s).reshape((s,s))

# for blockId in range(gridDim):
# 	for threadId in range(blockDim):
# 		initTss(blockId,threadId, blockDim, gridDim, L, Tss, dz, s, s*s)
# Tss2 = Tss.reshape((s,s))

# FLAT
L = L.flatten()
Tss = np.zeros(s*s)
eTss = eTss.flatten()

for blockId in range(gridDim):
	for threadId in range(blockDim):
		initTss_flat(blockId, threadId, blockDim, gridDim, L, Tss, dz, s, s*s)