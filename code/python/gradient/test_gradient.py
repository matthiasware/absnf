import numpy as np
inv = np.linalg.inv
n = 4
s = 3
m = 2

a = np.array([4, 4, -3]) # n
b = np.array([4,4]) # m

# s x n
Z = np.array([[-4, 0, -4, 1],
			  [3, 0, -2, -3],
			  [-3, -4, -4, 0]])

# s x s
L = np.array([[0, 0, 0],
			  [4, 0, 0],
			  [0, 4, 0]])

# m x n
J = np.array([[0, 0, 2, 0],
			  [4, 2, 0, 1]])

# m x s
Y = np.array([[0, 0, 2],
			  [4, 2, 0]])

# n
dz = np.array([0, -13, 26]);


#####################
Tss = np.diag(np.ones(s)) - L.dot(np.diag(np.sign(dz)));
I = inv(Tss)
I = I.dot(np.diag(np.sign(dz)));
K = Y.dot(I)

#################################

# Y = np.array([0,4,0,2,2,0])
# I = np.array([0,0,0,0,-1,0,0,4,1])
