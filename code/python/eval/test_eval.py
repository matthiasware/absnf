import numpy as np

def eval(a, b, Z, L, J, Y, dx, s):
	dz = a + Z.dot(dx.T)
	abs_dz = np.zeros(s)
	for i in range(s):
		dz[i] = dz[i] + L[i].dot(abs_dz)
		abs_dz[i] = abs(dz[i])
	dy = b + J.dot(dx.T)
	dy = dy + Y.dot(abs_dz.T)
	return dy, dz, abs_dz

n = 5
s = 4
m = 3

# s
a = np.array([0, 4, -3, 10])
# m
b = np.array([-8, 11, 7])
# s x n
Z = np.array([[-4, 0, -4, 1 ,-1],
			  [3, 0, -2, -3, -21],
			  [-3, -4, -4, -1, 33],
			  [-9, 0, -5, 3, 4]])

# s x s
L = np.array([[0, 0, 0, 0],
			  [4, 0, 0, 0],
			  [8, 9, 0, 0],
			  [2, 1, 7, 0]])

# m x n
J = np.array([[0, 0, 2, 1, 3],
			  [4, 2, 0, 1, 2],
			  [1, 3, -2, 1, 8]])

# m x s
Y = np.array([[0, 0, 2, 1],
			  [4, 2, 0, 4],
			  [1, 4, 7, 4]])

# n
dx = np.array([-5,8,0,1,2])

dy, dz, abs_dz = eval(a,b,Z,L,J,Y,dx,s)
print("dy\n", dy)
print("abs_dz\n", abs_dz)
print("dz\n", dz)

############################################################


print("a + Z*dx\n", dz)
for i in range(s):
	dz[i] = dz[i] + L[i].dot(abs_dz)
	abs_dz[i] = abs(dz[i])

print("calculated dz:\n", dz)
print("calculated abs_dz:\n", abs_dz)
print("re-calculated dz:\n", a + Z.dot(dx.T) + L.dot(abs_dz))

############################################################
# Tss
############################################################
Tss = np.diag(np.ones(s)) - L.dot(np.diag(np.sign(dz)))