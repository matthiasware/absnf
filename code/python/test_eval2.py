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
dx = np.array([-3,4,4,0])

dy, dz, abs_dz = eval(a,b,Z,L,J,Y,dx,s)
# print("dy\n", dy)
# print("abs_dz\n", abs_dz)
# print("dz\n", dz)

############################################################


# print("a + Z*dx\n", dz)
for i in range(s):
	dz[i] = dz[i] + L[i].dot(abs_dz)
	abs_dz[i] = abs(dz[i])

# print("calculated dz:\n", dz)
# print("calculated abs_dz:\n", abs_dz)
# print("re-calculated dz:\n", a + Z.dot(dx.T) + L.dot(abs_dz))


############################################################
# Tss
############################################################
Tss = np.diag(np.ones(s)) - L.dot(np.diag(np.sign(dz)))