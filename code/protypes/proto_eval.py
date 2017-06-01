import numpy as np

n = 4
s = 3

# n
a = np.array([1, 2, 3])
# s x n
Z = np.array([[-2, 1, 0, 1],
			  [3, -2, 1, 2],
			  [1, 3, -4, -2]])
# s x s
L = np.array([[0, 0, 0],
			  [2, 0, 0],
			  [1, -3, 0]])
# n
dx = np.array([1,2,3,4])
# s
dz = np.array([0, 0, 0])
# s
abs_dz = np.array([0, 0, 0])

dz = a + Z.dot(dx.T)
for i in range(s):
	dz[i] = dz[i] + L[i].dot(abs_dz)
	abs_dz[i] = abs(dz[i])

print "calculated dz:\n", dz
print "calculated dz:\n", abs_dz
print "re-calculated dz:\n", a + Z.dot(dx.T) + L.dot(abs_dz)


t_a = np.array([1, 2, 3])
t_M = Z = np.array([[-2, 1, 0, 1, 0, 0, 0],
			  		[3, -2, 1, 2, 2, 0, 0],
			  		[1, 3, -4, -2, 1, -3, 0]])
t_v = np.array([1,2,3,4,5,22,71])

print "calculated as whole:\n", t_a + t_M.dot(t_v.T)