import numpy as np
from numpy.testing import assert_array_equal as array_eq
inv = np.linalg.inv
det = np.linalg.det
norm = np.linalg.norm
qr = np.linalg.qr


def mem_eval(m,n,s, size):
  return (s+m+(s*n)+(s*s)+(m*n)+(m*s)+m+n+s+s)*size

def com_eval(m,n,s):
  return s + s*n + s*s + m + m*n + m*s

def com_parallel(m,n,s,p):
  return s/p + (s*n)/p + (s*s)/p + (m + m*n)/p + (m*s)/p

def eval(a, b, Z, L, J, Y, dx):
    s = len(a)
    dz = a + Z.dot(dx)
    abs_dz = np.zeros(s)
    for i in range(s):
        dz[i] = dz[i] + L[i].dot(abs_dz)
        abs_dz[i] = abs(dz[i])
    dy = b + J.dot(dx.T)
    dy = dy + Y.dot(abs_dz.T)
    return dy, dz


n = 4
m = 4
s = 3

a = np.array([4, 4, -3])  # s
b = np.array([-223, -432, -200, -48])  # m // => y = 0

# s x n
Z = np.array([[-4, 0, -4, 1],
              [3, 0, -2, -3],
              [-3, -4, -4, 0]])

# s x s
L = np.array([[0, 0, 0],
              [1, 0, 0],
              [0, 1, 0]])

# m x n
# The bigger J the smaller the inverse,
# the faster the modulo algorithm konverges
# Also the bigger L the slower it converges
J = np.array([[100, 1, 2, 1],
              [4, 120, 0, 1],
              [3, 5, 120, 6],
              [1, 1, 0, 130]])

# m x s
Y = np.array([[0, 0, 2],
              [4, 2, 0],
              [2, 1, 3],
              [0, 1, 3]])
# n
dx = np.array([2, 3, 1, 0])

IJ = inv(J)
K = Z.dot(IJ)
S = L - K.dot(Y)
c = a - K.dot(b)

dz_now = np.array([-1.59449432,  9.28890523,  9.39411967])
tol = 1e-8
verbose = True
i = 0
diff = 10000
maxiter = 100
if verbose:
    print("dz_start:", dz_now)
while i < maxiter and diff > tol:
    dz_old = dz_now
    dz_now =  c + S.dot(np.abs(dz_now))
    diff = norm(dz_now - dz_old)
    if verbose:
        print(i, diff, dz_now)
        i += 1

dy_real, dz_real = eval(a, b, Z, L, J, Y, dx)
print("dz_real: ", dz_real)
print("dz_now: ", dz_now)