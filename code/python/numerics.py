import numpy as np
from numpy.testing import assert_array_equal as array_eq
from numpy.testing import assert_array_almost_equal as array_aeq
inv = np.linalg.inv
det = np.linalg.det
norm = np.linalg.norm
from timeit import default_timer as timer

# def validateEval(a, b, Z, L, J, Y, dx, dy, dz):
#         array_aeq(a + Z.dot(dx) + L.dot(abs(dz)), dz, decimal=8)
#         array_aeq(b + J.dot(dx) + Y.dot(abs(dz)), dy, decimal=8)

def eval(a, b, Z, L, J, Y, dx):
        s = len(a)
        dz = a + Z.dot(dx)
        abs_dz = np.zeros(s)
        for i in range(s):
            dz[i] = dz[i] + L[i].dot(abs_dz)
            abs_dz[i] = abs(dz[i])
        dy = b + J.dot(dx.T)
        dy = dy + Y.dot(abs_dz.T)
        # validateEval(a, b, Z, L, J, Y, dx, dy, dz)
        
        return dy, dz

m = n = s = 10000
a = np.float64(np.random.rand(s))
b = np.float64(np.random.rand(m))
Z = np.float64(np.random.rand(s*m)).reshape((s,m))
L = np.tril(np.float64(np.random.rand(s*s)).reshape((s,s)), -1)
J = np.float64(np.random.rand(m*n)).reshape((m,n))
Y = np.float64(np.random.rand(m*s)).reshape((m,s))
dx = np.float64(np.random.rand(n))

stats = []

for i in range(1000,11000,1000):
    m = n = s = i
    a = np.float64(np.random.rand(s))
    b = np.float64(np.random.rand(m))
    Z = np.float64(np.random.rand(s*m)).reshape((s,m))
    L = np.tril(np.float64(np.random.rand(s*s)).reshape((s,s)), -1)
    J = np.float64(np.random.rand(m*n)).reshape((m,n))
    Y = np.float64(np.random.rand(m*s)).reshape((m,s))
    dx = np.float64(np.random.rand(n))
    start = timer()
    for k in range(1000):
        dy = None
        dz = None
        dy, dz = eval(a,b,Z,L,J,Y,dx)
    end = timer()
    diff = end - start
    stats.append((i,diff))
    print(i, diff)