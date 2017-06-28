import numpy as np
from numpy.testing import assert_array_equal as array_eq
from numpy.testing import assert_array_almost_equal as array_aeq
inv = np.linalg.inv
det = np.linalg.det
norm = np.linalg.norm
from timeit import default_timer as timer

def gradient(a, b, Z, L, J, Y, dz, s):
    I = np.float64(np.identity(s)).reshape((s,s))
    Sigma = np.diag(np.sign(dz))
    Tss = I - L.dot(Sigma)
    ITss  = inv(Tss)
    gamma = b + Y.dot(Sigma).dot(ITss).dot(a)
    Gamma = J + Y.dot(Sigma).dot(ITss).dot(Z)
    return gamma, Gamma
        
# m = n = s = 10000
# a = np.float64(np.random.rand(s))
# b = np.float64(np.random.rand(m))
# Z = np.float64(np.random.rand(s*m)).reshape((s,m))
# L = np.tril(np.float64(np.random.rand(s*s)).reshape((s,s)), -1)
# J = np.float64(np.random.rand(m*n)).reshape((m,n))
# Y = np.float64(np.random.rand(m*s)).reshape((m,s))
# dz = np.float64(np.random.rand(n))

stats = []

print("Repetitions: 0")
for i in range(1000,8000,1000):
    m = n = s = i
    a = np.float64(np.random.rand(s))
    I = np.float64(np.identity(s)).reshape((s,s))
    b = np.float64(np.random.rand(s))
    Z = np.float64(np.random.rand(s*s)).reshape((s,s))
    L = np.tril(np.float64(np.random.rand(s*s)).reshape((s,s)), -1)
    J = np.float64(np.random.rand(s*s)).reshape((s,s))
    Y = np.float64(np.random.rand(s*s)).reshape((s,s))
    dz = np.float64(np.random.rand(s))
    # for k in range(1000):
    gamma = None
    Gamma = None
    start = timer()
    for k in range(100):
        gamma, Gamma = gradient(a,b,Z,L,J,Y,dz,s)
    end = timer()
    diff = end - start
    stats.append((i,diff))
    print(i, diff)