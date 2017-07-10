import numpy as np
from numpy.testing import assert_array_equal as array_eq
inv = np.linalg.inv
det = np.linalg.det
norm = np.linalg.norm

def mem_grad(m,n,s,size):
    s_bytes = (s+m+s*n+s*s+m*n+m*s+s+m+m*n+s*s+s*s+m*s)*size
    s_gb = s_bytes * 1e-9
    return s_bytes, s_gb

class Absnf():
    def eval(self, a, b, Z, L, J, Y, dx):
        s = len(a)
        dz = a + Z.dot(dx)
        abs_dz = np.zeros(s)
        for i in range(s):
            dz[i] = dz[i] + L[i].dot(abs_dz)
            abs_dz[i] = abs(dz[i])
        dy = b + J.dot(dx.T)
        dy = dy + Y.dot(abs_dz.T)
        
        self.validateEval(a, b, Z, L, J, Y, dx, dy, dz)
        
        return dy, dz

    def validateEval(self, a, b, Z, L, J, Y, dx, dy, dz):
        array_eq(a + Z.dot(dx) + L.dot(abs(dz)), dz)
        array_eq(b + J.dot(dx) + Y.dot(abs(dz)), dy)

    def gradient(self, a, b, Z, L, J, Y, dz):
        Sigma = np.diag(np.sign(dz))
        Tss = I - L.dot(Sigma)
        ITss  = inv(Tss)
        gamma = b + Y.dot(Sigma).dot(ITss).dot(a)
        Gamma = J + Y.dot(Sigma).dot(ITss).dot(Z)
        return gamma, Gamma

    def _calcDX(self, IJ, b, dz):
        K = Y.dot(np.abs(dz))
        return -IJ.dot(b + K)

    def _solver_gnewton(self, S, c, dz_now, tol,maxiter, verbose):
        i = 0
        diff = 10000
        s = len(dz_now)
        I = np.identity(s)
        while i < maxiter and diff > tol:
            dz_old = dz_now
            dz_now =  inv(I - S.dot(np.diag(np.sign(dz_now)))).dot(c)
            diff = norm(dz_now - dz_old)
            if verbose:
                print(i, diff)
            i += 1
        return dz_now

    def _solver_modulus(self, S, c, dz_now, tol,maxiter, verbose):
        i = 0
        diff = 10000
        while i < maxiter and diff > tol:
            dz_old = dz_now
            dz_now =  c + S.dot(np.abs(dz_now))
            diff = norm(dz_now - dz_old)
            if verbose:
                print(i, diff, dz_now)
            i += 1
        return dz_now        

    def _solver_blockSeidel():
        pass

    def solve(self, a, b, Z, L, J, Y, dy, verbose=False, solver="modulus",
              tol = 1e-8, maxiter=1000, dz_start=None):
        if np.any(dy):
            b = b - dy
            print("b", b)
        IJ = inv(J)
        K = Z.dot(IJ)
        S = L - K.dot(Y)
        c = a - K.dot(b)
        if not dz_start:
            dz_now = np.random.rand(len(a)) * 10
        else:
            dz_now = dz_start
        # dz_now = a
        if solver == "newton":
            dz = self._solver_gnewton(S, c, dz_now, tol, maxiter, verbose)
            dx = self._calcDX(IJ, b, dz)
        elif solver == "modulus":
            dz = self._solver_modulus(S, c, dz_now, tol, maxiter, verbose)
            dx = self._calcDX(IJ, b, dz)
        else:
            raise Exception("Not Implemented")
        return dz, dx

############################################
# int seed = 1;
m = n = s = 5
a = np.array([9,8,-7,9,-5,])  # s
b = np.array([9,8,-7,9,-5,])  # m // => y = 0

# s x n
Z = np.array([[9,8,-7,9,-5,],
              [-2,-2,3,8,6,],
              [-3,0,-1,-5,0,],
              [8,0,-1,8,5,],
              [4,6,-8,0,7,]])

# s x s
# L = np.array([[0,0,0,0,0,],
#              [9,0,0,0,0,],
#              [8,-7,0,0,0,],
#              [9,-5,-2,0,0,],
#              [-2,3,8,6,0,]])

L = np.zeros((s,s))

# m x n
# The bigger J the smaller the inverse,
# the faster the modulo algorithm konverges
# Also the bigger L the slower it converges
J = np.array([[1000,8,-7,9,-5,],
              [-2,1000,3,8,6,],
              [-3,0,1000,-5,0,],
              [8,0,-1,1000,5,],
              [4,6,-8,0,1000,]])

# m x s
Y = np.array([[9,8,-7,9,-5,],
              [-2,-2,3,8,6,],
              [-3,0,-1,-5,0,],
              [8,0,-1,8,5,],
              [4,6,-8,0,7,]])
# n
dx = np.array([9,8,-7,9,-5])
dz_start = np.array([9,8,-7,9,-5,])

absnf = Absnf()
dy, dz = absnf.eval(a,b,Z,L,J,Y,dx)

# dz_mod, dx_mod = absnf.solve(a,b,Z,L,J,Y,dy, True)

if np.any(dy):
    b = b - dy
IJ = inv(J)
K = Z.dot(IJ)
S = L - K.dot(Y)
c = a - K.dot(b)

# dz_now = np.array([9,8,-7,9,-5,])
# dz_old = dz_now
# dz_now =  c + S.dot(np.abs(dz_now))
# diff = norm(dz_now - dz_old)

############################################