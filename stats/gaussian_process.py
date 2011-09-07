import numpy as np 
from scipy.linalg import cho_factor, cho_solve

def kernel_matrix(xyz, centers, sigma): 
    """
    Compute kernel matrix: Kij = g(xi-cj) 
    """
    dim = centers.shape[1]
    D = np.zeros((xyz.shape[0], dim, centers.shape[0]))
    D -= (centers/sigma).T 
    Dt = np.transpose(D, (2,1,0))
    Dt += (xyz/sigma).T 
    return np.exp(-.5*np.sum(D**2, 1)) 


# Correct local affines for overlapping kernels
def correct_affines(affines, centers, sigma): 
    K = kernel_matrix(centers, centers, sigma)
    L = np.linalg.cholesky(K) # K = L L.T
    Linv = np.linalg.inv(L) 
    Kinv = np.dot(Linv.T, Linv) 
    return np.dot(affines[:, 0:3, :].T, Kinv.T).T


def covariance_matrix(x, sigma):
    D = np.zeros((len(x), len(x)))
    D[:] = x
    D.T[:] -= x.T
    K = np.exp(-.5*D**2/sigma**2)
    return K

def marginal_log_likelihood(K, regul=1e-10):
    L, lower = cho_factor(K+regul*np.eye(K.shape[0])) # K = L L.T
    d = np.maximum(np.diagonal(L), 1e-200) 
    complexity = np.sum(np.log(d))
    c = cho_solve((L, lower), y)
    data_fit = np.dot(c, c) 
    return -.5*(data_fit+complexity)




n = 100
x = 10*np.random.rand(n)
y = x**2

params = np.linspace(-10,0)
sigmas = np.exp(params)
L = []

for s in sigmas:
    K = covariance_matrix(x, s)
    L.append(marginal_log_likelihood(K))

L = np.array(L) 
