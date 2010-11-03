import numpy as np 
from scipy.optimize import fmin_cg, fmin_bfgs

def f(x, p=2):
    ##s = np.arange(x.size)+1.
    ##r = np.abs(s*x)
    r = np.abs(x)
    return np.sum(r**p)
    #return np.dot(x, x)

x0 = np.random.rand(10)
epsilon = .00001

x = fmin_cg(f, x0)

##xx = fmin_bfgs(f, x0, epsilon=epsilon)
