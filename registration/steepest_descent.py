import numpy as np 
from scipy.optimize import brent, approx_fprime

_epsilon = np.sqrt(np.finfo(float).eps)

def _linesearch_brent(func, p, xi, tol=1e-3):
    """Line-search algorithm using Brent's method.

    Find the minimium of the function ``func(x0+ alpha*direc)``.
    """
    def myfunc(alpha):
        return func(p + alpha * xi)
    alpha_min, fret, iter, num = brent(myfunc, full_output=1, tol=tol)
    xi = alpha_min*xi
    return np.squeeze(fret), p+xi


def _wrap(function, args):
    ncalls = [0]
    def wrapper(x):
        ncalls[0] += 1
        return function(x, *args)
    return ncalls, wrapper


def fmin_steepest(f, x0, fprime=None, xtol=1e-4, ftol=1e-4, 
                  epsilon=_epsilon, 
                  maxiter=None, callback=None): 

    x = np.asarray(x0).flatten()
    fval = np.squeeze(f(x))
    it = 0 
    if maxiter == None: 
        maxiter = x.size*1000
    if fprime == None:
        grad_calls, myfprime = _wrap(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = _wrap(fprime, args)

    while it < maxiter:
        x0 = x 
        fval0 = fval
        direc = myfprime(x)
        direc = direc / np.sqrt(np.sum(direc**2))
        fval, x = _linesearch_brent(f, x, direc, tol=xtol*100)
        if not callback == None:
            callback(x)
        if (2.0*(fval0-fval) <= ftol*(abs(fval0)+abs(fval))+1e-20): 
            break
        it = it + 1
        
    print it
    print fval

    return x 


def myfun(x):
    s = np.arange(x.size)+1.
    return np.sum((s*x)**2)


x0 = 100*np.random.rand(10)
x = fmin_steepest(myfun, x0)        
