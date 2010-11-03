import numpy as np
from scipy.optimize import * 

def f(x):
    return np.sum(x**2)

def minimize_f(x0, **kwargs):
    kwargs.setdefault('xtol', 1e-3)
    print kwargs
    return fmin(f, x0, **kwargs)


x = minimize_f(np.ones(10), xtol=.1)
