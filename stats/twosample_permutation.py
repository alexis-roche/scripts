import numpy as np 
from pylab import *

n1 = 5
n2 = 4
nsimu = 100000

x1 = np.random.rand(n1) + .3
x2 = np.random.rand(n2) - .3

x = np.concatenate((x1,x2))
t0 = np.mean(x[0:n1])-np.mean(x[n1:-1])

t = np.zeros(nsimu)
for i in range(nsimu): 
    y = np.random.permutation(x)
    t[i] = np.mean(y[0:n1])-np.mean(y[n1:-1])


tmax = np.max(np.abs(t))
ts = np.linspace(-tmax,tmax)
hs = np.zeros(ts.size)
for i in range(ts.size):
    tt = ts[i]
    hs[i] = np.sum((t>ts[i]-.5)*(t<=ts[i]+.5))

bar(ts, hs, width=ts[1]-ts[0]) 
idx = np.min(np.where(ts>=t0))

bar(ts[idx:], hs[idx:], width=ts[1]-ts[0], color='g') 


