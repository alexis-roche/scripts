import numpy as np
import scipy.stats as ss
import pylab as pl

n = 100
k = 1./n

y = np.random.normal(size=n)
m = y.mean()
v = y.var()

s = np.sqrt(v/n)
x = np.linspace(-5*s+m, 5*s+m, num=1000)

p1 = (1+(x-m)**2/v)**(-.5*n)

vv = k*v
nn = 1 + k*(n-1)
p2 = (1+(x-m)**2/vv)**(-.5*nn)

p3 = (1+(x-m)**2/v)**(-.5*k*n)

pl.plot(x,p1)
pl.plot(x,p2,'r')
pl.plot(x,p3,'g')
