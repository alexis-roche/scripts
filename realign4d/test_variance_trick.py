import numpy as np 
from numpy.testing import assert_almost_equal

n = 100
x = np.random.rand(n)

mx = np.mean(x)
vx = np.mean((x-mx)**2)

y = x[1::]
my = np.mean(y)
vy = np.mean((y-my)**2)

a = (n-1)/float(n)
b = (n-1)/float(n**2)
vvx = a*vy + b*(x[0]-my)**2

print vx
print vvx

assert_almost_equal(vx, vvx)
