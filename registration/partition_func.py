import numpy as np
from pylab import * 

def Z(lda): 
    y = w * (x**lda)
    return y.sum()

K = 100
x = np.random.rand(K)
x /= x.max()
w = np.random.rand(K)
w /= w.sum()

lda = np.arange(100)
z = np.zeros(lda.size)

for i in range(lda.size): 
    z[i] = Z(lda[i])

plot(lda, z)

#plot(lda, log(z))
##plot(lda, -log(lda))
