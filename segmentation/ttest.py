import numpy as np
import pylab as p

from nipy.neurospin import glm

# Load data 
data = np.loadtxt('ad_data2.txt')
age = data[:,0]

# Linear regression
X = np.asarray([age, np.ones(len(age))]).T
Y = data[:,1:] # csf, gm, wm 
mod = glm.glm(Y, X)

# t-test: return z-scores 
# Note that absolute values indicate significance and 
# signs whether effects are positive or negative 
c = mod.contrast([1,0])
z = c.zscore()
pval = c.pvalue() 

# Display
fit = np.dot(X, mod.beta)
p.plot(age, Y[:,0], 'b+')
p.plot(age, fit[:,0], 'b')
p.plot(age, Y[:,1], 'r+')
p.plot(age, fit[:,1], 'r')
p.plot(age, Y[:,2], 'g+')
p.plot(age, fit[:,2], 'g')
