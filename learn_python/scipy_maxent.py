"""
Example use of scipy.maxentropy 

We are looking for a conditional model: p(i|j) 

scipy.maxentropy terminology (inherited from Adam Berger):
  i is the "class" == X
  j is the "context" == W


"""

import numpy as np 
import scipy.maxentropy as maxent


"""
F : number of features x (dimJ x dimI)
counts: counts(j+|W|i)=jointhisto(i,j) 
numcontexts = |W|
"""

"""
H: joint histogram, first dim is 'i', second dim is 'j'
"""
def joint_distribution_model(H):
    numcontexts = H.shape[1]
    counts = H.ravel()
    size = counts.size
    f0 = f_ssd(H.shape[0], H.shape[1])
    F = np.asarray([f0])
    return maxent.conditionalmodel(F, counts, numcontexts)
    
    
"""
F[0,i,j] = (i-j)^2 
"""
def f_ssd(dimI, dimJ):
    F = np.zeros([dimI, dimJ])
    for i in range(dimI):
        for j in range(dimJ): 
            F[i,j] = (i-j)**2
    return F.ravel()


##H = np.array([[10,34],[45,21],[23,12]])

H = np.ones([3,2])
m = joint_distribution_model(H)

m.verbose = True
m.fit()

p = m.probdist().reshape(H.shape)
lda = m.params

