import numpy as np 
from nibabel import load 
from nipy.neurospin.image import Image
from nipy.neurospin.registration import IconicRegistration, Affine
from nipy.utils import example_data
from os.path import join 

from iis import ImportanceSampler, mat_info


# Load images 
def make_matcher(): 
    I = Image(load(example_data.get_filename('neurospin',
                                             'sulcal2000',
                                             'nobias_ammon.nii.gz')))
    J = Image(load(example_data.get_filename('neurospin',
                                             'sulcal2000',
                                             'nobias_anubis.nii.gz')))


    # Create a registration instance
    R = IconicRegistration(I, J)
    R.set_source_fov(fixed_npoints=64**3)
    R.similarity = 'llr_cc'
    return R

# Create matcher instance 
print('Creating matcher instance')
R = make_matcher()

# Params
idx = slice(3,6)
##idx = 0
s = 1.
ndraws = 100

# Very empirical...
k = 25. 
###k = 50.
lda = k/float(R._source_npoints)

### d = p/(2*lda) = 60948.48
### Smax = 89851.59

# Initial transform 
print('Initializing transformation')
vgold = np.load('gold_ammon2anubis.npy')
T0 = Affine(vgold)
T = Affine(vgold)
x0 = T0.param[idx]

# Define cost function for ISOLA
def f(x): 
    #T = Affine(subtype=T0.subtype, flag2d=T0.flag2d)
    param = T.param
    param[idx] = x 
    T.param = param
    return np.exp(lda*R.eval(T))

# Perform Laplace approximation
print('Monte Carlo Laplace approximation')
S = (s**2)*np.diag(np.ones(x0.size))
#I = Isola(f, x0, S, ndraws=ndraws, discrete=True)
I = ImportanceSampler(f, x0, S, ndraws=ndraws, discrete=True) #symetrize=True

# Graphical display
from pylab import plot 

def display(I, h=s): 
    if not x0.size == 1: 
        raise ValueError('no display in dimension > 1')
    us = x0 + np.linspace(-3*h, 3*h, num=200)
    plot(I._xs[0], I._ps, 'o')
    qs = I.Fit.eval(np.reshape(us,(1,us.size)))
    plot(us, qs, 'r')

def check(I, i, h=s, num=30): 
    us = x0[i] + np.linspace(-3*h, 3*h, num=num)
    Us = np.tile(x0, (us.size, 1)).T
    Us[i,:] = us
    plot(us, I.sample(Us), 'b')
    plot(us, I.Fit.eval(Us), 'r')

#display(I)
#check(I, i)


