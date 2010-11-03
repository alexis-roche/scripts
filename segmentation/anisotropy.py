import numpy as np
import scipy.ndimage as nd
from os.path import join 
from nipy.io.imageformats import Nifti1Image as Image, load as load_image, save as save_image
import time 
import pylab 
from nipy.neurospin.utils.routines import svd

# Display
def display(im, slice=80): 
    pylab.figure() 
    pylab.imshow(im[:,slice,:])

# Anisotropy measures 
def fa(S, lda=0.0):
	"""
	Fractional anisotropy
	"""
	Sm = S.mean(0)
	r = ((S-Sm)**2).sum(0)/((S+lda)**2).sum(0)
	r[:] = np.sqrt(1.5*r[:])
	return r

def aniso(S, lda=0.0):
	"""
	trace/det
	"""
	r = (S.mean(0)+lda)/(S+lda).prod(0)**(1/3.)
	return r
	
def sv(S, id):
	return S[id,:]
	
def hessian(data, sigma=1): 
	H = np.zeros([3,3]+list(data.shape))	
	H[0,0,:] = nd.gaussian_filter(data, sigma=sigma, order=[2,0,0]) 
	H[1,1,:] = nd.gaussian_filter(data, sigma=sigma, order=[0,2,0]) 
	H[2,2,:] = nd.gaussian_filter(data, sigma=sigma, order=[0,0,2]) 
	Hxy = nd.gaussian_filter(data, sigma=sigma, order=[1,1,0])
	H[0,1,:] = Hxy[:]
	H[1,0,:] = Hxy[:]
	Hxz = nd.gaussian_filter(data, sigma=sigma, order=[1,0,1])
	H[0,2,:] = Hxz[:]
	H[2,0,:] = Hxz[:]
	Hyz = nd.gaussian_filter(data, sigma=sigma, order=[0,1,1])
	H[1,2,:] = Hyz[:]
	H[2,1,:] = Hyz[:]
	return H
	

#datadir = 'D:\home\Alexis\data\delphine\zozo'
datadir = 'D:\Alexis\data\patient_03S0908'

# Read input image 
im = load_image(join(datadir, 'BiasCorIm.img'))
data = im.get_data()
data = data.astype('float')
sigma = 3
lda = 1

# Compute hessian 
print('Computing Hessian...')
H = hessian(data, sigma=sigma)

# Singular value decomposition
print('Computing singular values...') 
S = svd(H)

# Anisotropy measure 
print('Computing FA...') 
#I = fa(S)
I = aniso(S, lda=lda)
    
display(I)	
pylab.pink()
pylab.show()
