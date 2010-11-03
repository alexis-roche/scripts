import numpy as np
import scipy.ndimage as nd
from os.path import join 
from nipy.io.imageformats import Nifti1Image as Image, load as load_image, save as save_image
import time 
import pylab 

id = 5 
subjects = ['test', '016', '020', '023_1', '031', '03S0908'] 
datadir = join('D:\home\AR203069\data\delphine', subjects[id])
tissues = ['CSF','GM','WM']

# Read input image 
im = load_image(join(datadir, 'BiasCorIm.img'))

# Compute brain posterior probability map 
P = np.zeros(im.get_shape())
for tissue in tissues: 
    tmp = load_image(join(datadir, tissue+'PostMap_1000.img'))
    P += tmp.get_data()
P /= 1000. 
##P = np.gaussian_filter(P, sigma=3)
S = 2*(P-.5)
del P 

# Compute gradient norm image 
G = nd.gaussian_gradient_magnitude(im.get_data(), sigma=3)

# Speed function 
Gn = G/float(G.max())

abs_S = np.abs(S)
sgn_S = np.sign(S)

tmp = abs_S**10

cGn = (1-tmp)*Gn + tmp
cGn *= sgn_S



# Display
pylab.figure() 
pylab.imshow(cGn[:,80,:])

pylab.pink()
pylab.show()


# Speed function



"""
lap = nd.gaussian_laplace(im.get_data(), sigma=3)
imlap = Image(lap, im.get_affine())
save_image(imlap, join(datadir, 'LaplacianImage.nii'))
"""

