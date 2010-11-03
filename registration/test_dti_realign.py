import numpy as np 
from nipy import load_image
from nipy.core.image.affine_image import AffineImage 
from nipy.neurospin.registration import * 

im4d = load_image('18620_0006.nii')

affine = np.eye(4)
affine[0:3,0:4] = im4d.affine[0:3,0:4]

I = AffineImage(im4d.get_data()[:,:,:,0], affine, 'ijk')
J = AffineImage(im4d.get_data()[:,:,:,5], affine, 'ijk')
R = IconicRegistration(I, J) 

# Use the correlation coefficient and trilinear interpolation -
# which I expect to be safer for that kind of problems
R.interp = 'tri'
R.similarity = 'cc'

# Register and resample
T = Affine()
T = R.optimize(T)
Jt = resample(J, T) 


# 4D approach 
"""
R = Realign4d(im4d, Affine)
R.estimate()
r_im4d = R.resample()
"""
