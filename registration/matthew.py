import numpy as np 
from nibabel import load, save
from nipy.neurospin.image import Image, asNifti1Image, transform_image
from nipy.neurospin.registration import IconicRegistration, Affine, GridTransform
from os.path import join 

datadir = join('d:\\', 'home', 'Alexis', 'My Dropbox', 'example_images')
I = load(join(datadir, 'b0.nii'))
J = load(join(datadir, 'b1000.nii'))

# Create a registration instance
print('Creating matcher instance')
R = IconicRegistration(Image(I), Image(J))
R.set_source_fov(fixed_npoints=64**3)
R.similarity = 'cr'
R.interp = 'tri'

T = R.optimize(Affine())
