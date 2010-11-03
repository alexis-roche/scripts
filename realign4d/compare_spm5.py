import numpy as np
from os.path import join

from nipy.io.imageformats import load as load_image, save as save_image
from nipy.neurospin.registration import FmriRealign4d
from pylab import * 


# Create Nifti1Image instances from both input files
rootpath = 'D:\\home\\Alexis\\data\\karla'

run = 'fms070149316-0032.nii'
raw = load_image(join(rootpath, run))
spm = load_image(join(rootpath, 'spm5_ra'+run))
npy = load_image(join(rootpath, 'new_ra'+run))

raw = raw.get_data()
spm = spm.get_data()
npy = npy.get_data()

def compare(x, y, z):
    plot(raw[x,y,z,:], 'g')
    plot(spm[x,y,z,:], 'r')
    plot(npy[x,y,z,:], 'b')


compare(39, 22, 18)


