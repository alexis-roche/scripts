import numpy as np
from os.path import join

from nipy.io.imageformats import Nifti1Image, load as load_image, save as save_image
from nipy.utils import example_data
from nipy.neurospin.registration import FmriRealign4d
from pylab import * 

# Create Nifti1Image instances from both input files
rootpath = 'D:\\home\\Alexis\\data\\eccv'

mask = load_image(join(rootpath, 'mask.nii'))
xyz = np.where(mask.get_data()>0)

run = 'run1'

im_raw = load_image(example_data.get_filename('fiac','fiac0',run+'.nii.gz'))
im_npy = load_image(join(rootpath, 'npy_ra'+run+'.nii.gz'))
im_spm = load_image(join(rootpath, 'ra'+run+'.nii'))

raw = im_raw.get_data()
npy = im_npy.get_data()
spm = im_spm.get_data()

TR = 2.5

def compare(x, y, z):
    t = TR*np.arange(raw.shape[3])
    plot(t, raw[x,y,z,:], 'r')
    plot(t, spm[x,y,z,:], 'g')
    plot(t, npy[x,y,z,:], 'b')
    xlabel('time (sec)')

#compare(39, 22, 18)
#compare(15,24,10)

def plusmieux(x, a): 
    return ((x>0)*x)**a


"""
from nipy.neurospin.glm import glm 

f = np.load(join(rootpath, run+'_design.npz'))
X = f['X']
conds = f['conditions']

# Gaussian filter 
import scipy.ndimage as nd

g_raw = glm(raw[xyz], X, axis=1)
g_npy = glm(npy[xyz], X, axis=1)
g_spm = glm(spm[xyz], X, axis=1)

def makeZim(g, c): 
    z = g.contrast(c).zscore()
    r = np.zeros(raw.shape[0:-1])
    r[xyz] = z
    return r 

# SSt-SSp contrast 
c = np.zeros(X.shape[1])
c[0:4] = [1,1,-1,-1]

zraw = makeZim(g_raw, c)
znpy = makeZim(g_npy, c)
zspm = makeZim(g_spm, c)

zraw = nd.gaussian_filter(zraw, sigma=1)
znpy = nd.gaussian_filter(znpy, sigma=1)
zspm = nd.gaussian_filter(zspm, sigma=1)

###dz = np.abs(znpy)-np.abs(zraw)

affine = im_raw.get_affine()
Zimg = Nifti1Image(zraw, affine)
save_image(Zimg, join(rootpath, 'zraw.nii'))
Zimg = Nifti1Image(znpy, affine)
save_image(Zimg, join(rootpath, 'znpy.nii'))
Zimg = Nifti1Image(zspm, affine)
save_image(Zimg, join(rootpath, 'zspm.nii'))
"""
