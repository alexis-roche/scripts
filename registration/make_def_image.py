import numpy as np 
from nibabel import load, save
from nipy.neurospin.image import Image, asNifti1Image
from os.path import join 
from glob import glob

datadir = join('d:\\', 'home', 'Alexis', 'data', 'remy')
I = Image(load(join(datadir, '2_axial_FL_AD_ROI.nii')))

defmodes = glob(join(datadir, 'NEW_SDF', 'SDF_*.nii'))
data = np.zeros(list(I.shape)+[len(defmodes)])
for i in range(len(defmodes)):
    print('image: %d (%s)' % (i, defmodes[i]))
    m = Image(load(defmodes[i]))
    mm = m.transform(np.eye(4), reference=I)
    data[:,:,:,i] = mm.data

save(asNifti1Image(Image(data, I.affine)), join(datadir, 'deformation_modes.nii'))
