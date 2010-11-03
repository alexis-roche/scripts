import numpy as np 
from nibabel import load, save
from nipy.neurospin.image import Image, asNifti1Image
from nipy.neurospin.registration import *
from os.path import join 
from glob import glob

from isola import Isola, mat_info

DATADIR = join('d:\\', 'home', 'Alexis', 'data', 'remy')
REFERENCE = '2_axial_FL_AD_ROI.nii'

ID = 5
TYPE = 'defrand'
#TYPE = 'affine'
NTRIALS = 5 
SIMI = 'crl1' 
INTERP = 'pv'


if ID == 3: 
    PATIENT = '3_axial_FLIPPED_ADJUSTED.nii'
    THRESHOLD = 300 
elif ID == 4: 
    PATIENT = '4LSAG25_5_sagittal_TSE_ADJUSTED.nii'
    THRESHOLD = 100
elif ID == 5:
    PATIENT = '05_3DT1_ACPC_Aligned.nii'
    THRESHOLD = 0 


# Create a registration instance
print('Creating matcher instance')
J = Image(load(join(DATADIR, PATIENT)))
J = J[np.where(J.data>THRESHOLD)] ## masking 
I = Image(load(join(DATADIR, REFERENCE)))
R = IconicRegistration(I, J)
#R.set_source_fov(fixed_npoints=64**3)
R.similarity = SIMI
R.interp = INTERP

# Affine registration
if 'id' in TYPE:
    T = Affine()

elif 'rig' in TYPE:
    T = R.optimize(Rigid())

elif 'aff' in TYPE:   
    T = R.optimize(Affine())

# Spline-based registration
elif 'def' in TYPE: 
    defmodes = Image(load(join(DATADIR, 'deformation_modes.nii')))
    std = np.loadtxt(join(DATADIR, 'deformation_modes_std.txt'))
    nparams = np.loadtxt(join(DATADIR, 'deformation_modes_norm_params.txt'))
    identity = std*nparams[:,-3]
    T = GridTransform(I, defmodes.data.reshape(list(I.shape)+[7,3]))
    # freeze mean deformation component
    T._param[0] = 1
    T._free_param_idx = [1,2,3,4,5,6]

    if 'defrand' in TYPE: 
        best_param = np.load('t3_def_crl1.npy')
        best_simi = 0.0 
        for i in range(NTRIALS): 
            T.param = best_param + std*np.random.normal(std.size)
            T = R.optimize(T)
            optimum = R.eval(T)
            if optimum > best_simi: 
                print(' *** New best ***') 
                best_simi = optimum 
                best_param = T.param
            else: 
                print(' *** Unbeaten optimum ***') 
                T.param = best_param 
    else: 
        T.param = identity 
        T = R.optimize(T)


# Create matcher instance
fname = 't'+str(ID)+'_'+TYPE+'_'+SIMI+'_'+INTERP
outfile = join(DATADIR, fname+'.nii') 
print('Saving resampled source in: %s' % outfile)
Jt = J.transform(T, reference=I)
save(asNifti1Image(Jt), outfile)
