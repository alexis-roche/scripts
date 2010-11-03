import numpy as np
from os.path import join, split
import sys
import time
from glob import glob 

from nipy.io.imageformats import load as load_image, save as save_image
from nipy.neurospin.registration import FmriRealign4d

# Create Nifti1Image instances from both input files
rootpath = 'D:\\home\\Alexis\\data\\karla'
runnames = glob(join(rootpath, 'fms*.nii'))
runs = [load_image(run) for run in runnames]

## DEBUG 
"""
idx = [4,5,6,7]
runs = [runs[i] for i in idx]
runnames = [runnames[i] for i in idx]
"""
print runnames

# Do the job 
R = FmriRealign4d(runs, tr=2.4, slice_order='ascending', interleaved=False)
R.correct_motion()
corr_runs = R.resample()

# Save images 
for i in range(len(runs)):
    aux = split(runnames[i])
    save_image(corr_runs[i], join(aux[0], 'new_ra'+aux[1]))


