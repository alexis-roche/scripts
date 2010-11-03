import numpy as np
import nibabel as ni
from scipy.ndimage import rotate
from nipy.neurospin.registration import register, transform

def add_padding(data,pad=10,value=0):
    new_data=value*np.ones((data.shape[0]+2*pad,data.shape[1]+2*pad,data.shape[2]+2*pad),dtype=data.dtype)
    new_data[pad:pad+data.shape[0],pad:pad+data.shape[1],pad:pad+data.shape[2]]=data
    return new_data

#synthesize volume
S0=255*np.ones((50,50,50))#.astype('uint16')
S0=add_padding(S0,10,100)
S0=add_padding(S0,40)   
#S0=rotate(S0,5,reshape=False)

S1=rotate(S0,30,reshape=False)
S0img=ni.Nifti1Image(S0,np.eye(4))
ni.save(S0img,'S0img.nii.gz')

S1img=ni.Nifti1Image(S1,np.eye(4))
ni.save(S1img,'S1img.nii.gz')

#register S1 to S0
T=register(S1img,S0img,interp='pv')

#save S1 as NS1 after registration
NS1img=transform(S1img, T)
ni.save(NS1img,'NS1img.nii.gz')

