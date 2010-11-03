from nipy.neurospin.image import load_image, save_image
from nipy.neurospin.registration import IconicRegistration
from nipy.utils import example_data
from os.path import join 
import numpy as np 
import pylab 



# Display function 
def display(im, slice=80, threshold=None): 
    pylab.figure()
    pylab.pink()
    array = im.data
    if threshold==None: 
        pylab.imshow(array[:,slice,:])
    else:
        pylab.imshow(array[:,slice,:]>threshold)



"""
Main 
"""

# Load images 
I = load_image(example_data.get_filename('neurospin',
                                         'sulcal2000',
                                         'nobias_ammon.nii.gz'))
J = load_image(example_data.get_filename('neurospin',
                                         'sulcal2000',
                                         'nobias_anubis.nii.gz'))


# Create a registration instance
R = IconicRegistration(I, J)
R.set_source_fov(fixed_npoints=64**3)
R.similarity = 'llr_mi'
T = np.eye(4)
#T[0:3,3] = [4,5,6]

print R.eval(T)

