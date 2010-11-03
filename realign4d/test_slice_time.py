from nipy.neurospin.registration.realign4d import * 
import numpy as np 
from pylab import * 


##slice_order = range(10)
##slice_order = range(11)
slice_order = [0,5,1,6,2,7,3,8,4,9]
##slice_order = [0,6,1,7,2,8,3,9,4,10,5]

nslices = len(slice_order)

Z = np.arange(0,2*nslices)
T = interp_slice_order(Z, slice_order)

plot(Z, T)
