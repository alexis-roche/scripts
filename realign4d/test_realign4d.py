import numpy as np 


"""
from nipy.neurospin.image.image_module import * 
x = np.random.rand(3,4,5,6)

c = cspline_transform(x)

r = np.zeros(1)

cspline_sample4d(r, c, 1, 2, 4, 4)
print r[0]
print x[1,2,4,4]
"""

from nipy.neurospin.registration.fmri_realign4d import * 
from nipy.io.imageformats import load, save
from nipy.testing import funcfile


im = load(funcfile) 

###I = Image4d(im.get_data(), im.get_affine(), tr=2., tr_slices=0.0)
I = Image4d(im.get_data(), im.get_affine(), tr=2.)
R = Realign4d(I, transforms=None)

t = 0 
X, Y, Z = grid_coords(R.xyz, R.transforms[t], 
                      R.from_world, R.to_world)
T = R.from_time(Z, R.timestamps[t])

Ia = I.array
Ja = R.resample()



from pylab import * 

x = 0
y = 10 
z = 1
i = Ia[x,y,z,:]
j = Ja[x,y,z,:]

t = np.arange(Ia.shape[3])

plot(t, i)
plot(t, j)
