from nipy.neurospin.image import * 
from nipy.neurospin.registration import * 
from nipy.neurospin.registration.grid_transform import * 

from nipy.testing import anatfile
from nipy.io.imageformats import load 


I = Image(load(anatfile))
J = Image(load(anatfile))

R = IconicRegistration(I, J) 
R.set_source_fov(fixed_npoints=64**3)

# TODO: handle cpts in voxel coordinates, test negative values
def new_apply_affine(T, xyz):
    """
    XYZ = apply_affine(T, xyz)

    T is a 4x4 matrix.
    xyz is a Nx3 array of 3d coordinates stored row-wise.  
    """
    XYZ = np.dot(xyz, T[0:3,0:3])
    XYZ[:,0] += T[0,3]
    XYZ[:,1] += T[1,3]
    XYZ[:,2] += T[2,3]
    return XYZ 

app = new_apply_affine

slices = [slice(0, s.stop, 4*s.step) for s in R._slices]
cpts_grid = np.rollaxis(np.mgrid[slices], 0, 4)

T = I.affine
###T = np.eye(4)

cpts = app(T, cpts_grid)
tmp = app(inverse_affine(T), cpts) # should == cpts_grid


Ts = SplineTransform(I, cpts, sigma=4.5)

"""
from_world = inverse_affine(Ts._toworld)
tmp = app(from_world, cpts)
tmp = np.round(tmp).astype('int')
"""

delta = tmp - cpts_grid
print delta.min()
print delta.max()

