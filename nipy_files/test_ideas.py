import nipy.io.imageformats as brifti
import numpy as np 
from scipy import ndimage 

_interp_order = 1 
_background = 0 


class Image(object): 
    """
    Image class. 
    
    An image is a real-valued mapping from a regular 3D lattice that
    can be masked, in which case only in-mask image values are kept in
    memory.
    """

    def __init__(self, data, affine, world=None, mask=None, shape=None, background=_background): 
        """ 
        The base image class.

        Parameters
        ----------

        data: ndarray
            n dimensional array giving the embedded data, with the first
            three dimensions being spatial.

        affine: ndarray 
            a 4x4 transformation matrix from voxel to world.

        world: string, optional 
            An identifier for the real-world coordinate system, e.g. 
            'scanner' or 'mni'. 
            
        background: number, optional 
            Background value (outside image boundaries).  
        """

        if mask == None: 
            self._shape = data.shape
        else: 
            self._shape = shape
        self._order=['C','F'][data.flags['F_CONTIGUOUS']]
        self._values = np.ravel(data, order=self._order)
        self._set_affine(affine)
        self._inv_affine = inverse_affine(affine)
        if mask == None: 
            self._mask = None
        else: 
            self._mask = validate_coords(mask)
        self._background = background
        self._set_world(world)


    def __getitem__(self, slices):
        """
        Extract an image patch corresponding to the specified bounding
        box. Compute the affine transfotrmation accordingly. 
        """
        steps = map(lambda x: max(x,1), [s.step for s in slices])
        starts = map(lambda x: max(x,0), [s.start for s in slices])
        t = np.diag(np.concatenate((steps,[1]),1))
        t[0:3,3] = starts
        affine = np.dot(self._affine, t)
        return Image(self._get_data()[slices], affine, world=self._world)

    def _get_shape(self):
        return self._shape

    def _get_dtype(self):
        return self._values.dtype

    def _get_values(self):
        return self._values

    def _get_data(self): 
        
        if not self._mask: 
            return np.reshape(self._values, self._shape, order=self._order)
        
        output = np.zeros(self._shape, dtype=self._get_dtype())
        if not self._background == 0:  
            output += self._background

        output[self._mask] = self._values
        return output

    def _get_affine(self):
        return self._affine

    def _set_affine(self, affine):
        affine = np.asarray(affine).squeeze()
        if not affine.shape==(4,4):
            raise ValueError('Not a 4x4 transformation matrix')
        self._affine = affine

    def _get_inv_affine(self):
        return self._inv_affine

    def _get_world(self):
        return self._world

    def _set_world(self, world):
        if not world == None: 
            world = str(world)
        self._world = world

    def _get_mask(self): 
        return self._mask

    def __call__(self, coords=None, grid_coords=False, 
                 dtype=None, interp_order=_interp_order):
        """ 
        Return interpolated values at the points specified by coords. 

        Parameters
        ----------
        coords: sequence of 3 ndarrays, optional
            List of coordinates. If missing, the image mask 
            is assumed. 
                
        grid_coords: boolean, optional
            Determine whether the input coordinates are in the
            world or grid coordinate system. 
        
        Returns
        -------
        output: ndarray
            One-dimensional array. 

        """
        if coords == None: 
            return self._values

        if dtype == None: 
            dtype = self._get_dtype()

        # Convert coords into grid coordinates
        X,Y,Z = validate_coords(coords)
        if not grid_coords:
            X,Y,Z = apply_affine(self._inv_affine, (X,Y,Z))
        XYZ = np.c_[(X,Y,Z)]

        # Avoid interpolation if coords are integers
        if issubclass(XYZ.dtype.type, np.integer):
            I = np.where(((XYZ>0)*(XYZ<self._shape)).min(1))
            output = np.zeros(XYZ.shape[0], dtype=dtype)
            if not self._background == 0:  
                output += self._background
            if I[0].size: 
                output[I] = self._get_data()[X[I], Y[I], Z[I]]

        # Otherwise, interpolate the data
        else: 
            output = ndimage.map_coordinates(self._get_data(), 
                                             XYZ.T, 
                                             order=interp_order, 
                                             cval=self._background,
                                             output=dtype)
            
        return output 

    


    shape = property(_get_shape)
    dtype = property(_get_dtype)
    values = property(_get_values)
    affine = property(_get_affine, _set_affine)
    inv_affine = property(_get_inv_affine)
    world = property(_get_world, _set_world)
    data = property(_get_data)
    mask = property(_get_mask)



"""
Util functions
"""
def load_image(fname): 
    im = brifti.load(fname)
    return Image(im.get_data(), im.get_affine())

def save_image(Im, fname):
    im = brifti.Nifti1Image(Im.data, Im.affine)
    brifti.save(im, fname)

def mask_image(im, mask, background=None): 
    """
    Return a masked image. 
    """
    mask = validate_coords(mask)
    if not background: 
        background = im._background
    return Image(im._get_data()[mask], im._affine, world=im._world,
                 mask=mask, shape=im._shape, background=background)

def set_image(im, values): 
    
    if not len(values) == len(im._values): 
        raise ValueError('Input array shape inconsistent with image values')

    if im._mask:
        return Image(values, affine=im._affine, world=im._world, 
                     mask=im._mask, shape=im._shape, background=im._background)
    else: 
        values = np.reshape(values, im._shape, 
                            order=['C','F'][values.flags['F_CONTIGUOUS']])
        return Image(values, affine=im._affine, world=im._world)


def move_image(im, transform, target=None, 
               dtype=None, interp_order=_interp_order):
    """
    Apply a spatial transformation to bring the image into the
    same grid as the specified target image. 
    
    transform: world transformation
    
    target: target image, defaults to input. 
    """
    if target == None: 
        target = im
        
    if dtype == None: 
        dtype = im._get_dtype()

    # Grid-to-grid transformation from target to source
    t = np.dot(im._inv_affine, np.dot(inverse_affine(transform), target._affine))
    
    # Perform image resampling 
    data = im._get_data()
    output = np.zeros(data.shape, dtype=dtype)
    ndimage.affine_transform(data, t[0:3,0:3], offset=t[0:3,3],
                             output_shape=target._shape,
                             order=interp_order, cval=im._background, 
                             output=output)
    return Image(output, affine=target._affine, world=im._world, 
                 background=im._background)


def validate_coords(coords): 
    """
    Convert coords into a tuple of ndarrays
    """
    X,Y,Z = [np.asarray(coords[i]) for i in range(3)]
    return X,Y,Z

def inverse_affine(affine):
    return np.linalg.inv(affine)

def apply_affine(affine, XYZ):
    """
    Parameters
    ----------
    affine: ndarray 
        A 4x4 matrix representing an affine transform. 

    coords: tuple of ndarrays 
        tuple of 3d coordinates stored row-wise: (X,Y,Z)  
    """
    tXYZ = np.array(XYZ, dtype='double')
    if tXYZ.ndim == 1: 
        tXYZ = np.reshape(tXYZ, (3,1))
    tXYZ[:] = np.dot(affine[0:3,0:3], tXYZ[:])
    tXYZ[0,:] += affine[0,3]
    tXYZ[1,:] += affine[1,3]
    tXYZ[2,:] += affine[2,3]
    return tuple(tXYZ)


def sample(data, coords, order=_interp_order, dtype=None, 
           background=_background): 

    from image_module import cspline_transform, cspline_sample3d
    
    if dtype == None: 
        dtype = data.dtype

    X, Y, Z = tuple(coords)
    npts = X.size

    cbspline = cspline_transform(data)
    output = np.zeros(npts, dtype='double')
    output = cspline_sample3d(output, cbspline, X, Y, Z)
    output.astype(dtype)
    
    return output



def resample(data, affine, shape=None, order=_interp_order, dtype=None, 
             background=_background): 

    from image_module import cspline_resample3d
    
    if shape == None:
        shape = data.shape
    if dtype == None: 
        dtype = data.dtype

    return cspline_resample3d(data, shape, affine, dtype=dtype)




"""
Main 
"""

from os.path import join 
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


datadir = 'D:\home\Alexis\data\delphine\zozo'
path = join(datadir, 'BiasCorIm.img')

# Load an image 
I = load_image(path)

# Mask an imahe 
mask = (100*np.random.rand(3,10)).astype('int')
##mask = np.where(I.data>1000)
J1 = mask_image(I, mask)

# Modify image values
J2 = set_image(I, I().astype('double')**2/100)

# Apply a spatial transform to an image 
t = np.eye(4)
t[:3,2] = [5,-6,4]
J = move_image(I, t)

"""
# Crop an image 
J = I[0:14:2,0:11,4:34:5]

# Spatially transform an image 
t = np.eye(4)
t[:3,2] = [5,-6,4]
J = J.move(t)

# Modify image values
J = I.fill(I().astype('double')**2/100)
"""
