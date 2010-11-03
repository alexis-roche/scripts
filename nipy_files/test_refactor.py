import numpy as np 
from nibabel import load 
from nipy.neurospin.image import Image
from os.path import join
import scipy.ndimage as nd


datadir = 'D:\home\Alexis\data\delphine\zozo'
path = join(datadir, 'BiasCorIm.img')

# Usecase 0: load an image 
I = Image(load(path))

# Usecase 1: image masking
I1 = I.set_mask(np.where(I.data>100))

# Usecase 2: define a regular image patch (subgridding)
I2 = I[5:10:2, 5:10:2, 5:10:2]

# Usecase 3: get all masked values in an image 
tmp = I2.values()

# Usecase 4: image interpolation 
XYZ = 100*np.random.rand(3,10)
tmp = I.values(XYZ)

# Usecase 4: move an image using an affine transformation 
T = np.eye(4)
T[0:3,3] = 10*np.random.rand(3)
J = I.transform(T)

# Usecase 5: set image values
I2 = I2.set(I2.data**2+2)
I1 = I1.set(I1.values()**2+2)
