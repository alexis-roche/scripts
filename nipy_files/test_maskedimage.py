from nipy.neurospin.image import *
from nibabel import load 
from os.path import join


datadir = 'D:\home\Alexis\data\delphine\zozo'
path = join(datadir, 'BiasCorIm.img')
I = Image(load(path))

# Usecase 1: image masking
I1 = I[np.where(I.data>100)]
I1.values()
#J = I1.transform(np.eye(4))
J = I1.set(I1.values()+1.)

# Usecase 2: define a regular image patch (subgridding)
I2 = I[5:10:2, 5:10:2, 5:10:2]
I2.values()
J = I2.transform(np.eye(4))
J = I2.set(I2.values()+1.)

