import numpy as np 
from nipy.neurospin.segmentation.mrf_module import finalize_ve_step

dx, dy, dz, K = 10, 10, 10, 4

ppm = np.random.rand(dx, dy, dz, K)

XYZ = np.mgrid[dx/2-dx/4:dx/2+dx/4,dy/2-dy/4:dy/2+dy/4,dz/2-dz/4:dz/2+dz/4]
XYZ = np.reshape(XYZ, (3, np.prod(XYZ.shape[1:])))

lik = np.random.rand(XYZ.shape[1], K)

finalize_ve_step(ppm, lik, XYZ, 0.)
