#!/usr/bin/env python 

import fff2.registration as registration
import fff2.utils.io as io
from os.path import join
from os import system
import sys
import time
import numpy as np

"""
Script for image registration using fff on Irina's data.
"""

subject = '/neurospin/lrmn/database_nmr/sujet11/anatomy/nobias_sujet11.ima'
template = '/neurospin/lnao/Panabase/vincent/dataBase_brainvisa/nmr/normalization/template/T1.ima'

# Registration params
source = subject
target = template
toresample = 'source'
iolib = 'aims'
similarity = 'correlation ratio'
method = 'powell'
search = 'affine 3D'

print ('Source brain: %s' % source)
print ('Target brain: %s' % target)
print ('I/O library: %s' % iolib)
print ('Similarity measure: %s' % similarity)


# Get data
print('Fetching image data...')
I = io.image()
I.load(source, iolib=iolib)
J = io.image()
J.load(target, iolib=iolib)

## Info
print 'source dimensions: ', I.array.shape
print 'source voxel size: ', I.voxsize
print 'target dimensions: ', J.array.shape
print 'target voxel size: ', J.voxsize

# Setup registration algorithm
print('Setting up registration...')
matcher = registration.iconic(I, J) ## I: source, J: target 
matcher.set(subsampling=[4,4,4], similarity=similarity)

# Register
print('Starting registration...')
tic = time.time()
##T, t = matcher.optimize(method=method, search='rigid 3D')
##T, t = matcher.optimize(method=method, search='similarity 3D', start=t)
t = None
T, t = matcher.optimize(method=method, search='affine 3D', start=t)
toc = time.time()
print('  Optimization time: %f sec' % (toc-tic))

# Resample image
print('Resampling image...')
tic = time.time()
It = io.image()
if toresample=='target':
    It.clone(I)
    It.set_array(matcher.resample(T), toresample='target')
else:
    It.clone(J)
    It.set_array(matcher.resample(T))
toc = time.time()
print('  Resampling time: %f sec' % (toc-tic))

# Save resampled source
print('Saving resampled image...')
outfile = 'toto'
print ('Saving resampled source in: %s' % outfile + '.ima')
It.save(outfile + '.ima')

# Convert to AIMS format and save
# DIRTY HACK
Tv = matcher.voxel_transform(T) ## canonic voxel coordinate systems
Dsrc_inv = np.diag(1/np.diag(matcher.source_transform))
Dtgt = np.diag(1/np.diag(matcher.target_transform_inv))
Ta = np.dot(np.dot(Dtgt, Tv), Dsrc_inv)

f = open(outfile+'.trm', 'w')
f.write(Ta[0,3].__str__()+'\t'+ Ta[1,3].__str__()+'\t'+Ta[2,3].__str__()+'\n')
f.write(Ta[0,0].__str__()+'\t'+ Ta[0,1].__str__()+'\t'+Ta[0,2].__str__()+'\n')
f.write(Ta[1,0].__str__()+'\t'+ Ta[1,1].__str__()+'\t'+Ta[1,2].__str__()+'\n')
f.write(Ta[2,0].__str__()+'\t'+ Ta[2,1].__str__()+'\t'+Ta[2,2].__str__()+'\n')
f.close()

#
cmd = 'AimsResample -m '+outfile+'.trm -i '+source+' -o '+outfile+'_aims.ima -r '+target
sytem(cmd)


# Save transfo
##np.savez(outfile, Ta, T, matcher.source_transform, matcher.target_transform_inv)





