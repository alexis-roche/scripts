import numpy as np 
import pylab 
import os

from nipy.io.imageformats import load, save
from nipy.neurospin.image import Image, Nifti1Image
from nipy.neurospin.segmentation.vem import vem

# Editable parameters 
s_idx = 0 # subject index 
r_idx = 2 # method index 
noise = 'gauss' # choose 'gauss' or 'laplace'
niters_em = 10 
niters_vem = 5
beta_val = 0.2
#base = os.path.join('D:\DELPHINE', 'TestVEM')
#baseres = os.path.join('D:\DELPHINE', 'TestVEM', 'results')
base = os.path.join('D:\home', 'Alexis', 'data', 'vembase')
baseres = os.path.join('D:\home', 'Alexis', 'data', 'vembase_res')

# Non-editable parameters 
betas_em = np.zeros(niters_em)
betas_vem = beta_val*np.ones(niters_vem)
subjects = ['Subject'+str(i) for i in (1,2,3,4,5,6)]
regs = ['Bene', 'Annot_T1', 'Annot_T2']
id_priors = ['BPriorMap', 'PriorMap', 'PriorMap']
id_posterior = 'PosteriorMap'
datadir = os.path.join(base, subjects[s_idx])
subdatadir = os.path.join(datadir, regs[r_idx])
id_prior = id_priors[r_idx]
tissues = ['CSF','WM','GM','REST']
ntissues = len(tissues)

# Display function 
def display(array, slice=80, threshold=None): 
    pylab.figure()
    if threshold==None: 
        pylab.imshow(array[:,slice,:])
    else:
        pylab.imshow(array[:,slice,:]>threshold)

def display_ppm(ppm): 
    pylab.pink() 
    for i in range(ntissues): 
        display(ppm[:,:,:,i])


# Load ppm
def load_ppm(): 
    # Read input files 
    for i in range(ntissues):
        fname = os.path.join(subdatadir, id_prior+'_EM_'+str(i)+'.nii')
        print(fname)
        if i == 0: 
            im = Image(load(fname))
            affine = im.affine
            data = np.zeros(list(im.shape)+[ntissues])
            data[:,:,:,0] = im.data
        else: 
            data[:,:,:,i] = Image(load(fname)).data
    
    #ppm = Image(data, affine)

    # Normalize and mask ppms 
    psum = data.sum(3)
    X,Y,Z = np.where(psum>0)
    for i in range(ntissues): 
        data[X,Y,Z,i] /= psum[X,Y,Z]
    mask = (X.astype('uint'), Y.astype('uint'), Z.astype('uint'))

    # For now, we may need to reduce the mask because the vem module
    # does not handle boundary conditions
    X,Y,Z = mask
    I = np.where((X>0)*(X<im.shape[0]-1)*(Y>0)*(Y<im.shape[1]-1)*(Z>0)*(Z<im.shape[2]-1))
    mask = X[I], Y[I], Z[I]
    
    return data, mask, affine 

# Load (bias corrected) mri scan 
def load_mri(): 
    fname = os.path.join(datadir, subjects[s_idx]+'.nii')
    print(fname)
    return Image(load(fname))

# Save tissue ppms
def save_results(ppm, id_algo, noise): 
    savedir = os.path.join(baseres, subjects[s_idx])
    tag = id_algo
    if noise == 'laplace': 
        tag += '_laplace'
    for i in range(ntissues): 
        im = Image(ppm[:,:,:,i], affine)
        fname = id_posterior + '_' + regs[r_idx] + '_' + tag + '_' + str(i) + '.nii'
        save(Nifti1Image(im), os.path.join(savedir, fname))


# Main 
print('Processing %s (%s) with %s model' % (subjects[s_idx], regs[r_idx], noise))
print('Loading priors...')
ppm, mask, _ = load_ppm()
print('Loading MRI data...')
im = load_mri()
#data = im.data[mask] 
affine = im.affine

# Prior computation 
#ppm = ppm[mask]

# VEM algorithm 
"""
ppm, mu, sigma = vem(ppm, im.data, mask, 
                     betas=betas_em, noise=noise)
"""
"""
print('saving EM results...')
save_results(ppm, 'EM', noise)
"""

ppm, mu, sigma = vem(ppm, im.data, mask, 
                     betas=betas_vem, noise=noise, 
                     prior=True, copy=True)

"""
print('saving EM results...')
save_results(ppm, 'VEM', noise)
"""
