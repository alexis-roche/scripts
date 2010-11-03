import numpy as np 
import pylab 
import os

from nipy.neurospin.image import Image, load_image, save_image
from nipy.neurospin.segmentation.mrf_module import finalize_ve_step


s_idx = 1 # subject index
r_idx = 2 # method index

niters_em = 10
niters_vem = 5
niters = niters_em + niters_vem 
beta_val = 0.2
betas = np.zeros(niters)
betas[niters_em:] = beta_val

subjects = ['Subject'+str(i) for i in (1,2,3,4,5,6)]
regs = ['Bene', 'Annot_T1', 'Annot_T2']
id_priors = ['BPriorMap', 'PriorMap', 'PriorMap']
id_posterior = 'PosteriorMap'

base = os.path.join('D:\home', 'Alexis', 'data', 'vembase')
baseres = os.path.join('D:\home', 'Alexis', 'data', 'vembase_res')

print base

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

# Load ppm
def load_ppm(): 
    # Read input files 
    for i in range(ntissues):
        fname = os.path.join(subdatadir, id_prior+'_EM_'+str(i)+'.nii')
        print(fname)
        if i == 0: 
            im = load_image(fname)
            affine = im.affine
            data = np.zeros(list(im.shape)+[ntissues])
            data[:,:,:,0] = im.data
        else: 
            data[:,:,:,i] = load_image(fname).data
    
    #ppm = Image(data, affine)

    # Normalize and mask ppms 
    psum = data.sum(3)
    X,Y,Z = np.where(psum>0)
    for i in range(ntissues): 
        data[X,Y,Z,i] /= psum[X,Y,Z]
    mask = (X.astype('uint'), Y.astype('uint'), Z.astype('uint'))

    return data, mask, affine 

# Load (bias corrected) mri scan 
def load_mri(): 
    fname = os.path.join(datadir, subjects[s_idx]+'.nii')
    print(fname)
    return load_image(fname)

# VM-step 
def vm_step(ppm, data, mask): 
    mu = np.zeros(ntissues)
    sigma = np.zeros(ntissues)
    for i in range(ntissues):
        P = ppm[:,:,:,i][mask]
        Z = P.sum()
        tmp = data*P
        mu_ = tmp.sum()/Z
        sigma_ = np.sqrt(np.sum(tmp*data)/Z - mu_**2)
        mu[i] = mu_ 
        sigma[i] = sigma_
    return mu, sigma 

def gaussian(x, mu, sigma): 
    return np.exp(-.5*((x-mu)/sigma)**2)/sigma

# VE-step 
def ve_step(ppm, data, mask, mu, sigma, prior, alpha=1., beta=0.0): 
    """
    posterior = e_step(gaussians, prior, data, posterior=None)    

    data are assumed masked. 
    """
    lik = np.zeros(np.shape(prior))
    for i in range(ntissues): 
        lik[:,i] = prior[:,i] * gaussian(data, mu[i], sigma[i])

    # Normalize
    X, Y, Z = mask
    ppm[X, Y, Z] = lik 

    if beta == 0.0: 
        ppm_sum = ppm[mask].sum(1)
        for i in range(ntissues): 
            ppm[X, Y, Z, i] /= ppm_sum
 
    else: 
        print('  .. MRF correction')
        XYZ = np.array((X, Y, Z), dtype='int') 
        ppm = finalize_ve_step(ppm, lik, XYZ, beta)

    return ppm


def save_results(ppm, id_algo): 
    savedir = os.path.join(baseres, subjects[s_idx])
    for i in range(ntissues): 
        im = Image(ppm[:,:,:,i], affine)
        fname = id_posterior + '_' + regs[r_idx] + '_' + id_algo + '_' + str(i) + '.nii'
        save_image(im, os.path.join(savedir, fname))


# Main 
print('Loading priors...')
ppm, mask, _ = load_ppm()
print('Loading MRI data...')
im = load_mri()
data = im.data[mask] 
affine = im.affine

# Prior computation 
prior = ppm[mask]


# VEM algorithm 
for i in range(niters):

    if i == niters_em: 
        print('saving EM results...')
        save_results(ppm, 'EM')

    print('VEM iter %d/%d' % (i+1, niters))
    print('  VM-step...')
    mu, sigma = vm_step(ppm, data, mask) 
    print('  VE-step...')
    ppm = ve_step(ppm, data, mask, mu, sigma, prior, beta=betas[i]) 

print('saving EM results...')
save_results(ppm, 'VEM')


# Display results
def display_ppm(ppm): 
    pylab.pink() 
    for i in range(ntissues): 
        display(ppm[:,:,:,i])

# Save results
print('Saving results...')

