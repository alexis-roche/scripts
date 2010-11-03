import numpy as np
import scipy.ndimage as nd
from nipy.io.imageformats import Nifti1Image as Image, load as load_image, save as save_image
from os.path import join 
import pylab 

# Editable globals
id = 5
em_iters = 5
save_images = True

# Non-editable globals
subjects = ['test', '016', '020', '023_1', '031', '03S0908'] 
datadir = join('C:\Documents and Settings\siemens_user\My Documents\My Dropbox\data', subjects[id])
savedir = 'd:\Alexis\tmp'
tissues = ['CSF','GM','WM','REST']
ntissues = len(tissues)


class Probmap(dict): 

    def __init__(self, parent=None): 
        if parent==None: 
            for tissue in tissues: 
                self[tissue] = None        
            self._mask = None
        else: 
            for tissue in tissues: 
                self[tissue] = parent[tissue].copy()
            self._mask = parent.mask()

	# SUM DES TISSUE
    def Z(self): 
        sum = self[tissues[0]].copy()
        for tissue in tissues[1:]:
            sum += self[tissue]
        return sum


    def mask(self): 
        if self._mask == None: 
            self._mask = np.where(self.Z() > 0)
        return self._mask 

    def normalize(self): 
        """
        Normalizing maps so that they sum up to one
        """
        mask = self.mask()
        Z = self.Z()[mask]
        for tissue in tissues:
            self[tissue][mask] /= Z 

	# Prob TIV
    def brain_ppm(self): 
        mask = self.mask()
        P = np.zeros(self['REST'].shape)
        P[mask] = 1.-self['REST'][mask]
        return P

        
def load_probmap(id): 
    """
    Example: 
      pm = load_probmap('PostMap_1000.img')
    """
    pm = Probmap()
    for tissue in tissues: 
        fname = tissue+id
        im = load_image(join(datadir, fname))
        tmp = im.get_data().copy()
        pm[tissue] = tmp
    pm.normalize()
    return pm 


class Gaussian(): 
    """
    Multivariate normal implementation.
    """
    def __init__(self, mu, sigma, method='eigh'): 
        tiny = 1e-30
        self.mu = mu
        self.sigma = sigma 
        
        if method == 'eigh': 
            # Eigenvalue decomposition approach
            d, P = np.linalg.eigh(sigma) ## sigma == P*d*P'
            s = np.maximum(np.sqrt(d), tiny) ## sigma_inv = (P*(1/s))*((1/s)*P') = Linv'*Linv
            self.Linv = np.dot(np.diag(1/s), P.T)
            self.norma = 1./np.maximum(np.prod(s), tiny)

        else: 
            # Cholesky decomposition approach
            L = np.linalg.cholesky(sigma) ## sigma == L*L'
            self.Linv = np.linalg.inv(L) ## sigma_inv = Linv'*Linv
            self.norma = 1./np.maximum(tiny, np.sqrt(np.prod(np.diag(L))))
            

    def pdf(self, x): 
        """
        x is assumed dim by nvoxels, where dim is the multivariate dim
        """
        tmp = np.dot(self.Linv, x) ## tmp is dim-by-nvoxels 
        tmp = (tmp*tmp).sum(0) # Mahalanobis distance x'*inv_sigma*x
        return self.norma*np.exp(-.5*tmp)
        

def e_step(gaussians, prior, data, posterior=None): 
    """
    posterior = e_step(gaussians, prior, data, posterior=None)    

    data are assumed masked. 
    """
    if posterior==None: 
        posterior = Probmap(prior)

    mask = prior.mask()
    for tissue in tissues: 
        posterior[tissue][mask] = prior[tissue][mask] * gaussians[tissue].pdf(data)
        
    posterior.normalize()
    return posterior
        

def m_step(posterior, data):
    """
    gaussians = m_step(posterior, data)
    
    data are assumed masked. 
    """
    gaussians = {}
    for tissue in tissues: 
        P = posterior[tissue][posterior.mask()]
        Z = P.sum()
        tmp = data*P
        mu = tmp.sum(1)/Z
        mu_ = mu.reshape(2,1)
        sigma = np.dot(tmp, data.T)/Z - np.dot(mu_, mu_.T)
        gaussians[tissue] = Gaussian(mu, sigma)
    return gaussians

# Main Program
# Load prior probability map 
print('Loading data...')
prior = load_probmap('_1000Prior.img')

# Load image data and apply masking
im1 = load_image(join(datadir, 'BiasCorIm.img'))
im2 = load_image(join(datadir, 'DistanceMap.img'))
data = np.asarray([im.get_data()[prior.mask()] for im in [im1,im2]])

# Save image data in nifti for visualization with `anatomist`
if save_images: 
    print('Saving prior image...')
    save_image(im1, join(savedir, 'BiasCorIm.nii'))
    im = Image(prior.brain_ppm(), affine=im1.get_affine())
    save_image(im, join(savedir, 'Prior_Brain.nii'))

# Allocate posterior probability map and initialize it from the prior
posterior = Probmap(prior)

# EM algorithm: refine posterior map
print('Starting EM...')
for iter in range(em_iters): 
    print('  M-step...')
    gaussians = m_step(posterior, data) 
    print('  E-step...')
    posterior = e_step(gaussians, prior, data, posterior=posterior)
    if save_images: 
        print('  Saving current posterior image...')
        im = Image(posterior.brain_ppm(), affine=im1.get_affine())
        save_image(im, join(savedir, 'MEM_Posterior_Brain_iter'+str(iter)+'.nii'))



    
