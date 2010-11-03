import numpy as np
import scipy.ndimage as nd
from nipy.io.imageformats import Nifti1Image as Image, load as load_image, save as save_image
import os 
import pylab 

id = 5

subjects = ['test', '016', '020', '023_1', '031', '03S0908'] 
datadir = os.path.join('D:\home\AR203069\data\delphine', subjects[id])
tissues = ['CSF','GM','WM','REST']
brain_tissues = ['CSF','GM','WM']
def_size = 2

def display(P, slice=80, threshold=None): 
    pylab.figure()
    if threshold==None: 
        pylab.imshow(P[:,slice,:])
    else:
        pylab.imshow(P[:,slice,:]>threshold)

def display_all(Pdict, slice=80): 
    for tissue in tissues: 
        display(Pdict[tissue], slice)

def read_ppms():
    """
    Open PPMs (White Matter, Gray Matter, CSF, Rest)
    """
    Pdict = {}
    for tissue in tissues: 
        fname = tissue+'PostMap_1000.img'
        im = load_image(os.path.join(datadir, fname))
        Pdict[tissue] = im.get_data()/1000.
    affine = im.get_affine()
    return Pdict, affine

def _sum(Pdict): 
    """
    Sum up values across tissues 
    """
    S = None
    for tissue in tissues:
        if S == None: 
            S = Pdict[tissue].copy()
        else:
            S += Pdict[tissue]
    return S

def normalize(Pdict): 
    print('  Normalizing maps...')
    S = _sum(Pdict)
    mask = np.where(S > 0)
    for tissue in tissues: 
        Pdict[tissue][mask] /= S[mask]
    return Pdict, mask 

def closing(P, size=def_size):
    return nd.grey_closing(P, size=[size,size,size]) 

def opening(P, size=def_size):
    return nd.grey_opening(P, size=[size,size,size]) 

def dilation(P, size=def_size):
    return nd.grey_dilation(P, size=[size,size,size]) 

def erosion(P, size=def_size):
    return nd.grey_erosion(P, size=[size,size,size]) 

def median_filter(P, size=def_size): 
    return nd.median_filter(P, size=[size,size,size])

def mean_filter(P, size=def_size): 
    return nd.filters.uniform_filter(P, size=[size,size,size])

def gaussian_filter(P, size=def_size): 
    size /= 2.*np.sqrt(2*np.log(2)) # convert implicit FWHM into sigma units
    return nd.filters.gaussian_filter(P, sigma=[size,size,size]) 

def log_opinion_pool(P, mask, size=def_size, tiny=1e-20, kind='gaussian'): 
    """
    Compute P(k) = prod_i P_i(k) where i represents neighbouring
    voxels. 
    """
    P = np.log(np.maximum(tiny, P))
    if kind=='mean': 
        P = mean_filter(P, size)
    elif kind=='gaussian': 
        P = gaussian_filter(P, size)
    elif kind=='median': 
        P = median_filter(P, size)
    else: 
        print('not a valid filter')
    tmp = np.exp(P[mask])
    P = np.zeros(P.shape)
    P[mask] = tmp 
    return P 

def geometry_cleaner(ppm, brain_prior, mask): 
    for tissue in brain_tissues:
        ppm[tissue][mask] *= brain_prior[mask]
    ppm['REST'][mask] *= 1.-brain_prior[mask]
    return ppm

def brain_ppm(Pdict, mask): 
    P = np.zeros(Pdict['REST'].shape)
    P[mask] = 1.-Pdict['REST'][mask]
    return P


# Main program
print('Opening raw PPMs...')
raw_ppm, affine = read_ppms()
raw_ppm, raw_mask = normalize(raw_ppm)

# Dummy instructions to force pylab in 'show' mode
"""
tmp = np.random.rand(10)
pylab.plot(tmp)
pylab.show()
pylab.close()
"""

# Display initial (dirty) maps
pylab.pink()

# Allocate the output ppms
ppm = {}
for tissue in tissues: 
    ppm[tissue] = raw_ppm[tissue].copy()
mask = raw_mask     

# Clean brain tissue maps 
print('Decrease brain tissue probabilities outside brain...')
brain_prior = np.zeros(ppm['REST'].shape)

crude_brain = 2

## Morphology-based crude brain
if crude_brain == 1: 
    tmp = closing(ppm['REST'], size=6)
    threshold = .5
    brain_prior[mask] = np.minimum(1.-tmp[mask]/threshold, 1.)
    del tmp 
## Template-base crude brain
elif crude_brain in [2,3]: 
    im = load_image(os.path.join(datadir, 'MaskInitAnnot.img'))
    brain_prior = im.get_data()
    if crude_brain == 3: 
        brain_prior = gaussian_filter(brain_prior+.0, size=1)
        brain_prior /= brain_prior.max()


## Clean using crude brain mask 
ppm = geometry_cleaner(ppm, brain_prior, mask)
ppm, mask = normalize(ppm)

# Smooth using log opinion pool 
print('Performing log opinion pool smoothing...')
for tissue in tissues: 
    ppm[tissue] = log_opinion_pool(ppm[tissue], mask, size=1)
ppm, mask = normalize(ppm)

# Brain ppms 
raw_brain = brain_ppm(raw_ppm, raw_mask)
brain = brain_ppm(ppm, mask) 

# Save images
print('Saving maps...')
save_image(Image(raw_brain, affine=affine), os.path.join(datadir, 'brain_raw_ppm.nii'))
save_image(Image(brain, affine=affine), os.path.join(datadir, 'brain_'+str(crude_brain)+'_ppm.nii'))

# Displays 
"""
for slice in [60, 80, 100, 120]:
    display(raw_brain, slice=slice)
    display(brain, slice=slice)
"""    

"""
display_all(raw_ppm)
display_all(ppm)
"""
