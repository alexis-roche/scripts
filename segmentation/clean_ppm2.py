import numpy as np
import scipy.ndimage as nd
import nipy.io.imageformats as brifti
import os 
import pylab 

datadir = 'D:\home\AR203069\data\delphine'
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
        fname = 'out'+tissue+'_100.img'
        im = brifti.load(os.path.join(datadir, fname))
        Pdict[tissue] = im.get_data()/1000.
    return Pdict

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

def clean_by_closing(ppm, base, tissues_to_clean, mask, size=def_size, threshold=.5): 
    base_closed = closing(base, size=size)[mask]
    prob_out = 1. - np.minimum(base_closed/threshold, 1.)
    for tissue in tissues_to_clean: 
        ppm[tissue][mask] *= prob_out
    return ppm

def brain_ppm(Pdict, mask): 
    P = np.zeros(Pdict['REST'].shape)
    P[mask] = 1.-Pdict['REST'][mask]
    return P


# Main program
print('Opening raw PPMs...')
raw_ppm = read_ppms()
raw_ppm, raw_mask = normalize(raw_ppm)

# Dummy instructions to force pylab in 'show' mode
tmp = np.random.rand(10)
pylab.plot(tmp)
pylab.show()
pylab.close()

# Display initial (dirty) maps
pylab.pink()

# Allocate the output ppms
ppm = {}
for tissue in tissues: 
    ppm[tissue] = raw_ppm[tissue].copy()
mask = raw_mask     

# Clean brain tissue maps 
print('Decrease brain tissue probabilities outside brain...')
naive = False
if naive: 
    ppm['REST'] = closing(ppm['REST'], size=6)
else: 
    ppm = clean_by_closing(ppm, ppm['REST'], ['WM','GM','CSF'], mask, size=6)
ppm, mask = normalize(ppm)

# Smooth using log opinion pool 
print('Performing log opinion pool smoothing...')
for tissue in tissues: 
    ppm[tissue] = log_opinion_pool(ppm[tissue], mask, size=1)
ppm, mask = normalize(ppm)

# Displays 
raw_brain = brain_ppm(raw_ppm, raw_mask)
brain = brain_ppm(ppm, mask) 

for slice in [60, 80, 100, 120]:
    display(raw_brain, slice=slice)
    display(brain, slice=slice)
    

"""
display_all(raw_ppm)
display_all(ppm)
"""
