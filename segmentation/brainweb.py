import numpy as np 
from pylab import * 
from os.path import join 
from nipy.io.imageformats import load, save
from nipy.neurospin.image import Image, asNifti1Image
from nipy.neurospin.segmentation.vem import VemTissueClassification

BETA = 0.2
SIGMA = 500.  
PREFIX = join('D:\home', 'Alexis', 'data')
DATADIR =  join(PREFIX, 'brainweb')
SAVEDIR = join(PREFIX, 'brainweb', 'results2')
TISSUES = ['CSF','GM','WM']

meth_dd = {'name': 'dd', 'hard': False, 'copy': False, 'niters': (25,0), 'noise':'gauss'}
meth_mf = {'name': 'mf', 'hard': False, 'copy': True, 'niters': (25,0), 'noise':'gauss'}
meth_ox = {'name': 'ox', 'hard': True, 'copy': True, 'niters': (25,0), 'noise':'gauss'}
meth_ox2 = {'name': 'ox2', 'hard': True, 'copy': True, 'niters': (25,5), 'noise':'gauss'}

METHODS = (meth_dd, meth_mf, meth_ox, meth_ox2)

SAVE = True
NUMCASES = [0,1,2,3,4,5,6,7]

"""
SAVE = False
NUMCASES = [0]
"""

# Non-editable parameters 
def get_case(numcase): 
    if numcase == 0: 
        c = 'brainweb_SS'
    else: 
        c = 'brainweb'+str(numcase+1)+'_SS'
    return c 

def get_betas(niters, niters_em): 
    betas = np.zeros(niters+niters_em)
    betas[niters_em::] = BETA
    return betas 

# Display function 
def display(array, slice=80, threshold=None): 
    figure()
    if threshold==None: 
        imshow(array[:,slice,:])
    else:
        imshow(array[:,slice,:]>threshold)

def display_ppm(ppm): 
    pink() 
    for i in range(len(TISSUES)): 
        display(ppm[:,:,:,i])

# Load (bias corrected) mri scan 
def load_mri(case): 
    fname = join(DATADIR, case+'.nii')
    return Image(load(fname))


def dice(gt, ppm, mu, mask):
	"""
	Compare with the ground truth 
	Return dice indices corresponding to 
	"""
	dices = np.zeros(len(TISSUES))
	gt_vals = [3,1,2]
	aux = np.argsort(mu)
	for k in range(len(TISSUES)): 
		kk = aux[k]
		tissue_mask = np.where(gt.data==gt_vals[k])
		XY = np.sum(ppm[tissue_mask][:,kk])
		X = len(tissue_mask[0])
		Y = np.sum(ppm[mask][:,kk])
		dices[k] = 2*XY/(X+Y)
		print('  %s : %f' % (TISSUES[k], dices[k]))
	return dices


# VEM algorithm 
def do_vem(im, mask, betas, hard, copy, noise, gt): 
    # Prior  
    ppm = np.zeros(list(im.shape)+[3])
    ppm[mask] = 1/3. 
    # Initial parameter estimates (crude histogram matching)
    mu = [700, 1700, 2300]
    sigma = [SIGMA, SIGMA, SIGMA]
    # VEM instance
    vem = VemTissueClassification(ppm, im.data, mask,
                                  noise=noise,
                                  hard=hard, copy=copy)
    # Initialize output arrays: free energy and Dice coeffs
    f = np.zeros(betas.size)
    d = np.zeros([betas.size, 3])
    niters = betas.size
    for i in range(niters):
    	print ('Iter n. %d/%d' % (i+1, niters))
	print(' VE-step')
	vem.ve_step(mu, sigma, beta=betas[i])
	print(' VM-step')
	mu, sigma = vem.vm_step()
	print(' Calculating free energy')
	f[i] = vem.free_energy()
	print(' Calculating Dice indices')
	d[i,:] = dice(gt, ppm, mu, mask)
	"""
	tmp = p0*log(p0/np.maximum(p, 1e-20))
	d[i] = np.sum(tmp)
	"""
    return f, d

# Main 
gt = Image(load(join(DATADIR, 'ground_truth.nii')))

# Loop 
#for numcase in range(8): 
for numcase in NUMCASES:
    case = get_case(numcase)
    print('Loading MRI data for subject %s...' % case)
    im = load_mri(case)
    mask = np.where(im.data>0)

    # EM
    for meth in METHODS: 
        print('Method: %s' % meth['name'])
        niters, niters_em = meth['niters']
        f, d = do_vem(im, mask, get_betas(niters, niters_em), 
                      meth['hard'], meth['copy'], meth['noise'], gt)

        # Save
        if SAVE: 
            fname = case + '_' + meth['name']
            np.savez(join(SAVEDIR, fname), f=f, d=d)


# Init: good or bad
# hard: True or False
# copy: True or False 

# Save 
"""
c = ['', 'copy'][copy]
h = ['', 'hard'][hard]
for i in range(len(TISSUES)): 
    pim = Image(ppm[:,:,:,i], im.affine)
    fname = case + '_' + TISSUES[i] + '_' + c + '_' + h + '.nii'
    save(asNifti1Image(pim), join(SAVEDIR, fname))
"""

