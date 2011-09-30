from glob import glob
from os import system
from os.path import join, split, splitext
import random

import numpy as np

from nibabel import load, save, Nifti1Image
from nipy import load_image, save_image
from nipy.core.image.affine_image import AffineImage
from nipy.algorithms.segmentation import brain_segmentation
from nipy.algorithms.registration import HistogramRegistration, resample


DATADIR = '/home/alexis/E/Data/adni_hippocampus'

# First simulation was conducted on
TEMPLATE = 'S10009'
##TEMPLATE = 'S18818'
##TEMPLATE = 'S29264'
##TEMPLATE = 'S44581'
###SUBJECT = 'S11442' # 'S45803'
FAKE_VOLS = (-1, -1, -1)

USHORT_MAX = 2 ** 16 - 1
HIPPO_MASK_MAX = 5376

OUTPUT = 'uncertainty.npz'
START = 10
BETA = 0.2
FREEZE_PROP = True
VEM_ITERS = 25  # set to zero to use pre-computed classifications
SCHEME = 'mf'
NOISE = 'gauss'
LABELS = ('CSF', 'GC', 'GM', 'GW', 'WM')
MIXMAT = np.array([[1, 0, 0], [.5, .5, 0], [0, 1, 0], [0, .5, .5], [0, 0, 1]])
# LABELS = ('CSF', 'GM', 'WM')
# MIXMAT = None
NIPY_REGISTRATION = False
HIPPO_CORNER = None  # (45, 100, 110)
HIPPO_SIZE = (64, 64, 32)


def get_subjects(): 
    files = glob(join(DATADIR, '*MR_MPR*'))
    subjects = []
    for f in files: 
        s = split(f)[1]
        s = s[3+s.find('Br_S'):]
        s = s[:s.find('_')]
        subjects.append(s)
    if not len(files)==len(subjects):
        raise Exception('Some MPR image files were not understood')
    return np.array(subjects)


def get_image_files(subject):
    files = glob(join(DATADIR, '*'+subject+'*.nii'))
    mask = True
    if not len(files) == 1:
        print('OOOPS: %d MR images found' % len(files))
        return None
    f_data = files[0]
    # assuming subject ID is always of length 10...
    mask_dir = join(DATADIR, 'ADNI', (split(f_data)[1])[5:15], 'Hippocampal_Mask')
    subdirs = glob(join(mask_dir, '*/*'))
    tmp = [subject in s for s in subdirs]
    if not tmp.count(True) == 1:
        print('OOOPS: %d hippocampal directories found' % tmp.count(True))
        return None
    files = glob(join(subdirs[tmp.index(True)], '*.nii'))
    if not len(files) == 1:
        print('OOOPS: %d hippocampal images found' % len(files))
        return None
    #print f_data
    #print files[0]
    return f_data, files[0]


def reorient_mask(im, ref_im): 
    """
    reorient mask image to match the gray-scale image assuming that
    the underlying affine is correct.

    On a: 
    a, T -> ref
    m, Q -> msk 
    
    T : index to world
    aoTinv est l'image en coordonnees reelles

    On veut: 
    m2 tq m2oTinv = moQinv
    -->  m2 = moQinvoT 

    QinvoT est une mat de permutations...  

    m2[1,0,0] = m[0,0,-1]
    m2[0,1,0] = m[0,-1,0]
    m2[0,1,0] = m[1,0,0]

    m1[1,0,0] = m[0,0,1]
    """
    dat = im.get_data().astype('int16')
    dat = np.transpose(dat, [2,1,0])
    dat = dat[::-1,::-1,::-1]
    return Nifti1Image(dat, ref_im.get_affine())


def compute_masks(fix_subject, mov_subject): 
    """
    Perform non-rigid registration of the moving image towards the fix
    image, and deform the moving mask accordingly. Return both the fix
    mask and the deformed moving mask.
    """
    # Get image filenames for both the 'fixed' and 'moving' subjects 
    fix_im, fix_msk = get_image_files(fix_subject)
    mov_im, mov_msk = get_image_files(mov_subject)
		
    # Load data
    fix_im = load(fix_im)
    fix_msk = reorient_mask(load(fix_msk), fix_im)
    mov_im = load(mov_im)
    mov_msk =reorient_mask(load(mov_msk), mov_im)

    # Save data on hard drive (for the registration program) 
    save(mov_im, 'moving.nii')
    save(mov_msk, 'moving_mask.nii') 
    save(fix_im, 'fixed.nii')
    save(fix_msk, 'fixed_mask.nii') 

    # Deform the moving mask so as to match the fixed mask r_mov_msk will
    # have the same orientation as `fixed.nii` 
    system('./register moving.nii fixed.nii moving_mask.nii r_moving_mask.nii r_moving.nii')
	
    # At this point we may compute an overlap index between `reo_fixed_mask.nii`
    # and `r_moving_mask.nii`
    r_mov_msk = load('r_moving_mask.nii') 

    return fix_msk, r_mov_msk

def _compute_vols(subject_msk, template_msk): 
    sm = subject_msk.get_data()
    tm = template_msk.get_data()
    # Test whether arrays have the same shape
    if not sm.shape == tm.shape: 
        print('Shape mismatch problem (probably due to incorrect mask orientation)')
        print sm.shape
        print tm.shape
        vol_inter, vol_s, vol_t = FAKE_VOLS
    else:
        vol_inter = len(np.where((sm*tm)>0.0)[0])
        vol_s = len(np.where(sm>0.0)[0])	
        vol_t = len(np.where(tm>0.0)[0])
    # cleanup memory mapped numpy arrays to avoid problems overwriting image files 
    del sm, tm
    return vol_inter, vol_s, vol_t

def compute_vols(subject, template):
    print('..Performing registration')
    tmp = compute_masks(subject, template)
    if tmp == None: 
        print('***Registration failed')
        return FAKE_VOLS
    s_msk, t_msk = tmp
    print('..Computing vols')
    vols = _compute_vols(s_msk, t_msk)
    return vols 	
	
def cleanup_files():
    """ 
    Delete all temporary files to avoid nibabel refusing to overwrite them	
    """
    system('del moving.nii')
    system('del moving_mask.nii') 
    system('del fixed.nii')
    system('del fixed_mask.nii')
    system('del r_moving.nii')
    system('del r_moving_mask.nii')


def dice(vols):
    v = vols.T
    return 2.0*v[0]/(v[1]+v[2])

def jaccard(vols):
    """
    vol(AuB)=vol(A)+vol(B)-vol(AnB)
    """
    v = vols.T
    return v[0]/(v[1]+v[2]-v[0])


def check_data(subjects): 
    fsubjects = []
    for s in subjects: 
        f = get_image_files(s)
        if f == None:
            print('Subject %s is weird' % s)
        else:
            fsubjects.append(s) 
    print('%d good subjects out of %d' % (len(fsubjects), len(subjects)))
    return fsubjects


def atlas_segmentation(subjects):
    vols = np.zeros((len(subjects),3))
    for idx in range(len(subjects)):
        print('Processing subject: %s (%d/%d)' % (subjects[idx], idx+1, len(subjects)))
        try: 
            vols[idx] = compute_vols(subjects[idx], TEMPLATE)
            print('Dice coefficient: %f' % dice(vols[idx]))
            print('Overlap ratio: %f' % jaccard(vols[idx]))
        except:
            print('*** Unexpected failure')
            vols[idx] = FAKE_VOLS
        # Save intermediate results in a unique numpy data file 
        np.savez('results_'+TEMPLATE, subjects=subjects, vols=vols)


def register_hippocampus_masks(subjects, examples=None): 

    save_dir = 'atlas'
    mask_file = lambda s : join(save_dir, 'mask_'+s+'.nii')

    # Remove the template from the list...
    if TEMPLATE in subjects: 
        subjects.remove(TEMPLATE)

    # Scan the result directory and remove all subjects that have
    # already been processed
    mask_images = glob(join(save_dir, 'mask_*.nii')) 
    for m in mask_images: 
        s = splitext(split(m)[1])[0].strip('mask_')
        if s in subjects:
            subjects.remove(s) 

    # Possibly draw subjects without replacement 
    if not examples == None: 
        subjects = random.sample(subjects, examples) 

    # Display message 
    print('Subjects to process: %d' % len(subjects))

    # Load and write template data 
    fix_im, fix_msk = get_image_files(TEMPLATE)
    fix_im = load(fix_im)
    fix_msk =reorient_mask(load(fix_msk), fix_im)
    save(fix_im, 'fixed.nii')
    save(fix_msk, mask_file(TEMPLATE))

    # Loop over subjects 
    for s in subjects: 
        try:
            mov_im, mov_msk = get_image_files(s)
            mov_im = load(mov_im)
            mov_msk =reorient_mask(load(mov_msk), mov_im)
            save(mov_im, 'moving.nii')
            save(mov_msk, 'moving_mask.nii') 
            system('./register moving.nii fixed.nii moving_mask.nii '+mask_file(s)+' r_moving.nii')
        except:
            print('*** Unexpected failure')            


def make_chuv_atlas(subjects, examples=None): 

    save_dir = 'atlas_chuv'
    mask_file = lambda s : join(save_dir, 'mask_'+s+'.nii')

    # Scan the result directory and remove all subjects that have
    # already been processed
    mask_images = glob(join(save_dir, 'mask_*.nii')) 
    for m in mask_images: 
        s = splitext(split(m)[1])[0].strip('mask_')
        if s in subjects:
            subjects.remove(s) 

    # Possibly draw subjects without replacement 
    if not examples == None: 
        subjects = random.sample(subjects, examples) 

    # Display message 
    print('Subjects to process: %d' % len(subjects))

    # Loop over subjects 
    for s in subjects: 
        try:
            mov_im, mov_msk = get_image_files(s)
            mov_im = load(mov_im)
            mov_msk =reorient_mask(load(mov_msk), mov_im)
            save(mov_im, 'moving.nii')
            save(mov_msk, 'moving_mask.nii') 
            system('./register moving.nii chuv_template.nii moving_mask.nii '+mask_file(s)+' r_moving.nii')
        except:
            print('*** Unexpected failure')            


def make_hippocampus_prior(save_dir='atlas', save_name='hippocampus_prior'): 
    mask_images = glob(join(save_dir, 'mask_*.nii')) 
    
    im = load(mask_images[0])
    affine = im.get_affine()
    count = np.zeros(im.get_shape()) 
    
    for m in mask_images:
        print(m)
        im = load(mask_images[0])
        count += im.get_data()/float(HIPPO_MASK_MAX)

    count = (USHORT_MAX/float(len(mask_images)))*count
    count = count.astype('ushort')
    save(Nifti1Image(count, affine), save_name+'.nii')
    np.savez(save_name+'_images.npz',  mask_images=mask_images)


def moment_matching(im, mask):
    """
    Rough parameter initialization by moment matching with a brainweb
    image for which accurate parameters are known.
    """
    mu_ = np.array([813.9, 1628.4, 2155.8])
    sigma_ = np.array([215.6, 173.9, 130.9])
    m_ = 1643.1
    s_ = 502.8
    data = im.get_data()[mask]
    m = np.mean(data)
    s = np.std(data)
    a = s/s_
    b = m - a*m_
    return a*mu_ + b, a*sigma_ 


def get_tiv_image(subject): 
    tmp = glob(join(DATADIR, 'skull_stripped', '*'+subject+'*.nii'))
    if not(len(tmp))==1: 
        print('WARNING: subject %s has %d TIV images' % (s, len(tmp)))
        if len(tmp)==0: 
            raise ValueError('No TIV image found')
    return tmp[0]


def get_gray_ppm_image(subject): 
    tmp = glob(join(DATADIR, 'proba_GM', '*'+subject+'*.nii'))
    if not(len(tmp))==1: 
        print('WARNING: subject %s has %d TIV images' % (s, len(tmp)))
        if len(tmp)==0: 
            raise ValueError('No GM probability image found')
    return tmp[0]


def reorient_tiv(im, ref_im): 
    dat = im.get_data()
    dat = np.transpose(dat, [1,0,2])
    dat = dat[:,:,::-1]
    return Nifti1Image(dat, ref_im.get_affine())

def perform_tissue_classification(tiv, vem_iters, beta, scheme='mf',
                                  noise='gauss', freeze_prop=False, 
                                  labels=('CSF','GM','WM'), mixmat=None): 
    """
    perform probabilistic tissue classification on TIV 
    """
    tiv_ = AffineImage(tiv.get_data(), tiv.get_affine(), 'scanner') 
    ppm_img_, _ = brain_segmentation(tiv_, beta=beta, niters=vem_iters, 
                                     labels=labels, mixmat=mixmat, 
                                     noise=noise, freeze_prop=freeze_prop, 
                                     scheme=scheme)

    count_tiv = len(np.where(tiv.get_data() > 0)[0])
    return Nifti1Image(ppm_img_.get_data()[..., 1], tiv.get_affine()), \
        Nifti1Image(ppm_img_.get_data()[..., 0], tiv.get_affine()), \
        count_tiv


def compound_proba(hippo_prior, gray_ppm, normalize=False):
    p1 = hippo_prior.get_data() * gray_ppm.get_data()
    if normalize:
        p0 = (USHORT_MAX-hippo_prior.get_data())*(1-gray_ppm.get_data())
        p1 /= np.maximum(p0+p1, 1e-100)
    else:
        p1 /= USHORT_MAX
    return Nifti1Image(p1, hippo_prior.get_affine())


def estimate_hippocampus(subject, vem_iters=VEM_ITERS, beta=BETA, register=True): 
    f_im, f_msk = get_image_files(subject)
    f_tiv = get_tiv_image(subject) 
    im = load(f_im)
    msk = reorient_mask(load(f_msk), im) # just for posterior evaluation
    tiv = reorient_tiv(load(f_tiv), im) 
    print im.get_shape()
    print tiv.get_shape()
    save(im, 'fixed.nii')
    save(msk, 'fixed_mask.nii')
    save(tiv, 'fixed_tiv.nii')

    # register atlas and deform hippocampus ppm
    if register:
        if NIPY_REGISTRATION:
            I = load_image('template.nii')
            J = AffineImage(im.get_data(), im.get_affine(), 'scanner')
            R = HistogramRegistration(I, J, similarity='crl1', interp='pv')
            T = R.optimize('affine')
            if not HIPPO_CORNER == None:
                R.subsample(corner=HIPPO_CORNER, size=HIPPO_SIZE)
                T = R.optimize(T)
            save_image(resample(I, T.inv(), reference=J), 'r_template.nii')
            tmp = resample(load_image('hippocampus_prior.nii'), T.inv(),
                           reference=J, dtype='double')
            #save_image(tmp, 'r_hippocampus_prior.nii')
            tmp_data = np.minimum(np.maximum(tmp.get_data(), 0.0),
                                  USHORT_MAX).astype('uint16')
            save_image(AffineImage(tmp_data, tmp.affine, 'scanner'),
                       'r_hippocampus_prior.nii')
        else:
            system('./register template.nii fixed.nii hippocampus_prior.nii '
                   + 'r_hippocampus_prior.nii r_template.nii')
            if not HIPPO_CORNER == None:
                I = load_image('template.nii')
                Izoom = I[tuple([slice(c, c + s) for c, s
                                 in zip(HIPPO_CORNER, HIPPO_SIZE)])]

                print type(Izoom)

                save_image(Izoom, 'zoom_template.nii')
                system('./register zoom_template.nii fixed.nii '
                       + 'hippocampus_prior.nii '
                       + 'r_hippocampus_prior.nii r_template.nii')

    # perform tissue classification
    if vem_iters == 0:
        f_gray_ppm = get_gray_ppm_image(subject)
        gray_ppm = reorient_tiv(load(f_gray_ppm), im)
        save(gray_ppm, 'fixed_gray_ppm.nii')
        count_tiv = len(np.where(tiv.get_data() > 0)[0])
        count_tiv_gm = np.sum(gray_ppm.get_data())
    else:
        gray_ppm, csf_ppm, count_tiv = perform_tissue_classification(
            tiv, vem_iters, beta,
            scheme=SCHEME, noise=NOISE, labels=LABELS,
            mixmat=MIXMAT, freeze_prop=FREEZE_PROP)
        count_tiv_gm = np.sum(gray_ppm.get_data())
        count_tiv_csf = np.sum(csf_ppm.get_data())
        sum_squares_tiv_gm = np.sum(gray_ppm.get_data() ** 2)
        sum_squares_tiv_csf = np.sum(csf_ppm.get_data() ** 2)

    # compound hippocampus probabilities
    hippo_prior = load('r_hippocampus_prior.nii')
    hippo_ppm = compound_proba(hippo_prior, gray_ppm)
    save(hippo_ppm, 'r_hippocampus_ppm.nii')

    # estimate hippocampus volume
    jacobian = np.abs(np.linalg.det(hippo_ppm.get_affine()))
    count = np.sum(hippo_ppm.get_data())
    sum_squares = np.sum(hippo_ppm.get_data() ** 2)

    # compute Dice coefficient
    hippo_msk = np.where(msk.get_data() > 0)
    count_true = float(len(hippo_msk[0]))
    count_inter = np.sum(hippo_ppm.get_data()[hippo_msk])
    dice_coeff = 2 * count_inter / (count + count_true)
    count_true_pv = np.sum(gray_ppm.get_data()[hippo_msk])

    # CSF
    hippo_csf_ppm = compound_proba(hippo_prior, csf_ppm)
    save(hippo_csf_ppm, 'r_hippocampus_csf_ppm.nii')
    count_csf = np.sum(hippo_csf_ppm.get_data())
    sum_squares_csf = np.sum(hippo_csf_ppm.get_data() ** 2)

    # hack
    """
    dat = np.zeros(gray_ppm.get_shape())
    dat[hippo_msk] = gray_ppm.get_data()[hippo_msk]
    save(Nifti1Image(dat, gray_ppm.get_affine()), 'compound.nii')
    """
    def relative_std(count, sum_squares):
        return np.sqrt(np.maximum(count - sum_squares, 0.0))\
            / np.maximum(count, 1e-20)

    # output
    return {'tiv': count_tiv * jacobian,
            'gm': count_tiv_gm * jacobian,
            'csf': count_tiv_csf * jacobian,
            'pvol_true': count_true_pv * jacobian,
            'vol_true': count_true * jacobian,
            'pvol': count * jacobian,
            'pvol_csf': count_csf * jacobian,
            'gm_csf': count / max(float(count_csf), 1e-100),
            'dice': dice_coeff,
            'jacobian': jacobian,
            'gm_rel_std': relative_std(count_tiv_gm, sum_squares_tiv_gm),
            'csf_rel_std': relative_std(count_tiv_csf, sum_squares_tiv_csf),
            'pvol_rel_std': relative_std(count, sum_squares),
            'pvol_csf_rel_std': relative_std(count_csf, sum_squares_csf)}


"""
subjects = get_subjects()
subjects = check_data(subjects)
make_chuv_atlas(subjects)
"""
f = np.load('hippocampus_schuff.npz')


# Test
#dico = estimate_hippocampus('S16321')  # register=False


tmp = np.array(f['patho'])
idx = np.where((tmp==' Normal')+(tmp==' AD')) 
_subjects = f['subjects'][idx]
_patho = f['patho'][idx]
_age = f['age'][idx]
_sex = f['sex'][idx]

if START == 0:
    subjects = []
    info = []
    measures = []
else:
    f = np.load(OUTPUT)
    subjects = list(f['subjects'])
    info = list(f['info'])
    measures = list(f['measures'])

for i in range(START, len(_subjects)):
    s = _subjects[i]
    p = _patho[i]
    try:
        m = estimate_hippocampus(s)
        measures.append(m)
        subjects.append(s)
        this_info = {'patho': _patho[i], 'age': _age[i], 'sex': _sex[i]}
        info.append(this_info)
        print s, this_info, m
    except:
        print('Hippocampus estimation failed for: %s' % s)
    np.savez(OUTPUT,
             subjects=subjects,
             info=info,
             measures=measures)
