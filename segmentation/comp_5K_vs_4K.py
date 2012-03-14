from os.path import join, split, splitext
from time import time
import numpy as np
import nibabel as nb

from nipy.algorithms.segmentation import (Segmentation,
                                          initialize_parameters,
                                          binarize_ppm)


NITERS = 25
NGB_SIZE = 6
BETA = .5

LABELS = ('CSF', 'GMc', 'GMd', 'WM')

GLOB_MU = 271.88542994875456
GLOB_SIGMA = 9150.47573161447

MU4 = np.array([112.287586, 230.84312703, 305.91563692, 379.11225839])
SIGMA4 = np.array([2265.97060371, 1222.01054703, 822.11289473, 611.15921583])
CONV_MAT4 = np.array([[1.,0,0],[0,1.,0],[0,1.,0],[0,0,1.]])

MU5 = np.array([112.287586, 180, 280, 305.91563692, 379.11225839])
SIGMA5 = np.array([2265.97060371, 1222.01054703, 1222.01054703, 822.11289473, 611.15921583])
CONV_MAT5 = np.array([[1.,0,0],[.5,.5,0],[0,1.,0],[0,.5,.5],[0,0,1.]])

RESPATH = '/home/alexis/E/junk'


def moment_matching(dat, ref_mu, ref_sigma):
    glob_mu = float(np.mean(dat))
    glob_sigma = float(np.var(dat))
    a = np.sqrt(glob_sigma / GLOB_SIGMA)
    b = glob_mu - a * GLOB_MU
    mu = a * ref_mu + b
    sigma = (a ** 2) * ref_sigma
    return mu, sigma


def save_map(img, ppm, mask, fid, stamp='', conv_mat=None):
    if not conv_mat == None:
        ppm = np.dot(ppm, conv_mat)
    label_map = np.zeros(mask.shape, dtype='uint8')
    label_map[mask] = np.argmax(ppm[mask], 1) + 1
    im = nb.Nifti1Image(label_map, img.get_affine())
    nb.save(im, join(RESPATH, 'CLASSIF_' + stamp + '_' + fid + '.nii'))


def compare_models(filepath):

    path, fname = split(filepath)
    fid, _ = splitext(fname)

    # Input image
    img = nb.load(join(path, fid + '.nii'))

    # Input mask
    mask = img.get_data() > 0

    # 4k-model
    mu, sigma = moment_matching(img.get_data()[mask], MU4, SIGMA4)
    S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=sigma,
                     ngb_size=NGB_SIZE, beta=BETA)
    S.run(niters=NITERS)
    save_map(img, S.ppm, mask, fid, stamp='4K')
    save_map(img, S.ppm, mask, fid, stamp='4K_conv', conv_mat=CONV_MAT4)

    # 5k-model
    mu, sigma = moment_matching(img.get_data()[mask], MU5, SIGMA5)
    S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=sigma,
                     ngb_size=NGB_SIZE, beta=BETA)
    S.run(niters=NITERS)
    print S.mu
    print S.sigma
    save_map(img, S.ppm, mask, fid, stamp='5K')
    save_map(img, S.ppm, mask, fid, stamp='5K_conv', conv_mat=CONV_MAT5)


fimg = '/home/alexis/E/Data/lausanne/MNS_CON_03_GradDist_N3.DenseTIV_EM_Bene.nii'
compare_models(fimg)
