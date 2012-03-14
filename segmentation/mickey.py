from glob import glob
from os.path import join, split, splitext
from time import time
import gc
import numpy as np
from scipy import sparse
import nibabel as nb

from pyamg import smoothed_aggregation_solver

from nipy.algorithms.segmentation import (Segmentation,
                                          initialize_parameters,
                                          binarize_ppm)
from nipy.algorithms.segmentation._segmentation import _make_edges


START = 0
PRE_NITERS = 5
NITERS = (5,5,5,5,5,5,5,5,5,5)
NGB_SIZE = 6
BETA = .5
SCALE_FACTOR = 1
TINY = 1e-50

LABELS = ('CSF', 'GMc', 'GMd', 'WM')
MU = np.array([112.287586, 230.84312703, 305.91563692, 379.11225839])
SIGMA = np.array([2265.97060371, 1222.01054703, 822.11289473, 611.15921583])
GLOB_MU = 271.88542994875456
GLOB_SIGMA = 9150.47573161447

RESPATH = '/home/alexis/D/Alexis/junk'

BRAINWEB_PATH = '/home/alexis/D/Alexis/brainweb'
DEFAULT_FILE = join(BRAINWEB_PATH, 'brainweb_SS.nii')

CACHE = {}


def moment_matching(dat):
    glob_mu = float(np.mean(dat))
    glob_sigma = float(np.var(dat))
    a = np.sqrt(glob_sigma / GLOB_SIGMA)
    b = glob_mu - a * GLOB_MU
    mu = a * MU + b
    sigma = (a ** 2) * SIGMA
    return mu, sigma


def mask_to_idx(mask):
    """
    Convert mask into array of indices
    """
    idx = -np.ones(mask.shape, dtype='int')
    tmp = mask > 0
    idx[tmp] = np.arange(tmp.sum())
    return idx


def make_laplacian(mask):
    edges = _make_edges(mask_to_idx(mask), NGB_SIZE)
    n = edges.max() + 1
    neg_weights = -np.ones(edges.shape[0])
    L = sparse.coo_matrix((neg_weights, edges.T), shape=(n, n))
    diag = np.vstack((np.arange(n), np.arange(n)))
    degrees = -np.ravel(L.sum(axis=1))
    return sparse.coo_matrix((np.hstack((neg_weights, degrees)),
                              np.hstack((edges.T, diag))),
                             shape=(n, n))


def get_Z(pi):
    x = np.sort(pi, 1)
    pi_max = x[:, -1]
    pi_min = x[:, -2]
    num = np.log(pi_max) - np.log(pi_min)
    den = np.maximum(pi_max - pi_min, TINY)
    return np.maximum(num / den, 0)


def config_random_walker(field, beta):
    # normalize to a distribution
    pi = (field.T / field.sum(1)).T
    Z = get_Z(pi)
    return (pi.T * Z).T, 1. / beta


def random_walker(mask, prior, gamma):
    """
    Assume prior is given on the mask, of shape (NPTS, K)
    """
    gc.enable()

    print('Assembling graph Laplacian...')
    L = make_laplacian(mask)
    n = L.shape[0]
    L = L + sparse.coo_matrix((gamma * prior.sum(axis=1),
                               (range(n), range(n))))

    print('Creating sparse solver...')
    mls = smoothed_aggregation_solver(L.tocsr())
    del L
    gc.collect()

    print('Loop over classes...')
    X = []
    for k in range(prior.shape[-1]):
        print('  Doing class %d...' % k)
        X += [mls.solve(gamma * prior[:, k])]

    del mls
    gc.collect()
    return np.array(X).T


def run_rw(S):
    prior, gamma = config_random_walker(S.ext_field(), BETA)
    q = random_walker(S.mask, prior, gamma)
    S.ppm[S.mask] = q
    return S


def _run_vem(S, niters):

    e, f, v, vf = [], [], [], []

    for it in range(niters):

        print('VE-step...')
        S.ve_step()

        print('Compute energies...')
        f += [S.free_energy()]
        vf += [np.sum(S.ppm[S.mask], 0)]
        ppm = S.ppm.copy()
        ppm[S.mask] = binarize_ppm(ppm[S.mask])
        e += [S.free_energy(ppm=ppm)]
        v += [np.sum(ppm[S.mask], 0)]

    print('VM-step...')
    S.vm_step()

    return S, e, f, v, vf


def run_vem(data, mask, laplace_init=False):

    mu, sigma = moment_matching(data[mask])

    S = Segmentation(data, mask=mask, mu=mu, sigma=sigma, beta=0)
    S.run(niters=PRE_NITERS)
    S = Segmentation(data, mask=mask, mu=S.mu, sigma=S.sigma,
                     ngb_size=NGB_SIZE, beta=BETA)

    e, f, v, vf = [], [], [], []

    # Initialize ppm with RW optionally
    if laplace_init:
        S = run_rw(S)
        S.ppm[S.mask] = binarize_ppm(S.ppm[S.mask])

    for s in range(len(NITERS)):

        # Run VEM for a few iterations
        S, es, fs, vs, vfs = _run_vem(S, niters=NITERS[s])

        e += es
        f += fs
        v += vs
        vf += vfs

    return S, e, f, v, vf


def save_ppm(img, ppm, mask, fid, stamp=''):
    for k in range(len(LABELS)):
        im = nb.Nifti1Image(ppm[..., k], img.get_affine())
        nb.save(im, join(RESPATH, LABELS[k] + '_' + stamp + '_' + fid + '.nii'))
    label_map = np.zeros(mask.shape, dtype='uint8')
    label_map[mask] = np.argmax(ppm[mask], 1) + 1
    im = nb.Nifti1Image(label_map, img.get_affine())
    nb.save(im, join(RESPATH, 'CLASSIF_' + stamp + '_' + fid + '.nii'))


def jaccard(map1, map2):
    J = []
    for k in range(len(LABELS)):
        m1 = map1 == k
        m2 = map2 == k
        J += [float(np.sum(m1 * m2)) / float(np.sum(m1 + m2))]
    return J


def compare_algos(filepath=DEFAULT_FILE, write_images=False):

    path, fname = split(filepath)
    fid, _ = splitext(fname)

    # Input image
    img = nb.load(join(path, fid + '.nii'))

    # Input mask
    mask = img.get_data() > 0

    # Run segmentation algorithms
    laplace_init = (False, True)
    stamps = ('VEM', 'FLICK-VEM')
    E, F, V, Vf = [], [], [], []
    label_map = []

    for i in range(2):

        gc.enable()

        # Run segmentation
        S, e, f, v, vf = run_vem(img.get_data(), mask, laplace_init=laplace_init[i])

        gc.collect()

        if write_images:
            save_ppm(img, S.ppm, mask, fid, stamp=stamps[i])

        # Score algorithms
        E += [e]
        F += [f]
        V += [v]
        Vf += [vf]

        # Compute label map
        label_map += [np.zeros(mask.shape, dtype='uint8')]
        label_map[i][mask] = np.argmax(S.ppm[mask], 1) + 1

        del S
        gc.collect()

    # Jaccard indices
    J = jaccard(label_map[0], label_map[1])

    # Garbage collection
    del img
    del mask
    gc.collect()

    # Format results
    return np.array(E), np.array(F), np.array(V), np.array(Vf), np.array(J)


def process_database(files, stamp):

    if START == 0:
        E, F, V, Vf, J = [], [], [], [], []
    else:
        fnpz = np.load(join(RESPATH, stamp) + '.npz')
        E = list(fnpz['E'])
        F = list(fnpz['F'])
        V = list(fnpz['V'])
        Vf = list(fnpz['Vf'])
        J = list(fnpz['J'])

    for i in range(START, len(files)): # for fil in files:
        fil = files[i]
        print fil
        e, f, v, vf, j = compare_algos(fil)
        E += [e]
        F += [f]
        V += [v]
        Vf += [vf]
        J += [j]
        print('Scores for %s: %s' % (fil, np.array(e).sum(1)))
        np.savez(join(RESPATH, stamp), E=np.array(E), F=np.array(F),
                 V=np.array(V), Vf=np.array(Vf), J=np.array(J))


#fimg = '/home/alexis/D/Alexis/Maeder/tiv_WIP542E_v8_702614.nii'
#fimg = '/home/alexis/D/Alexis/Maeder/tiv_WIP542E_v8_2661868.nii'

"""fimg = '/home/alexis/D/Alexis/Maeder/tiv_WIP542E_v8_472162.nii'
e, f, v, vf, j = compare_algos(fimg, write_images=True)
"""

"""
files = glob('/home/alexis/D/Alexis/lausanne/MNS*.nii')
process_database(files, 'lausanne')
"""

"""
fimg = '/home/alexis/D/Alexis/lausanne/MNS_CON_35_GradDist_N3.DenseTIV_EM_Bene.nii'
e, f, v, vf = compare_algos(fimg, write_images=True)
"""

files = glob('/home/alexis/D/Alexis/Maeder/tiv_WIP542E*.nii')
process_database(files, 'maeder')
