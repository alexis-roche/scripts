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


NITERS = 100
NGB_SIZE = 6
BETA = .5
SCALE_FACTOR = 1
TINY = 1e-50
CACHE = {}

LABELS = ('CSF', 'GMc', 'GMd', 'WM')
MU = np.array([112.287586, 230.84312703, 305.91563692, 379.11225839])
SIGMA = np.array([2265.97060371, 1222.01054703, 822.11289473, 611.15921583])
GLOB_MU = 271.88542994875456
GLOB_SIGMA = 9150.47573161447

RESPATH = '/home/alexis/D/Alexis/junk'

BRAINWEB_PATH = '/home/alexis/D/Alexis/brainweb'
DEFAULT_FILE = join(BRAINWEB_PATH, 'brainweb_SS.nii')


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
    print L.shape
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


def run_rw(img, mask, mu, sigma):
    S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=sigma,
                 ngb_size=NGB_SIZE, beta=BETA)
    prior, gamma = config_random_walker(S.ext_field(), BETA)
    q = random_walker(mask, prior, gamma)
    S.ppm[mask] = q
    return S


def run_vem(img, mask, mu, sigma):
    S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=sigma,
                 ngb_size=NGB_SIZE, beta=BETA)
    for it in range(NITERS):
        S.ve_step()
    return S


def run_naive(img, mask, mu, sigma):
    S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=sigma,
                 ngb_size=NGB_SIZE, beta=BETA)
    field = S.ext_field()
    S.ppm[mask] = (field.T / field.sum(1)).T
    return S


def post_vem(S, update_params=True):
    S.ppm[S.mask] = binarize_ppm(S.ppm[S.mask])
    e, f = [], []

    for it in range(NITERS):
        t0 = time()
        if update_params:
            S.vm_step()
        S.ve_step()
        dt = time() - t0 
        print('VEM iteration time: %f' % dt)
        f += [S.free_energy()]
        ppm = S.ppm.copy()
        ppm[S.mask] = binarize_ppm(ppm[S.mask])
        e += [S.free_energy(ppm=ppm)]
    return S, e, f


def save_ppm(img, ppm, mask, fid, stamp=''):
    for k in range(len(LABELS)):
        im = nb.Nifti1Image(ppm[..., k], img.get_affine())
        nb.save(im, join(RESPATH, LABELS[k] + '_' + stamp + '_' + fid + '.nii'))
    label_map = np.zeros(mask.shape, dtype='uint8')
    label_map[mask] = np.argmax(ppm[mask], 1) + 1
    im = nb.Nifti1Image(label_map, img.get_affine())
    nb.save(im, join(RESPATH, 'CLASSIF_' + stamp + '_' + fid + '.nii'))


def _fuzzy_dice(gpm, ppm, mask):
    dices = np.zeros(len(LABELS))
    for k in range(len(LABELS)):
        pk = gpm[k][mask]
        qk = ppm[mask][:, k]
        PQ = np.sum(np.sqrt(np.maximum(pk * qk, 0)))
        P = np.sum(pk)
        Q = np.sum(qk)
        dices[k] = 2 * PQ / float(P + Q)
    return dices


def _dice(gpm, ppm, mask):
    dices = np.zeros(len(LABELS))
    gpm = np.rollaxis(np.array(gpm), 0, 4)[mask]
    ppm = ppm[mask]
    sg = gpm.argmax(1)
    s = ppm.argmax(1)
    for k in range(len(LABELS)):
        pk = (sg == k)
        qk = (s == k)
        PQ = np.sum(pk * qk)
        P = np.sum(pk)
        Q = np.sum(qk)
        dices[k] = 2 * PQ / float(P + Q)
    return dices


def dice(ppm, mask):
    """
    Dice and fuzzy dice indices.
    """
    gpm = [nb.load(join(BRAINWEB_PATH, f)).get_data() for f in \
               ('phantom_1.0mm_normal_csf_.nii',
                'phantom_1.0mm_normal_gry_.nii',
                'phantom_1.0mm_normal_wht_.nii')]
    return _dice(gpm, ppm, mask), _fuzzy_dice(gpm, ppm, mask)


def compare_algos(filepath=DEFAULT_FILE, write_images=False, update_params=True):

    path, fname = split(filepath)
    fid, _ = splitext(fname)

    # Input image
    img = nb.load(join(path, fid + '.nii'))

    # Input mask
    mask = img.get_data() > 0

    # Initialize tissue class parameters
    ###mu, std, _ = initialize_parameters(img.get_data()[mask], len(LABELS))
    ###sigma = (SCALE_FACTOR * std) ** 2
    mu, sigma = moment_matching(img.get_data()[mask])

    # Run segmentation algorithms
    # Use the 'naive' EM to initialize tissue mean params
    algos = (run_naive, run_rw)
    stamps = ('NAIVE', 'RW')
    E, F, D, Df = [], [], [], []

    for i in range(len(algos)):

        # Run segmentation
        t0 = time()
        S = algos[i](img, mask, mu, sigma)
        dt = time() - t0
        print(stamps[i] + ' time: %f' % dt)

        if write_images:
            save_ppm(img, S.ppm, mask, fid, stamp=stamps[i])

        S, e, f = post_vem(S, update_params)

        if write_images:
            save_ppm(img, S.ppm, mask, fid, stamp=stamps[i] + '_post')

        # Score algorithms
        E += [e]
        F += [f]

        # Compute Dice indices for brainweb data
        if 'brainweb' in path and len(LABELS) == 3:
            d, df = dice(S.ppm, mask)
            D += [d]
            Df += [df]
            print('Dice indices %s, %s' % (d, df))

    CACHE['S'] = S

    # Format results
    return np.array(E), np.array(F), np.array(D), np.array(Df)


def process_database(files, stamp, update_params=True):
    E, F, D, Df = [], [], [], []
    for f in files:
        print f
        gc.enable()
        e, f, d, df = compare_algos(f, update_params)
        gc.collect()
        E += [e]
        F += [f]
        D += [d]
        Df += [df]
        print('Scores for %s: %s' % (f, np.array(e).sum(1)))
        np.savez(join(RESPATH, stamp), E=np.array(E), F=np.array(F), 
                 D=np.array(D), Df=np.array(Df))
    print np.array(E).shape
    return np.array(E), np.array(F), np.array(D), np.array(Df)


#fimg = '/home/alexis/D/Alexis/lausanne/MNS_CON_03_GradDist_N3.DenseTIV_EM_Bene.nii'
#e, f, d, df = compare_algos(fimg, write_images=True, update_params=False)
#e, f, d, df = compare_algos(write_images=True)

files = glob('/home/alexis/D/Alexis/Maeder/tiv_WIP542E*.nii')
#files = glob('/home/alexis/D/Alexis/lausanne/MNS*.nii')
E, F, D, Df = process_database(files, 'lausanne_frozen_params', update_params=False)
#E, F, D, Df = process_database(files, 'maeder')
