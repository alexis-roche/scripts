from os.path import join
import gc
import numpy as np
from scipy import sparse
import nibabel as nb

from pyamg import smoothed_aggregation_solver

from nipy.algorithms.segmentation import (Segmentation,
                                          initialize_parameters)
from nipy.algorithms.segmentation._segmentation import _make_edges


NITERS = 10
NGB_SIZE = 6
BETA = 0.5
LABELS = ('CSF', 'GM', 'WM')
USE_VEM = True

PATH = '/home/alexis/E/Data/brainweb'
RESPATH = '/home/alexis/E/Data/brainweb-results'
IM_ID = 'brainweb_SS'


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
    den = pi_max - pi_min
    return num / den


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


def run_rw(img, mask):
    mu, std, _ = initialize_parameters(img.get_data()[mask], 3)
    S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=std ** 2,
                 ngb_size=NGB_SIZE, beta=BETA)
    prior, gamma = config_random_walker(S.ext_field(), BETA)
    q = random_walker(mask, prior, gamma)
    S.ppm[mask] = q
    return S


def run_vem(img, mask):
    mu, std, _ = initialize_parameters(img.get_data()[mask], 3)
    S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=std ** 2,
                 ngb_size=NGB_SIZE, beta=BETA)
    for it in range(NITERS):
        S.ve_step()
    return S


def save_ppm(img, ppm, stamp=''):
    for k in range(len(LABELS)):
        im = nb.Nifti1Image(ppm[..., k], img.get_affine())
        nb.save(im, join(RESPATH, LABELS[k] + '_' + stamp + '_' + IM_ID + '.nii'))
    label_map = np.zeros(mask.shape, dtype='uint8')
    label_map[mask] = np.argmax(ppm[mask], 1) + 1
    im = nb.Nifti1Image(label_map, img.get_affine())
    nb.save(im, join(RESPATH, 'CLASSIF_' + stamp + '_' + IM_ID + '.nii'))


def _fuzzy_dice(gpm, ppm, mask):
    dices = np.zeros(3)
    for k in range(3):
        pk = gpm[k][mask]
        qk = ppm[mask][:, k]
        PQ = np.sum(np.sqrt(np.maximum(pk * qk, 0)))
        P = np.sum(pk)
        Q = np.sum(qk)
        dices[k] = 2 * PQ / float(P + Q)
    return dices


def _dice(gpm, ppm, mask):
    dices = np.zeros(3)
    gpm = np.rollaxis(np.array(gpm), 0, 4)[mask]
    ppm = ppm[mask]
    sg = gpm.argmax(1)
    s = ppm.argmax(1)
    for k in range(3):
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
    gpm = [nb.load(join(PATH, f)).get_data() for f in \
               ('phantom_1.0mm_normal_csf_.nii',
                'phantom_1.0mm_normal_gry_.nii',
                'phantom_1.0mm_normal_wht_.nii')]
    return _dice(gpm, ppm, mask), _fuzzy_dice(gpm, ppm, mask)


# Input image
img = nb.load(join(PATH, IM_ID + '.nii'))

# Input mask
mask = img.get_data() > 0

# Segmentation algorithms
S = run_vem(img, mask)
save_ppm(img, S.ppm)
print S.free_energy()
print S.map_energy()

S2 = run_rw(img, mask)
save_ppm(img, S2.ppm, 'RW')
print S2.free_energy()
print S2.map_energy()


"""
mu, std, _ = initialize_parameters(img.get_data()[mask], 3)

S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=std ** 2,
                 ngb_size=NGB_SIZE, beta=BETA)
prior, gamma = config_random_walker(S.ext_field(), BETA)

print('Assembling graph Laplacian...')
L = make_laplacian(mask)
n = L.shape[0]
print L.shape
D = sparse.coo_matrix((gamma * prior.sum(axis=1),
                       (range(n), range(n))))

#L = L + D
"""
