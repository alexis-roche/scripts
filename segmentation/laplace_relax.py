import sys
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


NITERS = 50
NGB_SIZE = 6
BETA = 0.5
TINY = 1e-50

LABELS = ('CSF', 'GMc', 'GMd', 'WM')
MU = np.array([112.287586, 230.84312703, 305.91563692, 379.11225839])
SIGMA = np.array([2265.97060371, 1222.01054703, 822.11289473, 611.15921583])
GLOB_MU = 271.88542994875456
GLOB_SIGMA = 9150.47573161447

RESPATH = '/home/alexis/D/Alexis/junk'
BRAINWEB_PATH = '/home/alexis/D/Alexis/brainweb'
DEFAULT_FILE = join(BRAINWEB_PATH, 'brainweb_SS.nii')

DATABASE = 'lausanne'
BINARIZE = True
NSIMU = -1
CONTINUE = False

if len(sys.argv) > 1:
    DATABASE = sys.argv[1]
if len(sys.argv) > 2:
    BINARIZE = bool(int(sys.argv[2]))
if len(sys.argv) > 3:
    NSIMU = int(sys.argv[3])
if len(sys.argv) > 4:
    CONTINUE = bool(int(sys.argv[4]))

print DATABASE, BINARIZE, NSIMU, CONTINUE

def moment_matching(dat):
    glob_mu = float(np.mean(dat))
    glob_sigma = float(np.var(dat))
    a = np.sqrt(glob_sigma / GLOB_SIGMA)
    b = glob_mu - a * GLOB_MU
    mu = a * MU + b
    sigma = (a ** 2) * SIGMA
    return mu, sigma


def init_classical(img, mask, niters=0):
    """
    moment matching following by indep EM
    """
    mu, sigma = moment_matching(img.get_data()[mask])
    S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=sigma,
                     beta=0.0)
    if niters > 0:
        S.run(niters=niters)
    S.set_markov_prior(BETA)
    return S



"""
Implementation of Laplace relaxation. To be copied / pasted into
nipy.algorithms.segmentation.
"""

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
    del L
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

def stupid_get_Z(pi):
    """
    max_k [-log pi_k - a(1-|pi|^2)]/(1 + |pi|^2 - 2pi_k)
    """
    a = NGB_SIZE * BETA
    npi2 = np.sum(pi ** 2, 1)
    tmp = (-np.log(pi.T) - a * (1 - npi2.T)).T
    tmp /= (1 + npi2.T - 2 * pi.T).T
    return tmp.max(1)

def config_random_walker(field, beta):
    # normalize to a distribution
    pi = (field.T / field.sum(1)).T
    """
    Z = get_Z(pi)
    return (pi.T * Z).T, 1. / (2 * beta)
    """
    return pi, 1. / (2 * beta)

def random_walker(mask, prior, gamma):
    """
    Assume prior is given on the mask, of shape (NPTS, K).
    Return random walker probability map.
    """
    gc.enable()

    # Assembling graph Laplacian
    L = make_laplacian(mask)
    n = L.shape[0]
    L = L + sparse.coo_matrix((gamma * prior.sum(axis=1),
                               (range(n), range(n))))

    # Creating sparse solver
    mls = smoothed_aggregation_solver(L.tocsr())
    del L
    gc.collect()

    # Loop over classes
    X = []
    for k in range(prior.shape[-1]):
        X += [mls.solve(gamma * prior[:, k])]

    del mls
    gc.collect()
    return np.array(X).T

"""
End of RW implementation.
"""

def map_energy(S):
    """
    negated log-posterior
    """
    ppm = S.ppm.copy()
    ppm[S.mask] = binarize_ppm(ppm[S.mask])
    return S.free_energy(ppm=ppm)


def init_laplace(img, mask, mu, sigma):
    S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=sigma,
                 ngb_size=NGB_SIZE, beta=BETA)
    prior, gamma = config_random_walker(S.ext_field(), BETA)
    q = random_walker(mask, prior, gamma)
    if BINARIZE:
        S.ppm[mask] = binarize_ppm(q)
    else:
        S.ppm[mask] = q
    return S


def init_uniform(img, mask, mu, sigma):
    return Segmentation(img.get_data(), mask=mask, mu=mu, sigma=sigma,
                        ngb_size=NGB_SIZE, beta=BETA)

def run_naive(img, mask, mu, sigma, niters=1):
    S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=sigma, beta=0)
    S.ve_step()
    return S


def save_ppm(ppm, affine, mask, fid, stamp=''):
    for k in range(len(LABELS)):
        im = nb.Nifti1Image(ppm[..., k], affine)
        nb.save(im, join(RESPATH, LABELS[k] + '_' + stamp + '_' + fid + '.nii.gz'))
    label_map = np.zeros(mask.shape, dtype='uint8')
    label_map[mask] = np.argmax(ppm[mask], 1) + 1
    im = nb.Nifti1Image(label_map, affine)
    nb.save(im, join(RESPATH, 'label_' + stamp + '_' + fid + '.nii.gz'))


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


def jaccard(map1, map2):
    J = []
    for k in range(len(LABELS)):
        m1 = map1 == k + 1
        m2 = map2 == k + 1
        J += [float(np.sum(m1 * m2)) / float(np.sum(m1 + m2))]
    return J


def convert_4to3(ppm4):
    ppm = np.zeros(list(ppm4.shape[0:3]) + [3])
    ppm[..., 0] = ppm4[..., 0]
    ppm[..., 1] = ppm4[..., 1] + ppm4[..., 2]
    ppm[..., 2] = ppm4[..., 3]
    return ppm


def run_ve(S, niters=1):
    f = [S.free_energy()]
    e = [map_energy(S)]
    for it in range(niters):
        #print('..VE-step n. %d/%d' % (it + 1, niters))
        S.ve_step()
        S.vm_step()
        f += [S.free_energy()]
        e += [map_energy(S)]
    return np.array(e), np.array(f)


def compare_algos(fil):
    img = nb.load(fil)
    mask = img.get_data() > 0

    print('Parameter initialization')
    S = init_classical(img, mask)
    map0 = S.map()
    mu0 = S.mu.copy()
    sigma0 = S.sigma.copy()

    print('Running classical VEM')
    e0, f0 = run_ve(S, niters=NITERS)
    jac0 = jaccard(S.map(), map0)
    map0 = S.map()
    tmp = e0[-1][0] + e0[-1][1]
    print('Final energy: %f' % tmp)

    print('Running Laplace relaxed VEM')
    S = init_laplace(img, mask, mu0, sigma0)
    jac1 = jaccard(S.map(), map0)
    e, f = run_ve(S, niters=NITERS)
    jac2 = jaccard(S.map(), map0)
    tmp = e[-1][0] + e[-1][1]
    print('Final energy: %f' % tmp)

    del img
    del mask
    del S
    del map0
    gc.enable()
    gc.collect()

    return {'e0': e0, 'f0': f0, 'e': e, 'f': f,
            'jac0': jac0, 'jac1': jac1, 'jac2': jac2}


def test_case(fil):
    path, tmp = split(fil)
    fname = splitext(tmp)[0]
    img = nb.load(fil)
    mask = img.get_data() > 0
    S = init_classical(img, mask)
    save_ppm(S.ppm, img.get_affine(), mask, splitext(fname)[0], stamp='naive')
    run_ve(S, niters=NITERS)
    save_ppm(S.ppm, img.get_affine(), mask, splitext(fname)[0], stamp='free')
    S = init_laplace(img, mask, S.mu, S.sigma)
    save_ppm(S.ppm, img.get_affine(), mask, splitext(fname)[0], stamp='laplace')
    run_ve(S, niters=NITERS)
    save_ppm(S.ppm, img.get_affine(), mask, splitext(fname)[0], stamp='free2')



if DATABASE == 'lausanne':
    files = glob('/home/alexis/D/Alexis/lausanne/MNS*.nii')
elif DATABASE == 'maeder':
    files = glob('/home/alexis/D/Alexis/Maeder/tiv_WIP542E*.nii')
elif DATABASE == 'adni':
    files = glob('/home/alexis/D/Alexis/adni_hippocampus/tiv_WIP542E_v9*.nii')
else:
    print('No simulation')
if BINARIZE:
    prefix = 'miccai_'
else:
    prefix = 'miccai_smooth_'
simfile = prefix + DATABASE + '.npz'

if not CONTINUE:
    res = []
    proc_files = []
else:
    f = np.load(simfile)
    res = list(f['res'])
    proc_files = list(f['files'])

nfiles = len(files)
for f in proc_files:
    files.remove(f)

for f in files[0:NSIMU]:
    print('%d files processed so far (out of %d)' % (len(proc_files),
                                                     nfiles))
    try:
        gc.enable()
        gc.collect()
        res += [compare_algos(f)]
        proc_files += [f]
    except:
        print "Unexpected error:", sys.exc_info()[0]
    np.savez(simfile, res=res, files=proc_files)
    


#test_case('/home/alexis/D/Alexis/Maeder/tiv_WIP542E_v8_2646524_12ch_postGD.nii')
#test_case('/home/alexis/D/Alexis/lausanne/MNS_CON_03_GradDist_N3.DenseTIV_EM_Bene.nii')
#test_case('/home/alexis/D/Alexis/lausanne/MNS_CON_35_GradDist_N3.DenseTIV_EM_Bene.nii')
