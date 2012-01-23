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
BETA = 1.0
LABELS = ('CSF', 'GM', 'WM')
USE_VEM = True

PATH = '/home/alexis/E/Data/brainweb'
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


def random_walker(mask, prior):
    """
    Assume prior is given on the mask, of shape (NPTS, K)
    """
    gc.enable()

    print('Assembling graph Laplacian...')
    L = make_laplacian(mask)
    n = L.shape[0]
    print L.shape
    gamma = 1.
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
    return np.array(X.T)


def run_rw(S):
    return


def run_vem(S):
    S.run(niters=NITERS)


def save_map_label(S):
    """
    Save the MAP classification
    """
    x = S.maximum_a_posteriori()
    map_img = nb.Nifti1Image(x, img.get_affine())
    nb.save(map_img, join(PATH, 'MAP_' + IM_ID + '.nii'))


# Input image
img = nb.load(join(PATH, IM_ID + '.nii'))

# Input mask
mask = img.get_data() > 0

# Initialize intensity parameters by moment matching
mu, std, _ = initialize_parameters(img.get_data()[mask], 3)

# Segmentation algorithm
S = Segmentation(img.get_data(), mask=mask, mu=mu, sigma=std ** 2,
                 ngb_size=NGB_SIZE, beta=BETA)


#run_vem(S)
#save_map_label(S)

# fake prior
prior = np.random.random((S.data.shape[0], 3))
X = random_walker(mask, prior)





