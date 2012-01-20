from os.path import join
import numpy as np
from scipy import sparse
import nibabel as nb

from nipy.algorithms.segmentation import (Segmentation,
                                          initialize_parameters)
from nipy.algorithms.segmentation._segmentation import _make_edges

from random_walker import random_walker_prior


NITERS = 10
NGB_SIZE = 6
BETA = 1.0
LABELS = ('CSF', 'GM', 'WM')
USE_VEM = True

PATH = '/home/alexis/E/Data/brainweb'
IM_ID = 'brainweb_SS'


def make_laplacian(mask):
    edges = _make_edges(mask.astype('uint'), NGB_SIZE)
    neg_weights = -np.ones(edges.shape[0])
    L = sparse.coo_matrix((neg_weights, edges.T))
    diag = np.arange(n)
    connect = -np.ravel(L.sum(axis=1))
    lap = sparse.coo_matrix((np.hstack((neg_weights, connect)),
                             (np.hstack((i_indices,diag)), np.hstack((j_indices, diag)))), 
                            shape=(n, n))
    return L


def rw_step(S):
    """
    Input is a Segmentation object
    """
    y = np.zeros((40, 40, 40))
    y[10:-10, 10:-10, 10:-10] = 1
    y += 0.7 * np.random.random((40, 40, 40))
    p = y.max() - y.ravel()
    q = y.ravel()
    prior = np.array([p, q])

    print y.shape
    print prior.shape

    return random_walker_prior(y, prior, mode='amg')


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


#L = make_laplacian(mask)

edges = _make_edges(mask.astype('uint'), NGB_SIZE)
n = edges.max() + 1
neg_weights = -np.ones(edges.shape[0])
L = sparse.coo_matrix((neg_weights, edges.T), shape=(n, n))
diag = np.vstack((np.arange(n), np.arange(n)))
connect = -np.ravel(L.sum(axis=1))
lap = sparse.coo_matrix((np.hstack((neg_weights, connect)),
                         np.hstack((edges.T, diag))),
                        shape=(n, n))
