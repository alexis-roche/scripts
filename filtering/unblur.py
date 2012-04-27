import numpy as np
import scipy.ndimage as nd
import scipy.optimize as op
import nibabel as nb
import pylab as pl


def fwhm_to_sigma(fwhm, voxsize):
    """
    Convert fwhm in mm to sigma in voxel units. Assumes isotropic
    voxels.
    """
    k = 2 * np.sqrt(2 * np.log(2))
    sigma_mm = fwhm / k
    return sigma_mm / voxsize


def get_clean_data(im):
    x = im.get_data().squeeze()
    msk = np.isnan(x)
    x[msk] = np.mean(x[True - msk])
    return x


def unblur(y, sigma):

    cache = {'xf': None, 'res': None}
    shape = y.shape

    def residual(xf):
        if not xf is cache['xf']:
            x = xf.reshape(shape)
            res = nd.gaussian_filter(x, sigma) - y
            cache['res'] = res.ravel()
            cache['xf'] = xf
        return cache['res']

    def error(xf):
        return np.sum(residual(xf) ** 2)

    def grad_error(xf):
        return 2 * residual(xf)

    x = op.fmin_cg(error, y.reshape(shape), fprime=grad_error)
    return x.reshape(shape)


im = nb.load('/home/alexis/E/Data/fiac_group/group_DSp_minus_SSp_for_DSt/DSp_minus_SSp_for_DSt_fiac_00.img')

y = get_clean_data(im)
sigma = fwhm_to_sigma(5, 3)

x = unblur(y, sigma)

pl.figure()
pl.imshow(y[:,:,30])

pl.figure()
pl.imshow(x[:,:,30])
