import numpy as np
import scipy.ndimage as nd
import scipy.optimize as op
import nibabel as nb
from glob import glob
from os.path import split, splitext, join


def fwhm_to_sigma(fwhm, voxsize):
    """
    Convert fwhm in mm to sigma in voxel units. Assumes isotropic
    voxels.
    """
    k = 2 * np.sqrt(2 * np.log(2))
    sigma_mm = fwhm / k
    return sigma_mm / voxsize


def blur(x, msk, sigma):
    """
    x should be zero in the mask
    """
    gx = nd.gaussian_filter(x, sigma)
    norma = 1 - nd.gaussian_filter(msk.astype('float'), sigma)
    gx[True - msk] /= norma[True - msk]
    gx[msk] = 0.
    return gx


def unblur(y, msk, sigma, maxiter=20):
    """
    y should be zero in the mask
    """
    cache = {'xf': None, 'res': None}
    nvoxels = np.sum(1 - msk)
    print nvoxels
    x = np.array(y)
    dom = True - msk

    def residual(xf):
        if not xf is cache['xf']:
            x[dom] = xf
            cache['res'] = blur(x, msk, sigma)[dom] - y[dom]
            cache['xf'] = xf
        return cache['res']

    def callback(xf):
        print error(xf)

    def error(xf):
        return .5 * np.sum(residual(xf) ** 2)

    xf = op.fmin_cg(error, x[dom], fprime=residual, 
                    maxiter=maxiter, callback=callback)
    x[dom] = xf
    return x


def unblur_image(img, sigma):
    y = img.get_data().squeeze()
    msk = np.isnan(y)
    y[msk] = 0
    x = unblur(y, msk, sigma)
    x[msk] = np.nan
    return nb.Nifti1Image(x, img.get_affine())


def unblur_var_image(img, sigma):
    x = np.zeros((51, 51, 51))
    x[25, 25, 25] = 1
    y = nd.gaussian_filter(x, sigma)
    K = np.sum(y ** 2) / np.sum(y)
    print K
    y = img.get_data().squeeze()
    msk = np.isnan(y)
    y[msk] = 0
    x = unblur(y, msk, sigma / np.sqrt(2), maxiter=50)
    x[True - msk] /= (K * 382)
    x[True - msk] = np.maximum(x[True - msk], 0)
    print('Minimum of var image: %f' % np.min(x[True - msk]))
    x[msk] = np.nan
    return nb.Nifti1Image(x, img.get_affine())


path = '/home/alexis/D/Alexis/fiac_group'
files = glob(join(path, 'group*/*.nii'))
sigma = fwhm_to_sigma(5, 3)

for f in files:
    con = (split(split(f)[0])[1]).lstrip('group_')
    num = (splitext(split(f)[1])[0]).lstrip(con + '_fiac_')
    print con, num
    print('Unblur contrast image...')
    im = unblur_image(nb.load(f), sigma)
    nb.save(im, join(path, con, 'fiac' + num + '.nii'))
    print('Unblur variance image...')
    fv = join(path, 'variance_' + con, 'var_fiac' + num + '.nii')
    im = unblur_var_image(nb.load(fv), sigma)
    nb.save(im, join(path, con, 'var_fiac' + num + '.nii'))
    print('Done.')
