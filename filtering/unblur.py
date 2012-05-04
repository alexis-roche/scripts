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


def unblur(y, msk, sigma):
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

    xf = op.fmin_cg(error, x[dom], fprime=residual, maxiter=20,
                    callback=callback)
    x[dom] = xf
    return x


def unblur_image(img, sigma):
    y = im.get_data().squeeze()
    msk = np.isnan(y)
    y[msk] = 0
    x = unblur(y, msk, sigma)
    return nb.Nifti1Image(x, im.get_affine())


def unblur_var_image(img, sigma):
    x = np.zeros((51, 51, 51))
    x[25, 25, 25] = 1
    y = nd.gaussian_filter(x, sigma)
    K = np.sum(y ** 2) / np.sum(y)
    y = im.get_data().squeeze()
    msk = np.isnan(y)
    y[msk] = 0
    x = unblur(y, msk, sigma / np.sqrt(2)) / K
    return nb.Nifti1Image(x, im.get_affine())


all_files = (glob('/home/alexis/D/Alexis/fiac_group/group*/*.img'),
             glob('/home/alexis/D/Alexis/fiac_group/variance*/*.img'))
unblur_fns = (unblur_image, unblur_var_image)
sigma = fwhm_to_sigma(5, 3)

for files, unblur_fn in zip(all_files, unblur_fns):
    print len(files)
    for f in files:
        print f
        path, fname = split(f)
        fname, _ = splitext(fname)
        im = nb.load(f)
        nb.save(im, join(path, fname + '.nii'))
        uim = unblur_fn(im, sigma)
        nb.save(uim, join(path, 'unblur_' + fname + '.nii'))
        print('Done.')

