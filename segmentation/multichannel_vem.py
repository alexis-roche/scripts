from os.path import join
import gc
import numpy as np
import nibabel as nb
from nipy.algorithms.segmentation._segmentation import _ve_step

PATH = '/home/alexis/E/My Dropbox/AlexisTobiShare/image_data'
SUBJECT = '2011_10_26'
SCAN = 1


class TissueClassifier(object):

    def __init__(self, data, mu, sigma, beta=0):

        ntissues = len(mu)
        data = data.squeeze()
        if hasattr(mu[0], '__iter__'):
            nchannels = len(mu[0])
            space_shape = data.shape[0:-1]
        else:
            nchannels = 1
            space_shape = data.shape

        self.slices = [slice(1, s - 1) for s in space_shape]
        XYZ = np.mgrid[self.slices]
        XYZ = np.reshape(XYZ, (XYZ.shape[0], np.prod(XYZ.shape[1::]))).T
        self.XYZ = np.asarray(XYZ, dtype='int', order='C')
        data_msk = data[self.slices]

        if nchannels == 1:
            self.data = data_msk.reshape((np.prod(data_msk.shape), 1))
        else:
            self.data = data_msk.reshape((np.prod(data_msk.shape[0:-1]),
                                          data_msk.shape[-1]))
        self.ppm = np.zeros(list(space_shape) + [ntissues])
        self.ext_field = np.zeros([self.data.shape[0], ntissues])

        self.mu = np.asarray(mu).reshape((ntissues, nchannels))
        self.sigma = np.asarray(sigma).reshape((ntissues,
                                                nchannels, nchannels))

        self.beta = float(beta)

        # Should check whether input data is consistent with parameter
        # sizes

    def ntissues(self):
        return len(self.mu)

    def nchannels(self):
        return self.data.shape[-1]

    def vm_step(self):

        print(' VM step...')

        for i in range(self.ntissues()):
            #P = self.ppm[..., i].reshape(np.prod(self.ppm.shape[0:-1]))
            P = self.ppm[..., i][self.slices].ravel()
            Z = P.sum()
            tmp = self.data.T * P.T
            mu = tmp.sum(1) / Z
            mu_ = mu.reshape((len(mu), 1))
            sigma = np.dot(tmp, self.data) / Z - np.dot(mu_, mu_.T)
            self.mu[i] = mu
            self.sigma[i] = sigma

        gc.enable()
        gc.collect()

    def ve_step(self):

        print(' VE step...')

        # Compute posterior external field (no voxel interactions)
        for i in range(self.ntissues()):
            print('  tissue %d' % i)
            centered_data = self.data - self.mu[i]
            inv_sigma = np.linalg.inv(self.sigma[i])
            det_sigma = np.maximum(1e-20, np.linalg.det(self.sigma[i]))
            maha = np.sum(centered_data * np.dot(inv_sigma,
                                                 centered_data.T).T, 1)
            self.ext_field[:, i] = np.exp(-.5 * maha)
            self.ext_field[:, i] *= (1. / np.sqrt(det_sigma))

        self.ext_field.clip(1e-300, 1e300, out=self.ext_field)
        if self.beta == 0:
            print('  ... Normalizing...')
            tmp = self.ext_field.T
            tmp /= tmp.sum(0)
            self.ppm[:] = self.ext_field.reshape(self.ppm.shape)
        else:
            print('  ... MRF regularization')
            self.ppm = _ve_step(self.ppm, self.ext_field, self.XYZ,
                                self.beta, False, 0)

        gc.enable()
        gc.collect()

    def run(self, niters=10):

        for i in range(niters):
            print(' Iter %d/%d...' % (i + 1, niters))
            self.ve_step()
            self.vm_step()


def get_file_name(ids):
    if ids == 'mp-rage':
        f = SUBJECT + '_normal_MP-RAGE_p2'
    else:
        f = SUBJECT + '_mp2rage_2pt-dixon'
        f += '_scan' + str(SCAN) + '_'
        if ids == 'fat':
            f += 'combFAT'
        elif ids == 'flat':
            f += 'combUNI'
        elif ids == 'inv1':
            f += 'combINV1'
        elif ids == 'inv2':
            f += 'combINV2'
        elif ids == 'inv_rms':
            f += 'rmsINVs'
        else:
            raise ValueError('image type not understood')
    f += '.nii'
    return join(PATH, f)


def config_mprage():
    # MPRAGE based segmentation (single channel)
    im = nb.load(get_file_name('mp-rage'))
    aff = im.get_affine()
    data = im.get_data()
    mu = [5., 750., 90., 180., 380.]
    sigma = [10., 10., 10., 10., 10.]
    return data, aff, mu, sigma


def config_mp2rage_dixon():
    # MP2RAGE+DIXON based segmentation (single channel)
    im1 = nb.load(get_file_name('flat'))
    im2 = nb.load(get_file_name('fat'))
    aff = im1.get_affine()
    dat1 = im1.get_data().squeeze()
    data = np.zeros(list(dat1.shape) + [2], dtype=dat1.dtype)
    data[..., 0] = dat1
    data[..., 1] = im2.get_data().squeeze()
    mu = [[2000., 0.], [2000., 500.], [150., 0.], [1200., 0.], [2800., 0.]]
    sigma = [100 * np.eye(2) for i in range(len(mu))]
    return data, aff, mu, sigma

# AIR, FAT, CSF, GM, WM
# mu = [[5., 5.], [750., 750.], [90., 90.], [180., 180.], [380., 380.]]
# sigma = [100 * np.eye(2) for i in range(len(mu))]


data, aff, mu, sigma = config_mprage()
#data, aff, mu, sigma = config_mp2rage_dixon()

TC = TissueClassifier(data, mu, sigma, beta=0.2)
TC.run()

data = TC.ppm.argmax(-1)
nb.save(nb.Nifti1Image(data, aff), 'classif.nii')
