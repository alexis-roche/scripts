from os.path import join
import gc
import numpy as np
import scipy.ndimage as nd
import nibabel as nb
from nipy.algorithms.segmentation._segmentation import _ve_step, _gen_ve_step

PATH = PATH = '/home/alexis/D/Alexis/dixon'

# subject-specific params
SUBJECTS = ('2011_10_26', '2011_10_27', '2011_11_02', '2011_11_08')
BOTTOM_CORNER = {'2011_10_26': (0, 80, 0),
                 '2011_10_27': (0, 74, 0),
                 '2011_10_31': (0, 80, 0),                 
                 '2011_11_02': (0, 87, 0),
                 '2011_11_08': (0, 79, 0)}

# subject-independent params
CHANNELS = ('flat', 'fat', 'inv2')
TISSUES = ('air', 'gm', 'wm', 'csf', 'fat', 'dura', 'muscle')
BRAIN_TISSUES = ('gm', 'wm', 'csf')
## ITK-SNAP colors: clear, red, green, blue, yellow, cyan, magenta
BETA = 0.025
CUTOFF = 10
NITERS = 10
BAD_LINKS = (('air', 'csf'), ('air', 'gm'), ('air', 'wm'),
             ('muscle', 'gm'), ('muscle', 'wm'),
             ('dura', 'gm'), ('dura', 'wm'),
             ('fat', 'gm'), ('fat', 'wm'), 
             ('csf', 'air'), 
             ('gm', 'air'), ('gm', 'muscle'), ('gm', 'dura'),
             ('wm', 'air'), ('wm', 'muscle'), ('wm', 'dura'))
MEANS_MP2RAGE = {'flat': {'air': 2000, 'fat': 3500, 'dura': 2500, 'muscle': 1800, 
                          'csf': 200, 'gm': 1500, 'wm': 2800},
                 'fat': {'air': 5, 'fat': 400, 'dura': 150, 'muscle': 40, 
                         'csf': 5, 'gm': 5, 'wm': 10}, 
                 'inv1': {'air': 10, 'fat': 300, 'dura': 100, 'muscle': 50, 
                          'csf': 150, 'gm': 200, 'wm': 70},
                 'inv2': {'air': 20, 'fat': 800, 'dura': 350, 'muscle': 500, 
                          'csf': 150, 'gm': 500, 'wm': 500}}

SIZE_OPENING = 5
SIZE_CLOSING = 0
THRESHOLD = 0.5


class TissueClassifier(object):

    def __init__(self, data, mu=None, sigma=None, 
                 ppm=None, prior=None, U = None, beta=0, 
                 bottom_corner=(0, 0, 0), top_corner=(0, 0, 0)):

        data = data.squeeze()
        if not len(data.shape) in (3, 4):
            raise ValueError('invalid input image')
        if len(data.shape) == 3:
            nchannels = 1
            space_shape = data.shape
        else:
            nchannels = data.shape[-1]
            space_shape = data.shape[0:-1]

        self.slices = [slice(max(bc, 1), s - max(tc, 1))\
                           for s, bc, tc in zip(space_shape, bottom_corner, top_corner)]
        XYZ = np.mgrid[self.slices]
        XYZ = np.reshape(XYZ, (XYZ.shape[0], np.prod(XYZ.shape[1::]))).T
        self.XYZ = np.asarray(XYZ, dtype='int', order='C')
        data_msk = data[self.slices]

        if nchannels == 1:
            self.data = data_msk.reshape((np.prod(data_msk.shape), 1))
        else:
            self.data = data_msk.reshape((np.prod(data_msk.shape[0:-1]),
                                          data_msk.shape[-1]))

        if ppm == None:
            nclasses = len(mu)
            self.ppm = np.zeros(list(space_shape) + [nclasses])
            self.is_ppm = False
            self.mu = np.asarray(mu).reshape((nclasses, nchannels))
            self.sigma = np.asarray(sigma).reshape((nclasses,
                                                    nchannels, nchannels))
        elif mu == None:
            nclasses = ppm.shape[-1]
            self.ppm = np.asarray(ppm)
            self.is_ppm = True
            self.mu = np.zeros((nclasses, nchannels))
            self.sigma = np.zeros((nclasses, nchannels, nchannels))
        else:
            raise ValueError('missing information')
        self.ext_field = np.zeros([self.data.shape[0], nclasses])

        if not prior == None:
            self.prior = np.asarray(prior)[self.slices].reshape([self.data.shape[0], nclasses])
        else:
            self.prior = None
        
        self.set_energy(U, beta)

        # Should check whether input data is consistent with parameter
        # sizes

    def set_energy(self, U, beta):
        if not U == None:
            self.U = np.asarray(U).copy()  # make sure it's C-contiguous
        else:
            self.U = None
        self.beta = float(beta)

    def nclasses(self):
        return self.ppm.shape[-1]

    def nchannels(self):
        return self.data.shape[-1]

    def vm_step(self, freeze=()):

        print(' VM step...')

        classes = range(self.nclasses())
        for i in freeze:
            classes.remove(i)

        for i in classes:
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
        for i in range(self.nclasses()):
            print('  tissue %d' % i)
            centered_data = self.data - self.mu[i]
            inv_sigma = np.linalg.inv(self.sigma[i])
            det_sigma = np.maximum(1e-20, np.linalg.det(self.sigma[i]))
            maha = np.sum(centered_data * np.dot(inv_sigma,
                                                 centered_data.T).T, 1)
            self.ext_field[:, i] = np.exp(-.5 * maha)
            self.ext_field[:, i] *= (1. / np.sqrt(det_sigma))

        if not self.prior == None:
            self.ext_field *= self.prior
        self.ext_field.clip(1e-300, 1e300, out=self.ext_field)

        if self.beta == 0:
            print('  ... Normalizing...')
            tmp = self.ext_field.T
            tmp /= tmp.sum(0)
            self.ppm[self.slices][:] = self.ext_field.reshape(self.ppm[self.slices].shape)
        else:
            print('  ... MRF regularization')
            if self.U == None:
                self.ppm = _ve_step(self.ppm, self.ext_field, self.XYZ,
                                    self.beta, False, 0)
            else:
                self.ppm = _gen_ve_step(self.ppm, self.ext_field, self.XYZ,
                                        self.U, self.beta)

        gc.enable()
        gc.collect()

    def run(self, niters=NITERS, freeze=()):

        if self.is_ppm:
            self.vm_step(freeze=freeze)
        for i in range(niters):
            print(' Iter %d/%d...' % (i + 1, niters))
            self.ve_step()
            self.vm_step(freeze=freeze)
        self.is_ppm = True

def brain_mask(pbrain, threshold=THRESHOLD, size_opening=SIZE_OPENING):
    """
    Compute a brain mask from a crude in-brain probability map.
    """
    # opening
    size = [SIZE_OPENING for i in range(3)] 
    opening = nd.morphology.grey_opening(pbrain, size=size)

    # fin main connected component
    bin_opening = (opening > THRESHOLD)
    label, nlabels = nd.label(bin_opening)
    imax, count_max = 0, 0
    for i in range(1, nlabels):
        count = np.sum(label == i)
        if count > count_max:
            imax, count_max = i, count
    main_cc = np.zeros(label.shape, dtype='int8')
    main_cc[np.where(label == imax)] = 1

    # closing
    if  SIZE_CLOSING == 0:
        msk = main_cc
    else:
        structure = np.ones([SIZE_CLOSING for i in range(3)])
        msk = nd.morphology.binary_closing(main_cc, structure=structure)


    # hole filling (in each plane)
    structure = np.zeros((3, 3, 3))
    structure[:, :, 1] = 1
    msk = nd.morphology.binary_fill_holes(msk, structure=structure)
    structure = np.zeros((3, 3, 3))
    structure[:, 1, :] = 1
    msk = nd.morphology.binary_fill_holes(msk, structure=structure)
    structure = np.zeros((3, 3, 3))
    structure[1, :, :] = 1
    msk = nd.morphology.binary_fill_holes(msk, structure=structure)

    return msk


def get_file_name(ids, subject, scan=None):
    if ids == 'mp-rage':
        f = subject + '_normal_MP-RAGE_p2'
    else:
        f = subject + '_mp2rage_2pt-dixon'
        f += '_scan' + str(scan) + '_'
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


def config_mprage(subject):
    # MPRAGE based segmentation (single channel)
    im = nb.load(get_file_name('mp-rage', subject))
    aff = im.get_affine()
    data = im.get_data()
    mu = [5., 750., 90., 180., 380.]
    sigma = [10., 10., 10., 10., 10.]
    return data, aff, mu, sigma


def config_mp2rage_dixon(channels, tissues, subject, scan):
    ims = []
    mus = []
    for c in channels:
        im = nb.load(get_file_name(c, subject, scan))
        ims += [im]
        mus += [[MEANS_MP2RAGE[c][t] for t in tissues]]
        print im.get_data_dtype()
        print im.get_affine()

    aff = ims[0].get_affine()
    data0 = ims[0].get_data().squeeze()
    data = np.zeros(list(data0.shape) + [len(channels)], dtype=data0.dtype)
    data[..., 0] = data0
    for i in range(1, len(channels)):
        data[..., i] = ims[i].get_data().squeeze()
    mu = [[m[i] for m in mus] for i in range(len(tissues))]
    sigma = [100 * np.eye(len(channels)) for i in range(len(tissues))]
    return data, aff, mu, sigma


def config_energy(tissues, interaction=1, cutoff=CUTOFF, bad_links=None):
    U = interaction * np.ones((len(tissues), len(tissues)))
    U[(range(len(tissues)), range(len(tissues)))] = 0
    if bad_links == None:
        bad_links = ()
    for link in bad_links:
        if link[0] in tissues and link[1] in tissues:
            U[tissues.index(link[0]), tissues.index(link[1])] = cutoff
    return U


def compute_map(ppm):
    return ppm.argmax(-1)


def compute_gm_vol(ppm, mask, aff):
    i = TISSUES.index('gm')
    count = ppm[..., i][mask].sum()
    jacobian = np.abs(np.linalg.det(aff))
    return count * jacobian


def segment_mp2rage(subject, scan):
   
    bottom_corner = BOTTOM_CORNER[subject]
    prefix = subject + '_scan' + str(scan) + '_'

    # tissue classification
    data, aff, mu, sigma = config_mp2rage_dixon(CHANNELS, TISSUES, subject, scan)

    # First pass: standard Potts model
    TC = TissueClassifier(data, mu=mu, sigma=sigma, beta=BETA, 
                          bottom_corner=bottom_corner)
    TC.run(niters=5)
    nb.save(nb.Nifti1Image(compute_map(TC.ppm), aff), prefix + 'map0.nii')

    # First pass: penalize muscle if sourrounded by wm or gm. This is
    # basically a hack to avoid massive mis-classification of gm as
    # muscle - a situation which cannot be handled by the mean field
    # algorithm. Future work: use graph cuts.
    U = config_energy(TISSUES, bad_links=(
            ('muscle', 'wm'), ('muscle', 'gm')))
    TC = TissueClassifier(data, mu=mu, sigma=sigma, U=U, beta=BETA,
                          bottom_corner=bottom_corner)
    TC.run(niters=4)

    """
    nb.save(nb.Nifti1Image(compute_map(TC.ppm), aff), prefix + 'map1.nii')
    # Third pass: penalize all impossible connections and smooth
    U = config_energy(TISSUES, bad_links=BAD_LINKS)
    TC.set_energy(U, BETA)
    TC.run(niters=5)
    """
    # Compute MAP
    label = compute_map(TC.ppm)
    nb.save(nb.Nifti1Image(label, aff), prefix + 'map.nii')

    # Morphomat
    print('Computing brain mask...')
    pbrain = TC.ppm[..., [TISSUES.index(t) for t in BRAIN_TISSUES]].sum(-1)
    mask = brain_mask(pbrain)
    label[np.where(mask == 0)] = 0
    nb.save(nb.Nifti1Image(pbrain, aff), prefix + 'pbrain.nii')
    nb.save(nb.Nifti1Image(label, aff), prefix + 'masked_map.nii')

    # Save ppm 
    print('Saving PPM...')
    nb.save(nb.Nifti1Image(TC.ppm, aff), prefix + 'ppm.nii')
    
    # Gray matter volume (mm3)
    gm_vol = compute_gm_vol(TC.ppm, mask, aff)
    print('Gray matter volume: %f' % gm_vol)
    return gm_vol


segment_mp2rage('2011_10_26', 1)

#SUBJECTS = ( '2011_11_08',)  # hack


"""
gm_vols = {}
for subject in SUBJECTS:
    for scan in (1, 2):
        key = subject + '_scan' + str(scan) 
        gm_vols[key] = segment_mp2rage(subject, scan)
        print gm_vols
"""
"""
def morpho(count, count_tiv, affine):
    jacobian = np.abs(np.linalg.det(affine))
    vol = count * jacobian
    vol_tiv = count_tiv * jacobian
    rvol = count / float(count_tiv)
    x = np.array((count, count_tiv, vol, vol_tiv, rvol, jacobian))
    print x
    return x

y = []
for subject in SUBJECTS:
    for scan in (1, 2):
        print('Subject: %s, Scan: %d' % (subject, scan))
        im = nb.load(subject + '_scan' + str(scan) + '_ppm.nii')
        ppm = im.get_data()
        pbrain = ppm[..., [TISSUES.index(t) for t in BRAIN_TISSUES]].sum(-1)
        mask = brain_mask(pbrain)
        i = TISSUES.index('gm')
        count = ppm[..., i][mask].sum()
        count_tiv = mask.sum()
        y += [morpho(count, count_tiv, im.get_affine())]
    print('Subject: %s, MR-RAGE' % subject)
    im = nb.load(join(PATH, 'test', 'gm_WIP542E_v4_' + subject + '_normal_MP-RAGE_p2.nii'))
    count = im.get_data().sum() / 1000.
    im = nb.load(join(PATH, 'test', 'tiv_WIP542E_v4_' + subject + '_normal_MP-RAGE_p2.nii'))
    count_tiv = (im.get_data() > 0).sum()
    y += [morpho(count, count_tiv, im.get_affine())]
x = np.array(y)
np.save('toto', x)
"""
