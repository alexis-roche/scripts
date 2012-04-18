import numpy as np
import nibabel as nb
import sys
from os.path import join, split, splitext

WIP = 'WIP542E_v9'


def get_wip_file(label, path, fname):
    return join(path, label + '_' + WIP + '_' + fname)


def compute_label_map(f):
    path, fname = split(f)
    im = nb.load(get_wip_file('csf', path, fname))
    ppm = np.zeros(list(im.get_shape()) + [3])
    ppm[..., 0] = im.get_data()
    for (i, label) in zip(range(1, 3), ('gm', 'wm')):
        im = nb.load(get_wip_file(label, path, fname))
        ppm[..., i] = im.get_data()
    mask = ppm.sum(-1) > 0
    label = np.zeros(ppm.shape[0:-1], dtype='uint8')
    label[mask] = ppm[mask].argmax(-1) + 1
    im_label = nb.Nifti1Image(label, im.get_affine())
    nb.save(im_label, join(path,
                           'relabel_' + splitext(fname)[0] + '.nii.gz'))


compute_label_map(sys.argv[1])
print sys.argv[1]
