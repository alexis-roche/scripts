import numpy as np

from nipy.algorithms.segmentation._segmentation import _make_edges



im = np.random.rand(5, 4, 7)
msk = (im > .5).astype('uint')

edges = _make_edges(msk, 6)
