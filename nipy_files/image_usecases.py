# Usecase 1
#################################
# take mean of 4D image
from glob import glob
import numpy as np
import nipy as ni

fname = 'some4d.nii'

img_list = ni.load_list(fname, axis=3)
vol0 = img_list[0]
arr = vol0.array[:]
for vol in img_list[1:]:
   arr += vol.array
mean_img = ni.Image(arr, vol0.coordmap)
ni.save(mean_img, 'mean_some4d.nii')

# Usecase 2
#################################
# take mean of some PCA volumes
fnames = glob('some3d*.nii')
vol0 = ni.load(fnames[0])
arr = vol0.array[:]
for fname in fnames[1:]:
    vol = ni.load(fname)
    arr += vol.array
mean_img = ni.Image(arr, vol0.coordmap)
ni.save(mean_img, 'mean_some3ds.nii')

# Usecase 3
#################################
# motion correction
img_list = ni.load_list(fname, axis=3)
reggie = ni.interfaces.fsl.Register(tol=0.1)
vol0 = img_list[0]
mocod = [] # unresliced
rmocod = [] # resliced
for vol in img_list[1:]:
    rcoord_map = reggie.run(moving=vol, fixed=vol0)
    cmap = ni.ref.compose(rcoord_map, vol.coordmap)
    mocovol = ni.Image(vol.array, cmap)
    # But...
    try:
       a_vol = ni.Image(vol.array, rcoord_map)
    except CoordmapError, msg
       assert msg == 'need coordmap with voxel input'
    mocod.append(mocovol)
    rmocovol = ni.reslice(mocovol, vol0)
    rmocod.append(rmocovol)
rmocod_img = ni.list_to_image(rmocovol)
ni.save(rmocod_img, 'rsome4d.nii')
try:
    mocod_img = ni.list_to_image(mocovol)
except ImageListError:
    print 'That is what I thought; the transforms were not the same'

# Usecase 4
#################################
# slice timing
img_list = ni.load_list(fname, axis=2)
slicetimer = ni.interfaces.fsl.SliceTime(algorithm='linear')
vol0 = img_list[0]
try:
   vol0.timestamp
except AttributeError:
   print 'we do not have a timestamp'
try:
   vol0.slicetimes
except AttributeError:
   print 'we do not have slicetimes'
try:
   st_list = slicetimer.run(img)
except SliceTimeError, msg:
   assert msg == 'no timestamp for volume'
TR = 2.0
slicetime = 0.15
sliceaxis = 2
nslices = vol0.array.shape[sliceaxis]
slicetimes = np.range(nslices) * slicetime
timestamps = range(len(img_list)) * TR
# Either the images are in a simple list
for i, img in enumerate(img_list):
   img.timestamp = timestamps[i]
   img.slicetimes = slicetimes
   img.axis['slice'] = sliceaxis # note setting of voxel axis meaning
# if the sliceaxes do not match, error when run
img_list[0].axis['slice'] = 1
try:
   st_list = slicetimer.run(img)
except SliceTimeError, msg:
   assert msg == 'images do not have the same sliceaxes']
# Or - with ImageList object
img_list.timestamps = timestamps
img_list.slicetimes = slicetimes
img_list.axis['slice'] = sliceaxis
# Either way, we run and save
st_list = slicetimer.run(img)
ni.save(ni.list_to_image(st_img), 'stsome4d.nii')

# Usecase 5
#################################
# making an image from an affine
data = img.array
affine = np.eye(4)
scanner_img = ni.Image(data, ni.ref.voxel2scanner(affine))
mni_img = ni.Image(data, ni.ref.voxel2mni(affine))

# Usecase 6
#################################
# coregistration and normalization
anat_img = ni.load_image('anatomical.nii')
func_img = ni.load_image('epi4d.nii')
template = ni.load_image('mni152T1.nii')

# coreg
coreger = ni.interfaces.fsl.flirt(tol=0.2)
coreg_cmap = coreger.run(fixed=func_img, moving=anat_img)
c_anat_img = ni.Image(anat_img.data, coreg_cmap.compose_with(anat_img.cmap))

# normalize
template_cmap = template.coordmap
template_dims = template.data.shape
c_anat_cmap = c_anat_img.coordmap
normalizer = ni.interfaces.fsl.fnirt(param=3)
norm_cmap = normalizer.run(moving=template, fixed=c_anat_img)
full_cmap = norm_cmap.composed_with(template_cmap)
w_anat_data = img.resliced_to_grid(full_cmap, template_dims)
w_anat_img = ni.Image(w_anat_data, template.coordmap)

w_func_list = []
for img in ni.image_list(func_img, axis=3):
  w_img_data = img.resliced_to_grid(full_cmap, template_dims)
  w_func_list.append(ni.Image(w_img_data, template_cmap))
ni.save(ni.list_to_image(w_func_list), 'stsome4d.nii')

''' Normalizing no warping
This is how it would work if all the non-affine transformations are
handled outside the image class, as coordinates and coordinate maps
'''

img = ni.load('some_anatomical.nii')
template = ni.load('some_template.nii')

reggie =  ni.interfaces.fsl.Fnirter()

w2w = reggie.run(source=template, destination=img)

tv2iw = w2w.compose_with(template.get_coordmap())

shape = template.get_shape()
indices = np.indices(shape).reshape(3, np.prod(shape))
img_w_pts = tv2iw.apply(indices)
# the mapping for ``img`` must be invertible for this to work
data = img.values_at_world(img_w_pts)

# Or whatever
resampled_image = Image(data, template.affine, template.world)
