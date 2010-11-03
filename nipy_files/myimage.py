import os 
import nipy.io.imageformats as brifti


def load_image(path): 
    base = brifti.load(path)
    return Image(base.get_data(), base.get_affine(), header=base._header)

class Image():

	def __init__(self, data, affine):
		self._data = data
		self._affine = affine 
		self._shape = data.shape
		self.mask = None

	def masked(self):
		return not self._mask == None
		
	def mask(self, coords=None, lowth=None, highth=None, ):
		"""
		Define a mask and optimize memory load accordingly. 
		"""		
		return 	
		
	def get_data(self): 
		"""
		Grid data.
		The difficulty here is that we might be storing the mask data only, 
		so that they need be reconstructed if that's the case. 		
		"""
		if not self.masked():
			return self._data
		# Reconstruct the array data

	def get(self, coords, interp='nearest', worldcoords=False):
		"""
		Returns the interpolated image values at the points specified by coords. 
		"""
		
	def set(self, data, background=0): 
		"""
		Set in-mask values using data and out-of-mask values to background. 
		"""
			



datadir = 'D:\home\Alexis\data\delphine\zozo'
#datadir = 'D:\Alexis\data\patient_03S0908'
path = os.path.join(datadir, 'BiasCorIm.img')
im = load_image(path) 
