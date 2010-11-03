class caca:

	def __init__(self):
		self.bouse = 1
		self.beurk = 1
		self.doux_fumet = 0

	def __getattr__(self, name):
		print "Attribute %s" % name

	def __getitem__(self, name):
		print "Item %s" % name

	def drop(self, n):
		print "Dropped %d kilos of caca" % n 
		
	def __call__(self):
		return "Scatological class to learn Python"


class etron(caca):

	def set_length(self, l=5):
		self.length = l

class Chose: 
	def __init__(self, x): 
		self.bouse = x

class Chose2(object): 
	def __init__(self, x): 
		self.bouse = x
	

	
