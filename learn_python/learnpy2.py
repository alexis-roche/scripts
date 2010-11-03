class zozo:

	def __init__(self):
		self.chose = 0

	def zarma(self, truc, fn):
		self.bidule = fn(self, truc)

def dosomething(x):

	def doanother(x, word):
		x.chose = word
		
	x.zarma('chiottes', doanother)


