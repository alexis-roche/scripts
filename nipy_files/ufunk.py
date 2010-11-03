import numpy as np

def f1(x):
	return x+1
u1 = np.frompyfunc(f1, 1, 1)

def f2(x, y):
	return x+y 
u2 = np.frompyfunc(f2, 2, 1)	

def f3(x, g=None):
	if g == None:
		g = 0
	return x+g
g = 1
u3 = np.frompyfunc(f3, 1, 1)	

	
# random arrays
x = np.random.rand(10,11)
y = np.random.rand(10,11)

# tests
#z = u1(x)
#z = u2(x, y)
z = u3(x)