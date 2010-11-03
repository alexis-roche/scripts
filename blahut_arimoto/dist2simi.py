import numpy as np 
import scipy.optimize as optimize
import pylab 

"""
H: joint histogram, first dim is 'x', second dim is 'y'
F : number of features x dimX x dimY 

H and m MUST be normalized

x is the variable to be predicted based on the observation y.
"""

def ssd(x,j):
    return (x-j)**2

def m1x(x,j):
    return x

def m1y(x,y):
    return y

def m2xy(x,y):
    return x*y

def m2xx(x,y):
    return x**2

def m2yy(x,y):
    return y**2


_NITER = 10
_METHOD = 'powell'
_TOL = 1e-5

class maxCEnt: 

    def __init__(self, F, Fmean): 
        self.dimX = H.shape[0]
        self.dimY = H.shape[1]
        self.H = H/H.sum()
        self.nfeatures = len(F)
        aux = np.rollaxis(np.asarray(F), 0, 3) - np.asarray(Fmean)
        aux = aux / np.abs(aux).max() # normalize between -1 and 1
        self.F = np.rollaxis(aux, 2, 0)            
        self._lda = None
        self._mY = None

    """
    Returns a list: pdf values, partition function
      p(x,y) = (1/Z) m(y) exp[ -lda t(x,y) ] 
      Z(lda,m) = \int m(y) exp[ -lda t(x,y) ] dx dy
    """
    def _ratio(self, lda):
        F = self.F
        dimX = self.dimX
        dimY = self.dimY
        U = np.dot(lda, F.reshape(F.shape[0], dimX*dimY)).reshape(dimX, dimY)
        return np.exp(-U) 
    
    def _pdf(self, lda, mY):
        ratio = self._ratio(lda)
        p = mY*ratio
        Z = p.sum()
        p = p/Z
        return p, Z

    def pdf(self, lda, mY):
        return self._pdf(lda, mY)[0]

    def Z(self, lda, mY):
        return self._pdf(lda, mY)[1]

    def z(self, lda):
        return self._ratio(lda).sum(0)
    
    """
    Variational dual function
    """        
    def vdual(self, lda, mY):
        return -np.log(self.Z(lda, mY))

    def adjust_lda(self, mY, lda0=None, method=_METHOD, tol=_TOL):
        if lda0 == None:
            lda0 = np.zeros(self.nfeatures)            
        def cost(lda):
            return -self.vdual(lda, mY)
        if method == 'bfgs':
            lda = optimize.fmin_bfgs(cost, lda0, gtol=tol)
        elif method == 'cg': 
            lda = optimize.fmin_cg(cost, lda0, gtol=tol)
        elif method == 'powell': 
            lda = optimize.fmin_powell(cost, lda0, ftol=tol)
        else:
            print('Unknown optimization method.')
            lda = lda0        
        self._lda = lda
        return lda

    def adjust_mY(self, lda, mY):
        mY = self.pdf(lda, mY).sum(0)
        self._mY = mY
        return mY

    """
    Blahut Arimoto algorithm
    """
    def fit(self, lda=None, mY=None, niter=_NITER, method=_METHOD):
        if lda == None:
            lda = np.zeros(self.nfeatures)            
        if mY == None:
            mY = np.ones(self.dimY)/self.dimY 
        for i in range(niter): 
            print ('Iteration %d' % i)
            print('B-step...')
            lda = self.adjust_lda(mY, lda0=lda, method=method)
            print('A-step...')
            mY = self.adjust_mY(lda, mY)
        return self.pdf(lda, mY)

    """
    Fixed lambda Blahut Arimoto algorithm
    """
    def dual(self, lda, mY=None, niter=_NITER, method=_METHOD):
        if mY == None:
            mY = np.ones(self.dimY)/self.dimY 
        for i in range(niter): 
            print ('Iteration %d' % i)
            mY = self.adjust_mY(lda, mY)
        return self.vdual(lda, mY)


class feature:
    
    def __init__(self, dim):
        self.dimX = dim[0]
        self.dimY = dim[1]        
        self.features = []
        self.nfeatures = 0

    def add(self, fun): 
        f = np.zeros([self.dimX, self.dimY])
        for x in range(self.dimX):
            for y in range(self.dimY): 
                f[x,y] = fun(x, y)
        self.features.append(f)
        self.nfeatures = self.nfeatures + 1

    def mean(self, H):
        Fmean = []
        for f in self.features:
            Fmean.append(np.sum(f*H))
        return Fmean   


def pdfshow(p, gamma=.1, invert=False):
    stamp_x = 'Unknown x'
    stamp_y = 'Observation y'
    if invert==True: 
        pylab.imshow(p**gamma)
        pylab.xlabel(stamp_y)
        pylab.ylabel(stamp_x)
    else:
        pylab.imshow((p.T)**gamma)
        pylab.xlabel(stamp_x)
        pylab.ylabel(stamp_y)
    myaxes = pylab.axes()
    ymin, ymax = myaxes.get_ylim()
    myaxes.set_ylim(ymax, ymin)
    pylab.show()



##H = np.ones([12,12])
##H = np.random.rand(12, 12)

H = np.load('ammon_anubis.npz')['H']

F = feature(H.shape)

if True:
    F.add(ssd)
else: 
    F.add(m1x)
    F.add(m1y)
    F.add(m2xx)
    F.add(m2yy)
    F.add(m2xy)

fmean = F.mean(H)
M = maxCEnt(F.features, fmean)

"""
mY0 = np.ones(M.dimY)/M.dimY
lda = M.adjust_lda(mY0)
p, Z, U = M._pdf(lda, mY0)
mY = M.adjust_mY(lda, mY0)
"""

p = M.fit()
pdfshow(p)

pylab.plot(M._mY)
pylab.show()


"""
dual = M.dual(M._lda, niter=1000)
pylab.plot(M._mY)
pylab.show()
"""

"""
ldas = (np.arange(100)-50)/5. + M._lda
duals = np.zeros(len(ldas))
for i in range(len(ldas)):
    duals[i] = M.dual(ldas[i])
    
pylab.plot(ldas, duals)
pylab.show()
"""
