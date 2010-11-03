import numpy as np 
import scipy.optimize as optimize
import pylab 

"""
H: joint histogram, first dim is 'i', second dim is 'j'
F : number of features x dimI x dimJ 

H and m MUST be normalized

i is the variable to be predicted and j is the context. 
"""

def ssd(i,j):
    return (i-j)**2

def m1i(i,j):
    return i

def m1j(i,j):
    return j

def m2ij(i,j):
    return i*j

def m2ii(i,j):
    return i**2

def m2jj(i,j):
    return j**2


class maxentfit: 

    def __init__(self, H, mI): 
        self.dimI = H.shape[0]
        self.dimJ = H.shape[1]
        self.H = H/H.sum()
        self.nfeatures = 0
        self.F = []
        self._F = None
        self.hJ = self.H.sum(0)
        self.mI = mI/mI.sum()

    def add_feature(self, fun, norm=1.0): 
        f = np.zeros([self.dimI, self.dimJ])
        for i in range(self.dimI):
            for j in range(self.dimJ): 
                f[i,j] = fun(norm*i,norm*j)
        self.F.append(f)
        self.nfeatures = self.nfeatures + 1
        self._F = np.asarray(self.F)

    def _probdist(self, lda):
        F = self._F
        dimI = self.dimI
        dimJ = self.dimJ
        mI = self.mI
        aux = 0.0
        aux = np.exp(-np.dot(lda, F.reshape(F.shape[0], dimI*dimJ)).reshape(dimI, dimJ))
        aux = (mI*aux.T).T
        Z = sum(aux)
        return Z, aux 
        
    def probdist(self, lda):
        Z, aux = self._probdist(lda) 
        p = (aux/Z)*self.hJ ## p(i|j)*p(j)
        return p

    def dual(self, lda):
        H = self.H
        F = self._F
        mI = self.mI
        hJ = self.hJ 
        tau = (H*F).sum(1).sum(1)
        Z = self._probdist(lda)[0]
        logZ = np.log(Z)
        psi = np.dot(hJ, logZ) + np.dot(lda, tau)
        return psi

    def fit(self, lda0=None, method='bfgs'):
        if lda0 == None:
            lda0 = np.zeros(self.nfeatures)            
        def cost(lda):
            return self.dual(lda)
        if method == 'bfgs':
            lda = optimize.fmin_bfgs(cost, lda0)
        elif method == 'cg': 
            lda = optimize.fmin_cg(cost, lda0)
        else:
            print('Unknown optimization method.')
            lda = lda0        
        return lda

    """
    Blahut Arimoto algorithm
    """
    def ba(self, lda0=None, method='bfgs', niter=5):
        if lda0 == None:
            lda0 = np.zeros(self.nfeatures)            
        lda = lda0
        for i in range(niter): 
            print ('Iteration %d' % i)
            lda = self.fit(lda, method)
            P = self.probdist(lda)
            self.mI = P.sum(1)
        return lda
    


def hisplay(H, gamma=.1, invert=True):
    stamp_i = 'Source i'
    stamp_j = 'Target j'
    if invert==True: 
        pylab.imshow((H.T)**gamma)
        pylab.xlabel(stamp_j)
        pylab.ylabel(stamp_i)
    else:
        pylab.imshow(H**gamma)
        pylab.xlabel(stamp_i)
        pylab.ylabel(stamp_j)
    myaxes = pylab.axes()
    ymin, ymax = myaxes.get_ylim()
    myaxes.set_ylim(ymax, ymin)
    pylab.show()


##H = np.ones([12,12])
##H = np.random.rand(12, 12)
H = np.load('ammon_anubis.npz')['H']
mI = np.ones(H.shape[0])

M = maxentfit(H, mI)

norm = 2./(H.shape[0]+H.shape[1])

"""
M.add_feature(ssd, norm=norm)
"""
M.add_feature(m1i, norm=norm)
M.add_feature(m1j, norm=norm)
M.add_feature(m2ii, norm=norm)
M.add_feature(m2jj, norm=norm)
M.add_feature(m2ij, norm=norm)

"""
lda = M.fit()
"""
lda = M.ba()

P = M.probdist(lda)


""" 
Gaussian fit
"""
tau = (M.H*M._F).sum(1).sum(1)
mu = tau[0:2]
S = np.array([[tau[2], tau[4]], [tau[4], tau[3]]])
Sinv = np.linalg.inv(S)

Pg = np.zeros(H.shape)
for i in range(Pg.shape[0]):
    for j in range(Pg.shape[1]): 
        dm = norm*np.array([i,j]) - mu 
        Pg[i,j] = np.exp( -.5* np.dot(dm, np.dot(Sinv,dm)))

Pg = Pg/Pg.sum()
