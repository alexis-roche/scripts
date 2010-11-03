import numpy as np 
import scipy.optimize as optimize
import pylab 


"""
Simple BA algorithm. 

Alternates two steps: 

1. B-step:
  Compute p(i|j)=m(i)exp[-lda*(i-j)^2]/Z(lda,j) ==> Z(lda,j)=sum_i m(i)exp[...]
2. A-step: 
  Marginalize: m(i)=sum_j h(j)p(i|j)

H: joint histogram, first dim is 'j', second dim is 'i'
"""

_NITER = 10
_TINY = 1e-20
_OPTIMIZER = 'powell'
_TOL = 1e-5



def sd(dimI, dimJ):
    s = np.zeros([dimJ, dimI])
    for i in range(dimI):
        for j in range(dimJ): 
            s[j,i] = (i-j)**2
    return s 

def hisplay(p, gamma=.1):
    stamp_x = 'Target j'
    stamp_y = 'Source i'
    pylab.imshow((p.T)**gamma)
    pylab.xlabel(stamp_x)
    pylab.ylabel(stamp_y)
    myaxes = pylab.axes()
    ymin, ymax = myaxes.get_ylim()
    myaxes.set_ylim(ymax, ymin)

class Lisboa: 

    def __init__(self, H, m0=None, tiny=_TINY): 
        H = H/H.sum()
        self.hI = H.sum(0)
        self.hJ = H.sum(1)
        self.SD = sd(H.shape[1], H.shape[0])
        if m0==None: 
            dimI = self.hI.size
            m0 = np.ones(dimI)/float(dimI)
        self._lda = 0.0 
        self._m = m0
        self._p = None
        self.niter = 0 
        self.tiny = tiny

    def update_joint_pdf(self):
        """
        Update self._p
        """
        aux = self._m*np.exp(-self._lda*self.SD)
        z = aux.sum(1)
        # Numerical test
        z = np.maximum(z, self.tiny)
        self._p = (aux.T/z).T

        self._Z = 

    def update_marginal_pdf(self):
        """
        Update self._m
        """
        self._m = (self.hJ*self._p.T).T.sum(0)


    def adjust_lda(self, optimizer=_OPTIMIZER, tol=_TOL):
        self._lda = lda
        def cost(lda):
            return -self.vdual(lda, mY)
        if optimizer == 'bfgs':
            lda = optimize.fmin_bfgs(cost, lda0, gtol=tol)
        elif optimizer == 'cg': 
            lda = optimize.fmin_cg(cost, lda0, gtol=tol)
        elif optimizer == 'powell': 
            lda = optimize.fmin_powell(cost, lda0, ftol=tol)
        else:
            print('Unknown optimization method.')
            lda = lda0        
        self._lda = lda

    def fit(self):
        return (self.hJ*self._p.T).T

    def ba(self, fixed_lda=None, niter=_NITER):
        if fixed_lda==None:
            adjust_lda = True
        else:
            adjust_lda = False
            self._lda = fixed_lda
        for i in range(niter):
            if adjust_lda:
                self.adjust_lda()
            self.update_joint_pdf()
            self.update_marginal_pdf()
            self.niter = i+1

    
    def show_m(self, gamma=.1):
        pylab.plot(self._m**gamma)
        pylab.plot(self.hI**gamma, 'r')
        pylab.show()

    def show_fit(self, gamma=.1):
        hisplay(self.fit(), gamma)
        pylab.show()
        



H = np.load('ammon_anubis.npz')['H']
L = Lisboa(H)
L.ba(fixed_lda=0.1, niter=10)

L.show_m()


