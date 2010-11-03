import numpy as np 

TOL = 1e-10
TINY = 1e-20
INF = 1e100
ITERMAX = 10

class elaw: 
    
    def __init__(self, x, lda): 
        self.x = x
        self.lda = lda
        if lda < INF:
            p = np.minimum(np.exp(-lda*x), INF)
            ##p = np.exp(-lda*x)
            self.z = p.sum()
            self.p = p/self.z
        else: 
            p = np.zeros(x.shape)
            p[x.argmin()] = 1.0
            self.p = p
            self.z = 0.0

    def moments(self): 
        m = (self.x*self.p).sum()
        v = ((self.x**2)*self.p).sum() - m**2
        return m, v



def newton(x, m, tol=TOL, positive_lda=True):
    # Special case: mean value too small, model not identifiable
    if m < x.min():
        return elaw(x, INF)
    iter = 0
    lda = 0.0
    delta = tol+1
    while delta > tol:
        el = elaw(x, lda)
        mp, vp = el.moments()
        delta = mp - m 
        lda = lda + (delta/np.maximum(vp, TINY))
        delta = np.abs(delta)
        iter = iter + 1
        if iter > ITERMAX: 
            delta = 0
        print lda, mp, vp
    # Discard negative lda values
    if positive_lda:
        if lda < 0: 
            el = elaw(x, 0.0)
    return el


def nml(x, tol=TOL, positive_lda=True): 
    y = np.zeros(x.shape)
    yf = y.ravel()
    xf = x.ravel()
    for i in range(xf.size):
        el = newton(xf, xf[i], tol, positive_lda)
        yf[i] = el.p[i]

    return y 

        
