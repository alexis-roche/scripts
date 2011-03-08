import numpy as np 



def metropolis(p, x0, sigma, niters=100): 
    x = np.zeros(niters)
    m = np.zeros(niters)    
    xs = x0 
    ps = p(x0)
    mu = x0 
    for i in range(niters): 
        xp = mu + sigma*np.random.normal()
        pp = p(xp)
        if np.random.rand() < pp/ps: 
            xs = xp
            ps = pp
        mu = xs # mu + (xs-mu)
        x[i] = xs 
        m[i] = mu 
    return x, m 


def robbins_monro(p, x0, sigma, niters=100): 
    x = np.zeros(niters)    
    m = np.zeros(niters)    
    xs = x0 
    ps = p(x0) 
    mu = x0 
    for i in range(niters):
        xs = mu + sigma*np.random.normal()
        ws = p(xs)*gaussian(xs, mu, sigma) 
        mu = mu + ws*(xs-mu)
        x[i] = xs
        m[i] = mu 
    return x, m 

        
def gaussian(x, m, s): 
    u = (x-m)/s 
    return np.exp(-(u**2/2.))/(s*np.sqrt(2.*np.pi)) 

def target(x): 
    return gaussian(x, 0, 1) 

x, mx = metropolis(target, 5, 1, niters=1000)
y, my = robbins_monro(target, 5, 1, niters=1000)

