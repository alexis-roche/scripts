import numpy as np 


def wmedian(x, w): 
    ind = np.argsort(x) # x[ind] increasing
    F = np.cumsum(w[ind])
    f = .5*(w.sum()+1)
    i = np.searchsorted(F, f)
    if i == 0: 
        return x[ind[0]]
    wr = (f-F[i-1])/(F[i]-F[i-1])
    jr = ind[i]
    jl = ind[i-1]
    return wr*x[jr]+(1-wr)*x[jl]
    

def test_wmedian(size): 
    x = np.random.rand(size)
    w = np.ones(size)
    m0 = np.median(x)
    m = wmedian(x,w)
    err = m-m0
    print ('Test (%d): error = %f' % (size, err))
    return err

test_wmedian(100)
test_wmedian(101)

#test_wmedian(100)
#test_wmedian(101)



