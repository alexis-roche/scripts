from nipy.neurospin.image import load_image, save_image
from nipy.neurospin.registration import IconicRegistration, Affine
from nipy.utils import example_data
from os.path import join 
import numpy as np 
import pylab 


"""
Random walk proposal (gaussian)
"""
def random_walk(T, T0, sigma=1.): 
    nparams = T.param.size
    T2 = Affine(subtype=T.subtype, flag2d=T.flag2d)
    T2.param = T.param + sigma*np.random.normal(size=nparams)
    return T2, 1.

"""
Independence chain proposal (gaussian)
"""
def independence_chain(T, T0, sigma=1.): 
    nparmas = T.param.size
    T2 = Affine(subtype=T.subtype, flag2d=T.flag2d)
    T2.param = T0.param + sigma*np.random.normal(size=nparams)
    sigma_ = np.maximum(sigma, 1e-20)
    tmp = (T2.param/sigma_)**2-(T.param/sigma_)**2
    r = np.exp(-.5*tmp.sum()) 
    return T2, r

    


# Load images 
def make_matcher(): 
    I = load_image(example_data.get_filename('neurospin',
                                             'sulcal2000',
                                             'nobias_ammon.nii.gz'))
    J = load_image(example_data.get_filename('neurospin',
                                             'sulcal2000',
                                             'nobias_anubis.nii.gz'))


    # Create a registration instance
    R = IconicRegistration(I, J)
    R.set_source_fov(fixed_npoints=64**3)
    R.similarity = 'llr_cc'
    return R


R = make_matcher()

# Initial transform 
T0 = Affine(radius=10)
##T0 = R.optimize(search='affine', radius=10)
nparams = T0.param.size

# Very empirical...
lda = 1./float(R._source_npoints)
##lda *= 100

sigma = 0.1*np.ones(nparams)
sigma[6:9] = 0.0 ## freeze scaling parameters for now 

niters = 1000
T = T0 
results = np.zeros([niters, nparams])
logL = lda*R.eval(T)
na = 0 
propose = random_walk
##propose = independence_chain

print('starting Metropolis')
for i in range(niters): 
    T2, r = propose(T, T0, sigma=sigma)
    logL2 = lda*R.eval(T2)
    a = np.exp(logL2-logL)/r # acceptance 
    print('a = %f' % a) 
    if np.random.rand() <= a: 
        print('  accepted')
        T = T2
        logL = logL2
        na += 1
    results[i,:] = T.param

print ('Overall acceptance rate: %f' % (na/float(niters)))

print('*** Start ***')
print(T0)
print('*** Final ***')
print(T)
