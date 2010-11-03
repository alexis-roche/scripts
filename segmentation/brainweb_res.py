import numpy as np 
from pylab import * 
from os.path import join 
from glob import glob 

#METH = 'mf'
CASES = [0,1,2,3,4,5,6,7]
beta = 0.2
sigma = 500
SIMU = 'beta' + str(beta).replace('.','_') + '__sigma' + str(sigma)

# Editable parameters 
PREFIX = join('D:\home', 'Alexis', 'data')
#SAVEDIR = join(PREFIX, 'brainweb', 'results2', SIMU)
SAVEDIR = join(PREFIX, 'brainweb', 'results2')
TISSUES = ['CSF','GM','WM']

# Get data 
def get_data(meth): 
    files = glob(join(SAVEDIR, 'brainweb*_'+meth+'.npz'))
    print files 
    tmp = np.load(files[0])
    d0 = tmp['d']
    f0 = tmp['f']
    d = np.zeros(list(d0.shape)+[len(files)])
    f = np.zeros(list(f0.shape)+[len(files)])
    for i in range(len(files)): 
        tmp = np.load(files[i])
        d[:,:,i] = tmp['d']
        f[:,i] = tmp['f']

    # Masking 
    offset = 0
    if meth == 'ox2': 
        offset = 5
    d = d[offset:,:,CASES]
    f = f[offset:,CASES]
    return d, f

def fstat(f): 
    fm = f.mean(1)
    fs = f.std(1)
    return fm, fs

def dstat(d, tissue): 
    idx = TISSUES.index(tissue)
    dm = d[:,idx,:].mean(1)
    ds = d[:,idx,:].std(1)
    return dm, ds

def conv_rate(u): 
    L = u[-1,:]
    uc = u - L 
    vc = np.zeros(u.shape)
    vc[:-1,:] = u[1:,:]
    vc[-1,:] = L 
    return vc/uc 

def plotres(t=None):
    methods = ['dd', 'mf', 'ox', 'ox2']
    lines = []
    for meth in methods: 
        d, f = get_data(meth)
        if t in TISSUES: 
            m, s = dstat(d, t)
        else: 
            m, s = fstat(f)
        x = np.arange(m.size)
        lines.append(errorbar(x, m, s))
    legend([l[0] for l in lines], methods, loc=0)

plotres('GM')
