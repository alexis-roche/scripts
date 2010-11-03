import numpy as np
import scipy.ndimage as nd
import nifti 
import os 
import pylab 

datadir = 'D:\home\AR203069\data\delphine'
tissues = ['CSF','GM','WM','REST']
def_size = 2

class PPM: 

    # Open PPMs (White Matter, Gray Matter, CSF, Rest)
    def __init__(self): 
        P = {}
        for tissue in tissues: 
            fname = 'out'+tissue+'_100.img'
            im = nifti.load(os.path.join(datadir, fname))
            P[tissue] = im.get_data()/1000.
        self.P = P 
        self.renormalize()

    def renormalize(self):
        S = self.sum()
        self.mask = np.where(S > 0)
        for tissue in tissues: 
            print tissue
            self.P[tissue][self.mask] /= S[self.mask]
        
    def map(self, tissue):
        return self.P[tissue]

    def sum(self): 
        # Sum up values across tissues 
        S = None
        for tissue in tissues:
            if S == None: 
                S = self.P[tissue].copy()
            else:
                S += self.P[tissue]
        return S

    def display(self, tissue='REST', slice=80): 
        pylab.figure()
        pylab.imshow(self.P[tissue][:,slice,:])

    def display_all(self, slice=80): 
        for tissue in tissues: 
            self.display(tissue, slice)

    def closing(self, tissue, size=def_size):
        nd.grey_closing(self.P[tissue], size=[size,size,size], output=self.P[tissue]) 

    def opening(self, tissue, size=def_size):
        nd.grey_opening(self.P[tissue], size=[size,size,size], output=self.P[tissue]) 

    def median_filter(self, tissue, size=def_size): 
        nd.median_filter(self.P[tissue], size=[size,size,size], output=self.P[tissue]) 

    def mean_filter(self, tissue, size=def_size): 
        nd.filters.uniform_filter(self.P[tissue], size=[size,size,size], output=self.P[tissue]) 

    def gaussian_filter(self, tissue, size=def_size): 
        size /= 2.*np.sqrt(2*np.log(2)) # convert implicit FWHM into sigma units
        nd.filters.gaussian_filter(self.P[tissue], sigma=[size,size,size], output=self.P[tissue]) 

    def log_opinion_poll(self, tissue, size=def_size, tiny=1e-20): 
        """
        Compute P(k) = prod_i P_i(k) where i represents neighbouring
        voxels. 
        """
        self.P[tissue] = np.log(np.maximum(tiny, self.P[tissue]))
        self.gaussian_filter(tissue, size)
        tmp = np.exp(self.P[tissue][self.mask])
        self.P[tissue] = np.zeros(self.P[tissue].shape)
        self.P[tissue][self.mask] = tmp 



# Main program
P = PPM()

# Dummy instructions to force pylab in 'show' mode
tmp = np.random.rand(10)
pylab.plot(tmp)
pylab.show()
pylab.close()

# Display initial (dirty) maps
pylab.pink()
P.display_all()

"""
# Morphomat 
P.closing('REST')
P.opening('WM')
P.opening('GM')
P.opening('CSF')
P.renormalize()
"""
"""
P.median_filter('REST', size=[5,5,5])
"""

# Knowledge-based rules
# If the majority of voxels think we are not in REST or CSF, then 
msk = np.where(P.map('CSF')+P.map('REST')>P.map('GM'))
P.map('GM')[msk] = 0.

# Log opinion poll
for tissue in tissues: 
    P.log_opinion_poll(tissue)

# Finally, 
P.renormalize()

# Display clean maps
P.display_all()





