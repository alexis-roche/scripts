import numpy as np 

nruns = 10 

transfo_mean = np.random.rand(nruns)
transforms = [np.random.rand(nruns) for i in range(nruns)]
toto = transforms[5].copy()

for i in np.arange(nruns):
    transforms[i] = [t*transfo_mean[i] for t in transforms[i]]


x = transforms[5]/toto # should be equal to transfo_mean[5]
xx = transfo_mean[5]


