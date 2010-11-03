import numpy as N 
import numpy.random as NR
import fff.mfx.routines as F
from pylab import * 
import scipy.stats as SS

# Second-level parameters
m0 = 0
m1 = 0
p0 = .5
p1 = 1 - p0
vf = 1

# First-level parameters
n = 10
stat = F.LR_gmfx
##v = NR.random(size=n)
v = NR.gamma(3,scale=1/float(6),size=n)

# Generate sample
k = NR.binomial(1, p1, size=n)
x = (m1*k + m0*(1-k)) + NR.normal(0,sqrt(vf), size=n)
y = x + NR.normal(0,sqrt(v))

# Compute empirical LR for a number of random permutations
nlim = 14
if n <= nlim:
	nsimu = 2**n
	magic = 0
else:
	nsimu = 2**nlim
	magic = -1
 
print nsimu 

# MFX
T = F.test_stat_mfx(y,v,stat=stat,niter=10,nsimu=nsimu,magic=magic)
T = array([T,-T])
T = N.ravel(T)
nsimu = 2*nsimu
u = N.arange(-5,5+.25,.25)
h = N.zeros(N.size(u))
for i in range(N.size(u)):
	h[i] = N.size(N.where(T>=u[i]))/float(nsimu)

# Display
Zu = SS.norm.ppf( 1-h ) 
I = N.where(abs(Zu)<100)
plot( u[I], Zu[I], linewidth=2 )

plot( u, u, 'r--', linewidth=2 )

xticks( size=14 )
yticks( size=14 )

xlabel('$t_r$', fontsize=16)
ylabel('z-score', fontsize=16)

print u 
