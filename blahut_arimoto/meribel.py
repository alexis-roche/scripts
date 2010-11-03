"""
X gaussian with fixed parameters 
We do BA under the constraint E[(Y-X)^2] == d2

Clearly, Y is gaussian too so we are left with a 2-dimensional fitting
pb
"""
import numpy as np 

mX = 0.0
sX = 1.0
d2 = .5
niters = 1000

# Init 
mY = 0.0
sY = 1e5 

# Do not edit 
ldX = .5/(sX**2)

def b_step(mY, sY): 
    """
    Find lda such that: p(x,y) \propto g(x)g(y)exp[-lda*(x-y)**2]
    """
    ldY = .5/(sY**2)
    two_b = ldX+ldY
    dm = mX-mY
    Delta = two_b**2/4. + 4.*d2*((ldX*ldY*dm)**2)
    gamma = (.5*two_b + np.sqrt(Delta))/(2*d2)
    lda = (gamma - ldX*ldY)/two_b
    return lda 

def a_step(lda, mY, sY):
    """
    Marginalize p(x,y) wrt x --> update g(y)
    """
    ldY = .5/(sY**2)
    gamma = ldX*ldY + (ldX+ldY)*lda 
    sY_ = np.sqrt(.5*(ldX+lda)/gamma)
    mY_ = (ldY*(ldX+lda)*mY + lda*ldX*mX)/gamma
    sX_ = np.sqrt(.5*(ldY+lda)/gamma)
    mX_ = (ldX*(ldY+lda)*mX + lda*ldY*mY)/gamma
    print ('*** mX_, sX_ = %f, %f' % (mX_, sX_))
    return mY_, sY_


def ds(lda, mY, sY): 
    dm = mX-mY 
    ldY = .5/(sY**2)
    gamma = ldX*ldY + (ldX+ldY)*lda 
    #return dm**2 + .5*(ldX+ldY)/gamma
    A = ((ldX*ldY*dm)/gamma)**2
    B = (ldX+ldY)/(2*gamma)
    return A+B

def textor(lda, mY, sY): 
    ldY = .5/(sY**2)
    gamma = ldX*ldY + (ldX+ldY)*lda 
    detA = 4*gamma
    S = 2.*lda*np.ones([2,2])
    S[0][0] += 2*ldY
    S[1][1] += 2*ldX
    S /= detA 
    tmp = S[0][0]+S[1][1]-2.*S[0][1]
    print('*** Var(X-Y)=%f' % tmp)
    print('*** detA=%f' % detA)
    return S 

# BA-like algorithm
for it in range(niters): 
    print('Iter n.%d' % it)
    print('B-step')
    lda = b_step(mY, sY)
    print(' lda = %f' % lda)
    print('A-step')
    mY, sY = a_step(lda, mY, sY) 
    print(' mY, sY = %f, %f' % (mY, sY))

