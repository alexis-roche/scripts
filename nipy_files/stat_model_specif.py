import sympy 
from sympy import * 

"""
class Term(sympy.Symbol):
    
    def __add__(self, other): 
        if self == other: 
            return self
        else: 
            return sympy.Symbol.__add__(self, other)

w1 = Term('word1')
w2 = Term('word2')
w3 = Term('word3')
m1 = Term('visual')
m2 = Term('audio')
f = w1*m1 + w2*m1 + w3*m1 + w1*m2 + w2*m2 + w3*m2 
"""

class Formula: 
    
    def __init__(self, f): 
        self.__symbol__ = f
        self.terms = f.as_coeff_factors()[1]

    def __str__(self): 
        return self.__symbol__.__str__()

class HRFModel: 
    
    def __init__(self): 
        self.dum = None

    def conv(self):
        

"""
visual = Symbol('visual')
audio = Symbol('audio')

w1 = Symbol('word1')
w2 = Symbol('word2')
w3 = Symbol('word3')

c1 = visual*w1
c2 = visual*w2
c3 = visual*w3
c4 = audio*w1
c5 = audio*w2
c6 = audio*w3

words = Factor(['word1','word2','word3'])
pres = Factor(['visual','audio'])
model = words*pres
"""

    
class Condition: 
    
    def __init__(self, name): 
        self.sym = Symbol(name)
        self.regressors = None
        self.filters = None

    def regressors(self, onsets, amplitudes=None, filters=[glover]): 
        self.regressors = []
        self.filters = filters
        for f in filters:
            self.regressors.append(events(onsets, amplitudes=amplitudes, g=self.sym, f=f))

    def __sub__(self, other): 
        return self.sym - other.sym

c1 = Condition('visual word1')
c2 = Condition('visual word2')
c3 = Condition('visual word3')
c4 = Condition('audio word1')
c5 = Condition('audio word2')
c6 = Condition('audio word3')

t = Symbol('t')
regressors = [t, t**2, t**3] + ['ux', 'uy'] + ['heartbeat']

m = Design(conditions=[c1,c2,c3], regressors=regressors)
## Regressors: session-dependent? 

# Proposal 1
m.add_contrast(c1-c2) ## Ignoring ambiguity problem
m.set_condition(sess, c1, onsets=o1, amplitudes=a1, duration=d1)
m.set_regressor(sess, ux, val=array, timestamps=timestamps, interp=cubic_spline, units='tr')


# Proposal 2
"""
Fernando suggests that models should be instantiated as experiments
from ordered dictionaries representing sessions.
"""
con = c1-c2 ## Ignoring ambiguity problem
expe = m.instantiate(sessions, hrf=[glover,glover1])
expe.set_condition(c1, onsets=o1, amplitudes=a1, duration=d1, session=sess1)
expe.set_regressor(ux, val=array, timestamps=timestamps, interp=cubic_spline, units='tr', session=sess1)

"""
In any case, we need to define different onsets for each session,
while keeping the notion of onset-independent contrast.

Should we pass timestamps and onsets information to the method that
instantiates a model? 

We might want to add a global regressor later. 
"""

"""
In this approach, a condition is absent from a session if not
explicitly set. Missing conditions should not be associated with
columns in the design matrix.
"""
X = expe.design_matrix(sess, timestamps)
con_vect = expe.contrast(sess, c1-c2, params)

"""
Here there are three levels: 
1. model 
2. experience 
3. design matrix
"""

"""
Do we add a method to output multi-session concatenated matrices? 
"""
X = []
for sess in [s1, s2, s3, s4]: 
    X.append(m.design())


