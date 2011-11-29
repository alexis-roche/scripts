import numpy as np
import scipy.stats as ss


def get_cutoff(r):
    """
    Returns the cutoff value (in proportion of the standard deviation)
    corresponding to the input rejection rate
    """
    return ss.norm.ppf(1 - .5 * r)


def correction_factor(r):
    c = get_cutoff(r)
    if c == np.inf:
        I = 1 - r
    else:
        I = -np.sqrt(2 / np.pi) * c * np.exp(-.5 * c ** 2) + 1 - r
    return 1 / I


for r in np.linspace(0, .5, num=11):
    print('Rejection rate: %f, variance correction factor: %f'\
              % (r, correction_factor(r)))
