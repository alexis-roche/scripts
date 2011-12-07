import numpy as np
from scipy.stats import norm


def robust_variance(x, r):
    """
    Given a sample `x`, compute the empirical variance of the
    sub-sample obtained by rejecting a proportion `r` of the largest
    values.
    """
    xs = x.copy()
    xs.sort()
    xs = xs[0:np.ceil((1 - r) * len(x))]
    return correction_factor(r) * np.var(xs)


def correction_factor(r, scale=False):
    """
    Correct the truncated variance estimate assuming a chi
    distribution with 1 degree of freedom.

    By default, corrects for the variance of the observed variable
    (assumed to be the absolute value of a Gaussian). If scale ==
    True, corrects for the variance of the underlying Gaussian
    variable.
    """
    c = norm.ppf(1 - .5 * r)  # cutoff value
    K = np.sqrt(2 / np.pi) / (1 - r)
    gc = np.exp(-.5 * c ** 2)
    m1 = K * (1 - gc)
    m2 = 1 - K * c * gc
    f = 1 / (m2 - m1 ** 2)
    if scale == False:
        f *= 1 - 2 / np.pi
    return f


def noise_std(s, order=1):
    if order == 2:
        s = np.sqrt(s)
    return np.sqrt(1 - 2 / np.pi) * s


# Parameters
s = 1
std = noise_std(s)
r = .000000001

# Simulate background noise
x = np.abs(np.random.normal(scale=s, size=(30000,)))

# Variance estimate
v = robust_variance(x, r)

print std, np.sqrt(v)
