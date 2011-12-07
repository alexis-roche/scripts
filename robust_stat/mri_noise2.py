import numpy as np
from scipy.stats import norm


def robust_moments(x, r):
    """
    Given a sample `x`, compute the mean and mean of squares of the
    sub-sample obtained by rejecting a proportion `r` of the largest
    values.
    """
    xs = x.copy()
    xs.sort()
    xs = xs[0:np.ceil((1 - r) * len(x))]
    m1 = np.mean(xs)
    m2 = np.mean(xs ** 2)
    m4 = np.mean(xs ** 4)
    print m1, m2, m4
    m1_t, m2_t, m4_t = correction_factors(r)
    print m1_t, m2_t, m4_t
    # Robust estimates
    e_x = m1 / m1_t
    e_x2 = m2 / m2_t
    e_x4 = 3 * m4 / m4_t
    var_x2 = e_x4 - e_x2 ** 2
    k = 2 * e_x2 ** 2 / var_x2  # degrees of freedom
    s2 = e_x2 / k  # squared scale
    print e_x, e_x2, e_x4, var_x2, k, s2
    return k, s2


def truncated_moments(r):
    """
    Compute the truncated moments of a chi variate with 1 DOF.
    Let In = int_0^c x^n exp[-(x^2)/(2c^2)] dx

    We have:
     I0 = sqrt(pi/2) * (1-r)
     I1 = 1 - exp(-c^2/2)
     I_n+1 = n * I_n-1 - exp(-c^2/2) c^n
    """
    c = norm.ppf(1 - .5 * r)  # cutoff value
    gc = np.exp(-.5 * c ** 2)
    I0 = np.sqrt(.5 * np.pi) * (1 - r)
    I1 = 1 - gc
    I2 = I0 - gc * c
    I4 = 3 * I2 - gc * c ** 3
    return I0, I1, I2, I4


def correction_factors(r):
    """
    Assume a chi distribution with 1 degree of freedom.
    """
    I0, I1, I2, I4 = truncated_moments(r)
    return I1 / I0, I2 / I0, I4 / I0


def noise_std(s, order=1):
    if order == 2:
        s = np.sqrt(s)
    return np.sqrt(1 - 2 / np.pi) * s


def simulate_noise(k, s, n):
    if k == np.inf:
        return np.abs(np.random.normal(scale=s, size=(n,)))
    x = s * np.random.normal(size=(k, n))
    return np.sqrt((x ** 2).sum(0))


# Parameters
s = 2.3
r = .5
k = 3
n = 30000

# Simulate background noise
x = simulate_noise(k, s, n)

# Compute truncated estimates
k, s2 = robust_moments(x, r)
print k, np.sqrt(s2)
