"""
Robust procedure to estimate the background noise std deviation in an
MR image.

We assume that the background noise follows a chi-distribution with k
degrees of freedom, where k is twice the number of effective coils,
and an unknown scale parameter to be estimated.

Algorithm:
1. Square all input intensities
2. Sort by ascending order and throw away the largest 50%
3. Compute the mean of the retained values
4. Apply a correction factor
"""
import numpy as np
from scipy.stats import chi2
from scipy.special import gamma


def estimate_scale(x, k, r):
    """
    Estimate the squared scale parameter of a chi distribution from a
    sample x, rejecting a proportion r of the largest values.
    """
    x2 = x ** 2
    x2.sort()
    x2_t = x2[0:np.ceil((1 - r) * len(x2))]
    m2 = np.mean(x2_t)
    c = chi2.ppf(1 - r, k)  # cutoff value
    m2_th = truncated_chi2_mean(c, k)
    return m2 / m2_th


def noise_variance(s2, k):
    """
    Compute the noise variance depending on the squared scale s2 and
    the degrees of freedom k.

    Use: Var(X) = E(X**2) - E(X)**2
     E(X**2) = k s2
     E(X) = sqrt(2) Gamma((k+1)/2)/Gamma(k/2) s
    """
    f = gamma(.5 * (k + 1)) / gamma(.5 * k)
    return (k - 2 * f ** 2) * s2


def truncated_chi2_mean(c, k):
    """
    chi2 mean up to the cutoff.

    Compute A/B with:
    A = integral[0..c] x chi2(k) dx
    B = integral[0..c] chi2(k) dx
    B is computed via chi2.cdf and A is computed via:
    A = k * integral[0..c] chi2(k+2) dx
    """
    A = k * chi2.cdf(c, k + 2)
    B = chi2.cdf(c, k)
    return A / B


def simulate_noise(k, s, n):
    x = s * np.random.normal(size=(k, n))
    return np.sqrt((x ** 2).sum(0))


# Degrees of freedom
k = 18  # twice the number of coils

# Scale
s = 2.5

# Simulate background noise
x = simulate_noise(k, s, 30000)

# Estimate the squared scale parameter
s2_e = estimate_scale(x, k, 0.5)
s_e = np.sqrt(s2_e)

# Estimate the noise std
std_e = np.sqrt(noise_variance(s2_e, k))

# True noise std
std = np.sqrt(noise_variance(s ** 2, k))


print s, s_e
print std, std_e, np.std(x)
