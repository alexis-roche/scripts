import numpy as np
from scipy.special import gamma


def mu(k):
    aux = .5 * k
    return np.sqrt(2) * gamma(aux + .5) / gamma(aux)


def fit_chi(mean, var, kmax=100, step=1):

    # Compute mean of squares
    meansq = var + mean ** 2

    # Optimize k using a quasi-exhaustive search
    k_opt, s_opt, std_opt, score_opt = 0, 0, 0, np.inf
    for k in np.arange(1, kmax, step):
        mu_k = mu(k)
        s = np.sqrt(meansq / k)
        std = s * np.sqrt(k - mu_k ** 2)
        score = np.abs(mean - s * mu_k)
        if score < score_opt:
            k_opt, s_opt, std_opt, score_opt = k, s, std, score
    print k_opt, s_opt, std_opt, score_opt
    return k_opt, s_opt, std_opt


def simulate_noise(k, s, n):
    if k == np.inf:
        return np.abs(np.random.normal(scale=s, size=(n,)))
    x = s * np.random.normal(size=(k, n))
    return np.sqrt((x ** 2).sum(0))


# Simulate noise
k = 10
s = 1
n = 30000
x = simulate_noise(k, s, n)

# Compute mean and variances
mean, var = np.mean(x), np.var(x)

# Fit chi distribution
k, s, std = fit_chi(mean, var)

print('Noise std dev estimate: %f' % std)
