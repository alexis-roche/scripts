from nipy.labs.glm import glm

import sys
import numpy as np
from scipy.stats import norm as ssnorm
from pylab import *

INPUT_FILE = 'newhippo.npz'
#INPUT_FILE = 'cheap_hippo_5class.npz'
SAVEFIG = True
SPECIFICITY_MIN = 0.0
DETREND = False
FEATURES = 'gm', 'csf', 'glob_gm_csf'
FEATURES_FN = None, np.negative, None
# FEATURES = 'vol_true', 'pvol', 'pvol_csf', 'gm_csf'
# FEATURES_FN = None, None, np.negative, np.log


def make_design(age, tiv, sex):
    regressors = [age]
    if not tiv == None:
        regressors += [tiv]
    if not sex == None:
        regressors += [sex == ' F']
    X = np.zeros((len(age), len(regressors) + 1))
    X[:, 0] = 1
    X[:, 1:] = np.array(regressors).T
    return X


class LinearModel():
    """
    Estimate the influence of age, sex and tiv on hippocampus volume 
    """
    def __init__(self, vols, age, tiv=None, sex=None):
        self.X = make_design(age, tiv, sex)
        M = glm(vols, self.X)
        self.beta = M.beta
        self.s2 = M.s2
        self.vols = vols

    def _normalize(self, vols, X):
        return vols - np.dot(X[:,2:], self.beta[2:])

    def normalize(self):
        return self._normalize(self.vols, self.X) 

    def renormalize(self, vols, age, tiv=None, sex=None):
        return self._normalize(vols, make_design(age, tiv, sex))

    def predict(self, start, stop, num=100):
        a = np.linspace(start, stop, num) 
        return a, self.beta[0] + self.beta[1]*a

def closeall(): 
    for i in range(100):
        close() 

def pathoplot(y, x, patho, yname='dunno', xname='dunno'): 
    msk0 = np.where(patho==' Normal')
    msk = np.where(patho==' AD') 
    plot(x[msk0], y[msk0], 'ko')
    plot(x[msk], y[msk], 'ro')
    xlabel(xname, fontsize=16)
    ylabel(yname, fontsize=16)


def make_roc(y, age, patho, tiv, sex, label='dunno', detrend=DETREND):

    # fit a linear model on controls and normalize AD vols for sex and
    # tiv using the same model
    msk0 = np.where(patho == ' Normal')
    msk = np.where(patho == ' AD')

    # use either a linear model with tiv and sex as confounds or a
    # linear model wrt age (without confounds)
    if detrend:
        M = LinearModel(y[msk0], age[msk0], tiv[msk0], sex[msk0])
        y0 = M.normalize()
        y = M.renormalize(y[msk], age[msk], tiv[msk], sex[msk])
    else:
        y = y / tiv
        M = LinearModel(y[msk0], age[msk0])
        y0 = y[msk0]
        y = y[msk]

    # plot curves
    figure()
    a, nm = M.predict(55, 95)
    delta = np.sqrt(M.s2) * ssnorm.isf(.05)
    plot(a, nm, 'k')
    plot(a, nm + delta, 'k:')
    plot(a, nm - delta, 'k:')
    plot(age[msk0], y0, 'ok')
    plot(age[msk], y, 'or')
    xlabel('age', fontsize=16)
    ylabel(label, fontsize=16)

    # roc curves
    z = y
    zm = M.beta[0] + age[msk] * M.beta[1]
    #alphas = 10**(-np.linspace(1,10))
    alphas = np.linspace(0, 1 - SPECIFICITY_MIN, num=9999)
    betas = 0 * alphas
    for i in range(len(alphas)):
        alpha = alphas[i]
        delta = np.sqrt(M.s2) * ssnorm.isf(alpha)
        betas[i] = float(len(np.where(z < (zm - delta))[0])) / float(z.size)

    return alphas, betas


def get_sensitivity(a, b, spec=None):
    if spec == None:
        return b[len(np.where(1 - a - b > 0)[0])]
    else:
        return b[len(np.where(a < 1 - spec)[0])]


def _savefig(name):
    if SAVEFIG:
        savefig(name)


if len(sys.argv) < 2:
    f = np.load(INPUT_FILE)
else:
    f = np.load(sys.argv[1])

tiv = np.array([x['tiv'] for x in f['measures']]) / 1000.
patho = np.array([x['patho'] for x in f['info']])
age = np.array([x['age'] for x in f['info']])
sex = np.array([x['sex'] for x in f['info']])

a, b = {}, {}
for feat, fn in zip(FEATURES, FEATURES_FN):
    if feat == 'glob_gm_csf':
        dat = np.array([x['gm'] / x['csf'] for x in f['measures']])
    else:
        dat = np.array([x[feat] for x in f['measures']]) / 1000.
    if not fn == None:
        dat = fn(dat)
    a[feat], b[feat] = make_roc(dat, age, patho, tiv, sex, feat + ' (mL)')
    s = get_sensitivity(a[feat], b[feat])
    s80 = get_sensitivity(a[feat], b[feat], .8)
    s90 = get_sensitivity(a[feat], b[feat], .9)
    print(feat + ' sensitivity: %f, %f (80), %f (90)' % (s, s80, s90))
    _savefig('model_' + feat)


figure()
legend_ = []
for feat in FEATURES:
    plot(a[feat], b[feat], linewidth=2)
    legend_ += [feat]
title('ROC curves')
xlabel('False AD alarm rate', fontsize=16)
ylabel('True AD alarm rate', fontsize=16)
#axis([0.0, 0.1, 0.0, 1.])
legend(legend_)
_savefig('roc_curves')



"""
figure()
pathoplot(pvol, volt, patho,
          yname='estimated hippocampus volume (mL)',
          xname='ground truth hippocampus volume (mL)')
_savefig('correl_bin_hvol')
"""

