"""
Contains various kernels for SVMs

A kernel should take two arguments,
each of which is a list of examples
as rows of a numpy matrix
"""
from __future__ import print_function, division
from numpy import matrix, vstack, hstack
import numpy as np
from scipy.spatial.distance import cdist
from scipy.io import loadmat, savemat
import math
import os

import hashlib
import time

CACHE_CUTOFF_T = 10
CACHE_DIR = '.kernel_cache'

from misvm.util import spdiag, slices


def by_name(full_name, gamma=None, p=None, use_caching=False):
    parts = full_name.split('_')
    name = parts.pop(0)

    try:
        # See if second part is a number
        value = float(parts[0])
        parts.pop(0)
    except:
        pass

    if name == 'linear':
        kernel = linear
    elif name == 'quadratic':
        kernel = quadratic
    elif name == 'polynomial':
        kernel = polynomial(int(p))
    elif name == 'rbf':
        kernel = rbf(gamma)
    else:
        raise ValueError('Unknown Kernel type %s' % name)

    try:
        # See if remaining part is a norm
        norm_name = parts.pop(0)
        if norm_name == 'fs':
            norm = featurespace_norm
        elif norm_name == 'av':
            norm = averaging_norm
        else:
            raise ValueError('Unknown norm %s' % norm_name)
    except IndexError:
        norm = no_norm

    kernel_function = set_kernel(kernel, norm)
    kernel_function.name = full_name
    if use_caching:
        kernel_function = cached_kernel(kernel_function)
        kernel_function.name = full_name
    return kernel_function


def averaging_norm(x, *args):
    return float(x.shape[0])


def featurespace_norm(x, k):
    return math.sqrt(np.sum(k(x, x)))


def no_norm(x, k):
    return 1.0


def _hash_array(x):
    return hashlib.sha1(x).hexdigest()


def cached_kernel(K):
    def cached_K(X, Y):
        if type(X) == list:
            x_hash = ''.join(map(_hash_array, X))
            y_hash = ''.join(map(_hash_array, Y))
        else:
            x_hash = _hash_array(X)
            y_hash = _hash_array(Y)
        full_hash = hashlib.sha1(x_hash + y_hash + K.name).hexdigest()
        cache_file = os.path.join(CACHE_DIR, full_hash + '.mat')
        if os.path.exists(cache_file):
            print('Using cached result!')
            result = np.matrix(loadmat(cache_file)['k'])
            return result
        # Check cache
        t0 = time.time()
        result = K(X, Y)
        tf = time.time()
        if (tf - t0) > CACHE_CUTOFF_T:
            print('Caching result...')
            if not os.path.exists(CACHE_DIR):
                os.mkdir(CACHE_DIR)
            savemat(cache_file, {'k': result}, oned_as='column')
        return result

    return cached_K


def set_kernel(k, normalizer=no_norm):
    """
    Decorator that makes a normalized
    set kernel out of a standard kernel k
    """

    def K(X, Y):
        if type(X) == list:
            norm = lambda x: normalizer(x, k)
            x_norm = matrix(list(map(norm, X)))
            if id(X) == id(Y):
                # Optimize for symmetric case
                norms = x_norm.T * x_norm
                if all(len(bag) == 1 for bag in X):
                    # Optimize for singleton bags
                    instX = vstack(X)
                    raw_kernel = k(instX, instX)
                else:
                    # Only need to compute half of
                    # the matrix if it's symmetric
                    upper = matrix([i * [0] + [np.sum(k(x, y))
                                               for y in Y[i:]]
                                    for i, x in enumerate(X, 1)])
                    diag = np.array([np.sum(k(x, x)) for x in X])
                    raw_kernel = upper + upper.T + spdiag(diag)
            else:
                y_norm = matrix(list(map(norm, Y)))
                norms = x_norm.T * y_norm
                raw_kernel = k(vstack(X), vstack(Y))
                lensX = list(map(len, X))
                lensY = list(map(len, Y))
                if any(l != 1 for l in lensX):
                    raw_kernel = vstack([np.sum(raw_kernel[i:j, :], axis=0)
                                         for i, j in slices(lensX)])
                if any(l != 1 for l in lensY):
                    raw_kernel = hstack([np.sum(raw_kernel[:, i:j], axis=1)
                                         for i, j in slices(lensY)])
            return np.divide(raw_kernel, norms)
        else:
            return k(X, Y)

    return K


def linear(x, y):
    """Linear kernel x'*y"""
    return x * y.T


def quadratic(x, y):
    """Quadratic kernel (1 + x'*y)^2"""
    return np.square(1e0 + x * y.T)


def polynomial(p):
    """General polynomial kernel (1 + x'*y)^p"""

    def p_kernel(x, y):
        return np.power(1e0 + x * y.T, p)

    return p_kernel


def rbf(gamma):
    """Radial Basis Function"""

    def rbf_kernel(x, y):
        return matrix(np.exp(-gamma * cdist(x, y, 'sqeuclidean')))

    return rbf_kernel
