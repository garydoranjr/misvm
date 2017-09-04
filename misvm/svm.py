"""
Implements a standard SVM
"""
from __future__ import print_function, division
import numpy as np
from misvm.quadprog import quadprog
from misvm.kernel import by_name as kernel_by_name
from misvm.util import spdiag
from sklearn.base import ClassifierMixin, BaseEstimator


class SVM(ClassifierMixin, BaseEstimator):
    """
    A standard supervised SVM implementation.
    """

    def __init__(self, kernel='linear', C=1.0, p=3, gamma=1e0, scale_C=True,
                 verbose=True, sv_cutoff=1e-7):
        """
        @param kernel : the desired kernel function; can be linear, quadratic,
                        polynomial, or rbf [default: linear]
        @param C : the loss/regularization tradeoff constant [default: 1.0]
        @param scale_C : if True [default], scale C by the number of examples
        @param p : polynomial degree when a 'polynomial' kernel is used
                   [default: 3]
        @param gamma : RBF scale parameter when an 'rbf' kernel is used
                      [default: 1.0]
        @param verbose : print optimization status messages [default: True]
        @param sv_cutoff : the numerical cutoff for an example to be considered
                           a support vector [default: 1e-7]
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.p = p
        self.scale_C = scale_C
        self.verbose = verbose
        self.sv_cutoff = sv_cutoff

        self._X = None
        self._y = None
        self._objective = None
        self._alphas = None
        self._sv = None
        self._sv_alphas = None
        self._sv_X = None
        self._sv_y = None
        self._b = None
        self._predictions = None

    def fit(self, X, y):
        """
        @param X : an n-by-m array-like object containing n examples with m
                   features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        self._X = np.asmatrix(X)
        self._y = np.asmatrix(y).reshape((-1, 1))
        if self.scale_C:
            C = self.C / float(len(self._X))
        else:
            C = self.C

        K, H, f, A, b, lb, ub = self._setup_svm(self._X, self._y, C)

        # Solve QP
        self._alphas, self._objective = quadprog(H, f, A, b, lb, ub,
                                                 self.verbose)
        self._compute_separator(K)

    def _compute_separator(self, K):

        self._sv = np.nonzero(self._alphas.flat > self.sv_cutoff)
        self._sv_alphas = self._alphas[self._sv]
        self._sv_X = self._X[self._sv]
        self._sv_y = self._y[self._sv]

        n = len(self._sv_X)
        if n == 0:
            self._b = 0.0
            self._predictions = np.zeros(len(self._X))
        else:
            _sv_all_K = K[self._sv]
            _sv_K = _sv_all_K.T[self._sv].T
            e = np.matrix(np.ones((n, 1)))
            D = spdiag(self._sv_y)
            self._b = float(e.T * D * e - self._sv_alphas.T * D * _sv_K * e) / n
            self._predictions = np.array(self._b
                                         + self._sv_alphas.T * D * _sv_all_K).reshape((-1,))

    def predict(self, X):
        """
        @param X : an n-by-m array-like object containing n examples with m
                   features
        @return : an array of length n containing real-valued label predictions
                  (threshold at zero to produce binary predictions)
        """
        if self._sv_X is None or len(self._sv_X) == 0:
            return np.zeros(len(X))
        else:
            kernel = kernel_by_name(self.kernel, p=self.p, gamma=self.gamma)
            K = kernel(np.asmatrix(X), self._sv_X)
            return np.array(self._b + K * spdiag(self._sv_y) * self._sv_alphas).reshape((-1,))

    def _setup_svm(self, examples, classes, C):
        kernel = kernel_by_name(self.kernel, gamma=self.gamma, p=self.p)
        n = len(examples)
        e = np.matrix(np.ones((n, 1)))

        # Kernel and Hessian
        if kernel is None:
            K = None
            H = None
        else:
            K = _smart_kernel(kernel, examples)
            D = spdiag(classes)
            H = D * K * D

        # Term for -sum of alphas
        f = -e

        # Sum(y_i * alpha_i) = 0
        A = classes.T.astype(float)
        b = np.matrix([0.0])

        # 0 <= alpha_i <= C
        lb = np.matrix(np.zeros((n, 1)))
        if type(C) == float:
            ub = C * e
        else:
            # Allow for C to be an array
            ub = C
        return K, H, f, A, b, lb, ub


def _smart_kernel(kernel, examples):
    """
    Optimize the case when instances are
    treated as singleton bags. In such
    cases, singleton bags should be placed
    at the beginning of the list of examples.
    """
    if type(examples) == list:
        for i, bag in enumerate(examples):
            if len(bag) > 1:
                break
        singletons, bags = examples[:i], examples[i:]
        if singletons and bags:
            ss = kernel(singletons, singletons)
            sb = kernel(singletons, bags)
            bb = kernel(bags, bags)
            return np.bmat([[ss, sb], [sb.T, bb]])

    return kernel(examples, examples)
