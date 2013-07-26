"""
Implements the Normalized Set Kernel
of Gartner et al.
"""
import numpy as np

from quadprog import quadprog
from kernel import by_name as kernel_by_name
from util import spdiag
from svm import SVM

class NSK(SVM):

    def __init__(self, *args, **kwargs):
        super(NSK, self).__init__(*args, **kwargs)
        self._bags = None
        self._sv_bags = None
        self._bag_predictions = None

    def fit(self, bags, y):
        self._bags = map(np.asmatrix, bags)
        self._y = np.asmatrix(y).reshape((-1, 1))
        if self.scale_C:
            C = self.C / float(len(self._bags))
        else:
            C = self.C

        if self.verbose: print 'Setup QP...'
        K, H, f, A, b, lb, ub = self._setup_svm(self._bags, self._y, C)

        # Solve QP
        if self.verbose: print 'Solving QP...'
        self._alphas, self._objective = quadprog(H, f, A, b, lb, ub,
                                                 self.verbose)
        self._compute_separator(K)

    def _compute_separator(self, K):

        self._sv = np.nonzero(self._alphas.flat > self.sv_cutoff)
        self._sv_alphas = self._alphas[self._sv]
        self._sv_bags = [self._bags[i] for i in self._sv[0]]
        self._sv_y = self._y[self._sv]

        n = len(self._sv_bags)
        if n == 0:
            self._b = 0.0
            self._bag_predictions = np.zeros(len(self._bags))
        else:
            _sv_all_K = K[self._sv]
            _sv_K = _sv_all_K.T[self._sv].T
            e = np.matrix(np.ones((n, 1)))
            D = spdiag(self._sv_y)
            self._b = float(e.T*D*e - self._sv_alphas.T*D*_sv_K*e) / n
            self._bag_predictions = np.array(self._b
                + self._sv_alphas.T*D*_sv_all_K).reshape((-1,))

    def predict(self, bags):
        if self._sv_bags is None or len(self._sv_bags) == 0:
            return np.zeros(len(bags))
        else:
            kernel = kernel_by_name(self.kernel, p=self.p, gamma=self.gamma)
            K = kernel(map(np.asmatrix, bags), self._sv_bags)
            return np.array(self._b + K*spdiag(self._sv_y)*self._sv_alphas).reshape((-1,))
