"""
Implements sMIL
"""
from __future__ import print_function, division
import numpy as np

from misvm.quadprog import quadprog
from misvm.kernel import by_name as kernel_by_name
from misvm.util import BagSplitter
from misvm.nsk import NSK


class sMIL(NSK):
    """
    Sparse MIL (Bunescu & Mooney, 2007)
    """

    def __init__(self, **kwargs):
        """
        @param kernel : the desired kernel function; can be linear, quadratic,
                        polynomial, or rbf [default: linear]
                        (by default, no normalization is used; to use averaging
                        or feature space normalization, append either '_av' or
                        '_fs' to the kernel name, as in 'rbf_av'; averaging
                        normalization is used in the original formulation)
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
        super(sMIL, self).__init__(**kwargs)

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        bs = BagSplitter(list(map(np.asmatrix, bags)),
                         np.asmatrix(y).reshape((-1, 1)))
        self._bags = bs.neg_inst_as_bags + bs.pos_bags
        self._y = np.matrix(np.vstack([-np.ones((bs.L_n, 1)),
                                       np.ones((bs.X_p, 1))]))
        if self.scale_C:
            iC = float(self.C) / bs.L_n
            bC = float(self.C) / bs.X_p
        else:
            iC = self.C
            bC = self.C
        C = np.vstack([iC * np.ones((bs.L_n, 1)),
                       bC * np.ones((bs.X_p, 1))])

        if self.verbose:
            print('Setup QP...')
        K, H, f, A, b, lb, ub = self._setup_svm(self._bags, self._y, C)

        # Adjust f with balancing terms
        factors = np.vstack([np.matrix(np.ones((bs.L_n, 1))),
                             np.matrix([2.0 / len(bag) - 1.0
                                        for bag in bs.pos_bags]).T])
        f = np.multiply(f, factors)

        if self.verbose:
            print('Solving QP...')
        self._alphas, self._objective = quadprog(H, f, A, b, lb, ub,
                                                 self.verbose)
        self._compute_separator(K)

        # Recompute predictions for full bags
        self._bag_predictions = self.predict(bags)