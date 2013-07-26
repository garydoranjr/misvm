"""
Implements sMIL
"""
import numpy as np

from quadprog import quadprog
from kernel import by_name as kernel_by_name
from util import BagSplitter
from nsk import NSK

class sMIL(NSK):

    def __init__(self, *args, **kwargs):
        super(sMIL, self).__init__(*args, **kwargs)

    def fit(self, bags, y):
        bs = BagSplitter(map(np.asmatrix, bags),
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
        C = np.vstack([iC*np.ones((bs.L_n, 1)),
                       bC*np.ones((bs.X_p, 1))])

        if self.verbose: print 'Setup QP...'
        K, H, f, A, b, lb, ub = self._setup_svm(self._bags, self._y, C)

        # Adjust f with balancing terms
        factors = np.vstack([np.matrix(np.ones((bs.L_n, 1))),
                             np.matrix([2.0/len(bag) - 1.0
                                  for bag in bs.pos_bags]).T])
        f = np.multiply(f, factors)

        if self.verbose: print 'Solving QP...'
        self._alphas, self._objective = quadprog(H, f, A, b, lb, ub,
                                                 self.verbose)
        self._compute_separator(K)

        # Recompute predictions for full bags
        self._bag_predictions = self.predict(bags)
