"""
Implements sbMIL
"""
from __future__ import print_function, division
import numpy as np

from misvm.smil import sMIL
from misvm.sil import SIL
from misvm.util import BagSplitter


class sbMIL(SIL):
    """
    Sparse, balanced MIL (Bunescu & Mooney, 2007)
    """

    def __init__(self, *args, **kwargs):
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
        @param eta : balance parameter
        """
        self.eta = kwargs.pop('eta', 0.0)
        self.eta = max(0.0, self.eta)
        self.eta = min(1.0, self.eta)
        super(sbMIL, self).__init__(*args, **kwargs)

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.asmatrix(y).reshape((-1, 1))
        bs = BagSplitter(self._bags, y)

        if self.verbose:
            print('Training initial sMIL classifier for sbMIL...')
        initial_classifier = sMIL(kernel=self.kernel, C=self.C, p=self.p, gamma=self.gamma,
                                  scale_C=self.scale_C, verbose=self.verbose,
                                  sv_cutoff=self.sv_cutoff)
        initial_classifier.fit(bags, y)
        if self.verbose:
            print('Computing initial instance labels for sbMIL...')
        f_pos = initial_classifier.predict(bs.pos_inst_as_bags)
        # Select nth largest value as cutoff for positive instances
        n = int(round(bs.L_p * self.eta))
        n = min(bs.L_p, n)
        n = max(bs.X_p, n)
        f_cutoff = sorted((float(f) for f in f_pos), reverse=True)[n - 1]

        # Label all except for n largest as -1
        pos_labels = -np.matrix(np.ones((bs.L_p, 1)))
        pos_labels[np.nonzero(f_pos >= f_cutoff)] = 1.0

        # Train on all instances
        if self.verbose:
            print('Retraining with top %d%% as positive...' % int(100 * self.eta))
        all_labels = np.vstack([-np.ones((bs.L_n, 1)), pos_labels])
        super(SIL, self).fit(bs.instances, all_labels)

    def _compute_separator(self, K):
        super(SIL, self)._compute_separator(K)
        self._bag_predictions = self.predict(self._bags)
