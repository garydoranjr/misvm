"""
Implements sbMIL
"""
import numpy as np

from smil import sMIL
from sil import SIL
from util import BagSplitter

class sbMIL(SIL):

    def __init__(self, *args, **kwargs):
        self.eta = kwargs.pop('eta', 0.0)
        self.eta = max(0.0, self.eta)
        self.eta = min(1.0, self.eta)
        super(sbMIL, self).__init__(*args, **kwargs)

    def fit(self, bags, y):
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.asmatrix(y).reshape((-1, 1))
        bs = BagSplitter(self._bags, y)

        if self.verbose: print 'Training initial sMIL classifier for sbMIL...'
        initial_classifier = sMIL(kernel=self.kernel, C=self.C, p=self.p, gamma=self.gamma,
                                  scale_C=self.scale_C, verbose=self.verbose,
                                  sv_cutoff=self.sv_cutoff)
        initial_classifier.fit(bags, y)
        if self.verbose: print 'Computing initial instance labels for sbMIL...'
        f_pos = initial_classifier.predict(bs.pos_inst_as_bags)
        # Select nth largest value as cutoff for positive instances
        n = int(round(bs.L_p*self.eta))
        n = min(bs.L_p, n)
        n = max(bs.X_p, n)
        f_cutoff = sorted((float(f) for f in f_pos), reverse=True)[n - 1]

        # Label all except for n largest as -1
        pos_labels = -np.matrix(np.ones((bs.L_p, 1)))
        pos_labels[np.nonzero(f_pos >= f_cutoff)] = 1.0

        # Train on all instances
        if self.verbose:
            print 'Retraining with top %d%% as positive...' % int(100*self.eta)
        all_labels = np.vstack([-np.ones((bs.L_n, 1)), pos_labels])
        super(SIL, self).fit(bs.instances, all_labels)

    def _compute_separator(self, K):
        super(SIL, self)._compute_separator(K)
        self._bag_predictions = self.predict(self._bags)
