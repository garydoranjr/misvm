"""
Implements Single Instance Learning SVM
"""
import numpy as np

from svm import SVM
from util import slices

class SIL(SVM):

    def __init__(self, *args, **kwargs):
        super(SIL, self).__init__(*args, **kwargs)
        self._bags = None
        self._bag_predictions = None

    def fit(self, bags, y):
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.asmatrix(y).reshape((-1, 1))
        svm_X = np.vstack(self._bags)
        svm_y = np.vstack([float(cls)*np.matrix(np.ones((len(bag), 1)))
                           for bag, cls in zip(self._bags, y)])
        super(SIL, self).fit(svm_X, svm_y)

    def _compute_separator(self, K):
        super(SIL, self)._compute_separator(K)
        self._bag_predictions = _inst_to_bag_preds(self._predictions, self._bags)

    def predict(self, bags):
        bags = [np.asmatrix(bag) for bag in bags]
        inst_preds = super(SIL, self).predict(np.vstack(bags))
        return _inst_to_bag_preds(inst_preds, bags)

def _inst_to_bag_preds(inst_preds, bags):
    return np.array([np.max(inst_preds[slice(*bidx)])
                     for bidx in slices(map(len, bags))])
