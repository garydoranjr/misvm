"""
Implements the STK of Gartner et al.
"""
import numpy as np

from svm import SVM

class STK(SVM):

    def __init__(self, *args, **kwargs):
        super(STK, self).__init__(*args, **kwargs)
        self._bags = None
        self._bag_predictions = None

    def fit(self, bags, y):
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.asmatrix(y).reshape((-1, 1))
        svm_X = _stats_from_bags(bags)
        super(STK, self).fit(svm_X, y)

    def _compute_separator(self, K):
        super(STK, self)._compute_separator(K)
        self._bag_predictions = self._predictions

    def predict(self, bags):
        bags = [np.asmatrix(bag) for bag in bags]
        svm_X = _stats_from_bags(bags)
        return super(STK, self).predict(svm_X)

def _stats_from_bags(bags):
    return np.vstack([np.hstack([np.min(bag, 0), np.max(bag, 0)])
                      for bag in bags])
