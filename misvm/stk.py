"""
Implements the STK of Gartner et al.
"""
from __future__ import print_function, division
import inspect
import numpy as np

from misvm.svm import SVM


class STK(SVM):
    """
    Statistics kernel of Gaertner, et al. (2002)
    """

    def __init__(self, **kwargs):
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
        super(STK, self).__init__(**kwargs)
        self._bags = None
        self._bag_predictions = None

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.asmatrix(y).reshape((-1, 1))
        svm_X = _stats_from_bags(bags)
        super(STK, self).fit(svm_X, y)

    def _compute_separator(self, K):
        super(STK, self)._compute_separator(K)
        self._bag_predictions = self._predictions

    def predict(self, bags):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @return : an array of length n containing real-valued label predictions
                  (threshold at zero to produce binary predictions)
        """
        bags = [np.asmatrix(bag) for bag in bags]
        svm_X = _stats_from_bags(bags)
        return super(STK, self).predict(svm_X)

    def get_params(self, deep=True):
        """
        return params
        """
        args, _, _, _ = inspect.getargspec(super(STK, self).__init__)
        args.pop(0)
        return {key: getattr(self, key, None) for key in args}

def _stats_from_bags(bags):
    return np.vstack([np.hstack([np.min(bag, 0), np.max(bag, 0)])
                      for bag in bags])
