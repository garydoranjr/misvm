"""
Implements MissSVM
"""
from __future__ import print_function, division
import numpy as np
import scipy.sparse as sp
from random import uniform
import inspect
from misvm.quadprog import IterativeQP, Objective
from misvm.util import BagSplitter, spdiag, slices
from misvm.kernel import by_name as kernel_by_name
from misvm.mica import MICA
from misvm.cccp import CCCP


class MissSVM(MICA):
    """
    Semi-supervised learning applied to MI data (Zhou & Xu 2007)
    """

    def __init__(self, alpha=1e4, **kwargs):
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
        @param restarts : the number of random restarts [default: 0]
        @param max_iters : the maximum number of iterations in the outer loop of
                           the optimization procedure [default: 50]
        @param alpha : the softmax parameter [default: 1e4]
        """
        self.alpha = alpha
        super(MissSVM, self).__init__(**kwargs)
        self._bags = None
        self._sv_bags = None
        self._bag_predictions = None

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        self._bags = list(map(np.asmatrix, bags))
        bs = BagSplitter(self._bags,
                         np.asmatrix(y).reshape((-1, 1)))
        self._X = np.vstack([bs.pos_instances,
                             bs.pos_instances,
                             bs.pos_instances,
                             bs.neg_instances])
        self._y = np.vstack([np.matrix(np.ones((bs.X_p + bs.L_p, 1))),
                             -np.matrix(np.ones((bs.L_p + bs.L_n, 1)))])
        if self.scale_C:
            C = self.C / float(len(self._bags))
        else:
            C = self.C

        # Setup SVM and adjust constraints
        _, _, f, A, b, lb, ub = self._setup_svm(self._y, self._y, C)
        ub[:bs.X_p] *= (float(bs.L_n) / float(bs.X_p))
        ub[bs.X_p: bs.X_p + 2 * bs.L_p] *= (float(bs.L_n) / float(bs.L_p))
        K = kernel_by_name(self.kernel, gamma=self.gamma, p=self.p)(self._X, self._X)
        D = spdiag(self._y)
        ub0 = np.matrix(ub)
        ub0[bs.X_p: bs.X_p + 2 * bs.L_p] *= 0.5

        def get_V(pos_classifications):
            eye_n = bs.L_n + 2 * bs.L_p
            top = np.zeros((bs.X_p, bs.L_p))
            for row, (i, j) in enumerate(slices(bs.pos_groups)):
                top[row, i:j] = _grad_softmin(-pos_classifications[i:j], self.alpha).flat
            return sp.bmat([[sp.coo_matrix(top), None],
                            [None, sp.eye(eye_n, eye_n)]])

        V0 = get_V(np.matrix(np.zeros((bs.L_p, 1))))

        qp = IterativeQP(D * V0 * K * V0.T * D, f, A, b, lb, ub0)

        best_obj = float('inf')
        best_svm = None
        for rr in range(self.restarts + 1):
            if rr == 0:
                if self.verbose:
                    print('Non-random start...')
                # Train on instances
                alphas, obj = qp.solve(self.verbose)
            else:
                if self.verbose:
                    print('Random restart %d of %d...' % (rr, self.restarts))
                alphas = np.matrix([uniform(0.0, 1.0) for i in range(len(lb))]).T
                obj = Objective(0.0, 0.0)
            svm = MICA(kernel=self.kernel, gamma=self.gamma, p=self.p,
                       verbose=self.verbose, sv_cutoff=self.sv_cutoff)
            svm._X = self._X
            svm._y = self._y
            svm._V = V0
            svm._alphas = alphas
            svm._objective = obj
            svm._compute_separator(K)
            svm._K = K

            class missCCCP(CCCP):

                def bailout(cself, svm, obj_val):
                    return svm

                def iterate(cself, svm, obj_val):
                    cself.mention('Linearizing constraints...')
                    classifications = svm._predictions[bs.X_p: bs.X_p + bs.L_p]
                    V = get_V(classifications)

                    cself.mention('Computing slacks...')
                    # Difference is [1 - y_i*(w*phi(x_i) + b)]
                    pos_differences = 1.0 - classifications
                    neg_differences = 1.0 + classifications
                    # Slacks are positive differences only
                    pos_slacks = np.multiply(pos_differences > 0, pos_differences)
                    neg_slacks = np.multiply(neg_differences > 0, neg_differences)
                    all_slacks = np.hstack([pos_slacks, neg_slacks])

                    cself.mention('Linearizing...')
                    # Compute gradient across pairs
                    slack_grads = np.vstack([_grad_softmin(pair, self.alpha)
                                             for pair in all_slacks])
                    # Stack results into one column
                    slack_grads = np.vstack([np.ones((bs.X_p, 1)),
                                             slack_grads[:, 0],
                                             slack_grads[:, 1],
                                             np.ones((bs.L_n, 1))])
                    # Update QP
                    qp.update_H(D * V * K * V.T * D)
                    qp.update_ub(np.multiply(ub, slack_grads))

                    # Re-solve
                    cself.mention('Solving QP...')
                    alphas, obj = qp.solve(self.verbose)
                    new_svm = MICA(kernel=self.kernel, gamma=self.gamma, p=self.p,
                                   verbose=self.verbose, sv_cutoff=self.sv_cutoff)
                    new_svm._X = self._X
                    new_svm._y = self._y
                    new_svm._V = V
                    new_svm._alphas = alphas
                    new_svm._objective = obj
                    new_svm._compute_separator(K)
                    new_svm._K = K

                    if cself.check_tolerance(obj_val, obj):
                        return None, new_svm

                    return {'svm': new_svm, 'obj_val': obj}, None

            cccp = missCCCP(verbose=self.verbose, svm=svm, obj_val=None,
                            max_iters=self.max_iters)
            svm = cccp.solve()
            if svm is not None:
                obj = float(svm._objective)
                if obj < best_obj:
                    best_svm = svm
                    best_obj = obj

        if best_svm is not None:
            self._V = best_svm._V
            self._alphas = best_svm._alphas
            self._objective = best_svm._objective
            self._compute_separator(best_svm._K)
            self._bag_predictions = self.predict(self._bags)

    def get_params(self, deep=True):
        super_args = super(MissSVM, self).get_params()
        args, _, _, _ = inspect.getargspec(MissSVM.__init__)
        args.pop(0)
        super_args.update({key: getattr(self, key, None) for key in args})
        return super_args


def _grad_softmin(x, alpha=1e4):
    """
    Computes the gradient of min function,
    taken from gradient of softmin as
    alpha goes to infinity. It is:
    0   if x_i != min(x), or
    1/n if x_i is one of the n
        elements equal to min(x)
    """
    grad = np.matrix(np.zeros(x.shape))
    minimizers = (x == min(x.flat))
    n = float(np.sum(minimizers))
    grad[np.nonzero(minimizers)] = 1.0 / n
    return grad
