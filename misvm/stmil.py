"""
Implements stMIL
"""
from __future__ import print_function, division
import numpy as np
from random import uniform

from misvm.nsk import NSK
from misvm.smil import sMIL
from misvm.quadprog import IterativeQP
from misvm.cccp import CCCP
from misvm.util import BagSplitter, spdiag


class stMIL(NSK):
    """
    Sparse, transductive MIL (Bunescu & Mooney, 2007)
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
        """
        self.restarts = kwargs.pop('restarts', 0)
        self.max_iters = kwargs.pop('max_iters', 50)
        super(stMIL, self).__init__(*args, **kwargs)

    def fit(self, bags, y):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
        """
        self._bags = map(np.asmatrix, bags)
        bs = BagSplitter(self._bags,
                         np.asmatrix(y).reshape((-1, 1)))
        self._all_bags = bs.neg_inst_as_bags + bs.pos_inst_as_bags + bs.pos_bags
        all_classes = np.vstack([-np.ones((bs.L_n, 1)),
                                 np.ones((bs.L_p + bs.X_p, 1))])

        if self.scale_C:
            niC = float(self.C) / bs.L_n
            piC = float(self.C) / bs.L_p
            pbC = float(self.C) / bs.X_p
        else:
            niC = float(self.C)
            piC = float(self.C)
            pbC = float(self.C)
        C = np.vstack([niC * np.ones((bs.L_n, 1)),
                       piC * np.ones((bs.L_p, 1)),
                       pbC * np.ones((bs.X_p, 1))])

        # Used to adjust balancing terms
        factors = np.vstack([np.matrix(np.ones((bs.L_n + bs.L_p, 1))),
                             np.matrix([2.0 / bag.shape[0] - 1.0
                                        for bag in bs.pos_bags]).T])

        best_obj = float('inf')
        best_svm = None
        for rr in range(self.restarts + 1):
            if rr == 0:
                if self.verbose:
                    print('Non-random start...')
                if self.verbose:
                    print('Initial sMIL solution...')
                smil = sMIL(kernel=self.kernel, C=self.C,
                            gamma=self.gamma, p=self.p, scale_C=self.scale_C)
                smil.fit(bags, y)
                if self.verbose:
                    print('Computing instance classes...')
                initial_svm = smil
                initial_classes = np.sign(smil.predict(bs.pos_inst_as_bags))
            else:
                if self.verbose:
                    print('Random restart %d of %d...' % (rr, self.restarts))
                initial_svm = None
                initial_classes = np.matrix([np.sign([uniform(-1.0, 1.0)
                                                      for i in range(bs.L_p)])]).T

            if self.verbose:
                print('Setup SVM and QP...')
            # Setup SVM and QP
            K, H, f, A, b, lb, ub = self._setup_svm(self._all_bags, all_classes, C)
            # Adjust f with balancing terms
            f = np.multiply(f, factors)
            qp = IterativeQP(H, f, A, b, lb, ub)

            class stMILCCCP(CCCP):

                def bailout(cself, svm, obj_val, classes):
                    return svm

                def iterate(cself, svm, obj_val, classes):
                    # Fix classes with zero classification
                    classes[np.nonzero(classes == 0.0)] = 1.0

                    cself.mention('Linearalizing constraints...')
                    all_classes = np.matrix(np.vstack([-np.ones((bs.L_n, 1)),
                                                       classes.reshape((-1, 1)),
                                                       np.ones((bs.X_p, 1))]))
                    D = spdiag(all_classes)

                    # Update QP
                    qp.update_H(D * K * D)
                    qp.update_Aeq(all_classes.T)

                    # Solve QP
                    alphas, obj = qp.solve(self.verbose)

                    # Update SVM
                    svm = NSK(kernel=self.kernel, gamma=self.gamma, p=self.p,
                              verbose=self.verbose, sv_cutoff=self.sv_cutoff)
                    svm._bags = self._all_bags
                    svm._y = all_classes
                    svm._alphas = alphas
                    svm._objective = obj
                    svm._compute_separator(K)
                    svm._K = K

                    if cself.check_tolerance(obj_val, obj):
                        return None, svm

                    # Use precomputed classifications from SVM
                    new_classes = np.sign(svm._bag_predictions[bs.L_n:-bs.X_p])
                    return {'svm': svm, 'obj_val': obj, 'classes': new_classes}, None

            cccp = stMILCCCP(verbose=self.verbose, svm=initial_svm, obj_val=None,
                             classes=initial_classes, max_iters=self.max_iters)
            svm = cccp.solve()
            if svm is not None:
                obj = float(svm._objective)
                if obj < best_obj:
                    best_svm = svm
                    best_obj = obj

        if best_svm is not None:
            self._all_bags = best_svm._bags
            self._y = best_svm._y
            self._alphas = best_svm._alphas
            self._objective = best_svm._objective
            self._compute_separator(best_svm._K)

    def _compute_separator(self, K):
        bags = self._bags
        self._bags = self._all_bags
        super(stMIL, self)._compute_separator(K)
        self._bags = bags
        self._bag_predictions = self.predict(self._bags)
