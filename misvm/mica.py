"""
Implements the MICA algorithm
"""
from __future__ import print_function, division
import sys
import numpy as np
import scipy.sparse as sp
from cvxopt import matrix as cvxmat, sparse, spmatrix
from cvxopt.solvers import lp
import inspect
from misvm.quadprog import IterativeQP, spzeros as spz, speye as spI, _apply_options
from misvm.util import spdiag, BagSplitter, slices, rand_convex
from misvm.kernel import by_name as kernel_by_name
from misvm.svm import SVM
from misvm.cccp import CCCP


class MICA(SVM):
    """
    The MICA approach of Mangasarian & Wild (2008)
    """

    def __init__(self, regularization='L2', restarts=0, max_iters=50, **kwargs):
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
        @param regularization : currently only L2 regularization is implemented
        """
        self.regularization = regularization
        if not self.regularization in ('L2',):
            raise ValueError('Invalid regularization "%s"'
                             % self.regularization)
        self.restarts = restarts
        self.max_iters = max_iters
        super(MICA, self).__init__(**kwargs)
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
        self._X = bs.instances
        Ln = bs.L_n
        Lp = bs.L_p
        Xp = bs.X_p
        m = Ln + Xp
        if self.scale_C:
            C = self.C / float(len(self._bags))
        else:
            C = self.C

        K = kernel_by_name(self.kernel, gamma=self.gamma, p=self.p)(self._X, self._X)
        new_classes = np.matrix(np.vstack([-np.ones((Ln, 1)),
                                           np.ones((Xp, 1))]))
        self._y = new_classes
        D = spdiag(new_classes)
        setup = list(self._setup_svm(new_classes, new_classes, C))[1:]
        setup[0] = np.matrix([0])
        qp = IterativeQP(*setup)

        c = cvxmat(np.hstack([np.zeros(Lp + 1),
                              np.ones(Xp + Ln)]))
        b = cvxmat(np.ones((Xp, 1)))
        A = spz(Xp, Lp + 1 + Xp + Ln)
        for row, (i, j) in enumerate(slices(bs.pos_groups)):
            A[row, i:j] = 1.0

        bottom_left = sparse(t([[-spI(Lp), spz(Lp)],
                                [spz(m, Lp), spz(m)]]))
        bottom_right = sparse([spz(Lp, m), -spI(m)])
        inst_cons = sparse(t([[spz(Xp, Lp), -spo(Xp)],
                              [spz(Ln, Lp), spo(Ln)]]))
        G = sparse(t([[inst_cons, -spI(m)],
                      [bottom_left, bottom_right]]))
        h = cvxmat(np.vstack([-np.ones((Xp, 1)),
                              np.zeros((Ln + Lp + m, 1))]))

        def to_V(upsilon):
            bot = np.zeros((Xp, Lp))
            for row, (i, j) in enumerate(slices(bs.pos_groups)):
                bot[row, i:j] = upsilon.flat[i:j]
            return sp.bmat([[sp.eye(Ln, Ln), None],
                            [None, sp.coo_matrix(bot)]])

        class MICACCCP(CCCP):

            def bailout(cself, alphas, upsilon, svm):
                return svm

            def iterate(cself, alphas, upsilon, svm):
                V = to_V(upsilon)
                cself.mention('Update QP...')
                qp.update_H(D * V * K * V.T * D)
                cself.mention('Solve QP...')
                alphas, obj = qp.solve(self.verbose)
                svm = MICA(kernel=self.kernel, gamma=self.gamma, p=self.p,
                           verbose=self.verbose, sv_cutoff=self.sv_cutoff)
                svm._X = self._X
                svm._y = self._y
                svm._V = V
                svm._alphas = alphas
                svm._objective = obj
                svm._compute_separator(K)
                svm._K = K

                cself.mention('Update LP...')
                for row, (i, j) in enumerate(slices(bs.pos_groups)):
                    G[row, i:j] = cvxmat(-svm._dotprods[Ln + i: Ln + j].T)
                h[Xp: Xp + Ln] = cvxmat(-(1 + svm._dotprods[:Ln]))

                cself.mention('Solve LP...')
                sol, _ = linprog(c, G, h, A, b, verbose=self.verbose)
                new_upsilon = sol[:Lp]

                if cself.check_tolerance(np.linalg.norm(upsilon - new_upsilon)):
                    return None, svm

                return {'alphas': alphas, 'upsilon': new_upsilon, 'svm': svm}, None

        best_obj = float('inf')
        best_svm = None
        for rr in range(self.restarts + 1):
            if rr == 0:
                if self.verbose:
                    print('Non-random start...')
                upsilon0 = np.matrix(np.vstack([np.ones((size, 1)) / float(size)
                                                for size in bs.pos_groups]))
            else:
                if self.verbose:
                    print('Random restart %d of %d...' % (rr, self.restarts))
                upsilon0 = np.matrix(np.vstack([rand_convex(size).T
                                                for size in bs.pos_groups]))
            cccp = MICACCCP(verbose=self.verbose, alphas=None, upsilon=upsilon0,
                            svm=None, max_iters=self.max_iters)
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

    def _compute_separator(self, K):
        sv = (self._alphas.flat > self.sv_cutoff)

        D = spdiag(self._y)
        self._b = (np.sum(D * sv) - np.sum(self._alphas.T * D * self._V * K)) / np.sum(sv)
        self._dotprods = (self._alphas.T * D * self._V * K).T
        self._predictions = self._b + self._dotprods

    def predict(self, bags):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @return : an array of length n containing real-valued label predictions
                  (threshold at zero to produce binary predictions)
        """
        if self._b is None:
            return np.zeros(len(bags))
        else:
            bags = [np.asmatrix(bag) for bag in bags]
            k = kernel_by_name(self.kernel, p=self.p, gamma=self.gamma)
            D = spdiag(self._y)
            return np.array([np.max(self._b + self._alphas.T * D * self._V *
                                    k(self._X, bag))
                             for bag in bags])

    def get_params(self, deep=True):
        """
        return params
        """
        super_args, _, _, _ = inspect.getargspec(super(MICA, self).__init__)
        args, _, _, _ = inspect.getargspec(MICA.__init__)
        args.pop(0)
        super_args.pop(0)
        args += super_args
        return {key: getattr(self, key, None) for key in args}


def linprog(*args, **kwargs):
    verbose = kwargs.get('verbose', False)
    # Save settings and set verbosity
    old_settings = _apply_options({'show_progress': verbose})

    # Optimize
    results = lp(*args, solver='glpk')

    # Restore settings
    _apply_options(old_settings)

    # Check return status
    status = results['status']
    if not status == 'optimal':
        print('Warning: termination of lp with status: %s'
              % status, file=sys.stderr)

    # Convert back to NumPy matrix
    # and return solution
    xstar = results['x']
    return np.matrix(xstar), results['primal objective']


def spo(r, v=1.0):
    """Create a sparse one vector"""
    return spmatrix(v, range(r), r * [0])


def t(list_of_lists):
    """
    Transpose a list of lists, since 'sparse'
    takes arguments in column-major order.
    """
    return list(map(list, zip(*list_of_lists)))
