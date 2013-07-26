"""
Implements mi-SVM and MI-SVM
"""
import numpy as np
from random import uniform
from cvxopt import matrix as cvxmat, sparse

from sil import SIL
from svm import SVM
from cccp import CCCP
from quadprog import IterativeQP, spzeros, speye
from kernel import by_name as kernel_by_name
from util import partition, BagSplitter, spdiag, rand_convex, slices

class MISVM(SIL):

    def __init__(self, *args, **kwargs):
        self.restarts = kwargs.pop('restarts', 0)
        self.max_iters = kwargs.pop('max_iters', 50)
        super(MISVM, self).__init__(*args, **kwargs)

    def fit(self, bags, y):
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.asmatrix(y).reshape((-1, 1))

        bs = BagSplitter(self._bags, y)
        best_obj = float('inf')
        best_svm = None
        for rr in range(self.restarts + 1):
            if rr == 0:
                if self.verbose: print 'Non-random start...'
                pos_bag_avgs = np.vstack([np.average(bag, axis=0) for bag in bs.pos_bags])
            else:
                if self.verbose: print 'Random restart %d of %d...' % (rr, self.restarts)
                pos_bag_avgs = np.vstack([rand_convex(len(bag))*bag for bag in bs.pos_bags])

            intial_instances = np.vstack([bs.neg_instances, pos_bag_avgs])
            classes = np.vstack([-np.ones((bs.L_n, 1)),
                                  np.ones((bs.X_p, 1))])

            # Setup SVM and QP
            if self.scale_C:
                C = self.C / float(len(intial_instances))
            else:
                C = self.C
            setup = self._setup_svm(intial_instances, classes, C)
            K = setup[0]
            qp = IterativeQP(*setup[1:])

            # Fix Gx <= h
            neg_cons = spzeros(bs.X_n, bs.L_n)
            for b, (l, u) in enumerate(slices(bs.neg_groups)):
                neg_cons[b, l:u] = 1.0
            pos_cons = speye(bs.X_p)
            bot_left = spzeros(bs.X_p, bs.L_n)
            top_right = spzeros(bs.X_n, bs.X_p)
            half_cons = sparse([[neg_cons,  bot_left],
                                [top_right, pos_cons]])
            qp.G = sparse([-speye(bs.X_p + bs.L_n), half_cons])
            qp.h = cvxmat(np.vstack([ np.zeros((bs.X_p + bs.L_n, 1)),
                                     C*np.ones((bs.X_p + bs.X_n, 1))]))

            # Precompute kernel for all positive instances
            kernel = kernel_by_name(self.kernel, gamma=self.gamma, p=self.p)
            K_all = kernel(bs.instances, bs.instances)

            neg_selectors = np.array(range(bs.L_n))

            class MISVMCCCP(CCCP):

                def bailout(cself, svm, selectors, instances, K):
                    return svm

                def iterate(cself, svm, selectors, instances, K):
                    cself.mention('Training SVM...')
                    alphas, obj = qp.solve(cself.verbose)

                    # Construct SVM from solution
                    svm = SVM(kernel=self.kernel, gamma=self.gamma, p=self.p,
                              verbose=self.verbose, sv_cutoff=self.sv_cutoff)
                    svm._X = instances
                    svm._y = classes
                    svm._alphas = alphas
                    svm._objective = obj
                    svm._compute_separator(K)
                    svm._K = K

                    cself.mention('Recomputing classes...')
                    p_confs = svm.predict(bs.pos_instances)
                    pos_selectors = bs.L_n + np.array([l + np.argmax(p_confs[l:u])
                                                       for l, u in slices(bs.pos_groups)])
                    new_selectors = np.hstack([neg_selectors, pos_selectors])

                    if selectors is None:
                        sel_diff = len(new_selectors)
                    else:
                        sel_diff = np.nonzero(new_selectors - selectors)[0].size

                    cself.mention('Selector differences: %d' % sel_diff)
                    if sel_diff == 0:
                        return None, svm
                    elif sel_diff > 5:
                        # Clear results to avoid a
                        # bad starting point in
                        # the next iteration
                        qp.clear_results()

                    cself.mention('Updating QP...')
                    indices = (new_selectors,)
                    K = K_all[indices].T[indices].T
                    D = spdiag(classes)
                    qp.update_H(D*K*D)
                    return {'svm': svm, 'selectors': new_selectors,
                            'instances': bs.instances[indices], 'K': K}, None

            cccp = MISVMCCCP(verbose=self.verbose, svm=None, selectors=None,
                             instances=intial_instances, K=K, max_iters=self.max_iters)
            svm = cccp.solve()
            if svm is not None:
                obj = float(svm._objective)
                if obj < best_obj:
                    best_svm = svm
                    best_obj = obj

        if best_svm is not None:
            self._X = best_svm._X
            self._y = best_svm._y
            self._alphas = best_svm._alphas
            self._objective = best_svm._objective
            self._compute_separator(best_svm._K)

    def _compute_separator(self, K):
        super(SIL, self)._compute_separator(K)
        self._bag_predictions = self.predict(self._bags)

class miSVM(SIL):

    def __init__(self, *args, **kwargs):
        self.restarts = kwargs.pop('restarts', 0)
        self.max_iters = kwargs.pop('max_iters', 50)
        super(miSVM, self).__init__(*args, **kwargs)

    def fit(self, bags, y):
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.asmatrix(y).reshape((-1, 1))

        bs = BagSplitter(self._bags, y)
        best_obj = float('inf')
        best_svm = None
        for rr in range(self.restarts + 1):
            if rr == 0:
                if self.verbose: print 'Non-random start...'
                initial_classes = np.vstack([-np.ones((bs.L_n, 1)),
                                              np.ones((bs.L_p, 1))])
            else:
                if self.verbose: print 'Random restart %d of %d...' % (rr, self.restarts)
                rand_classes = np.matrix([np.sign([uniform(-1.0, 1.0)
                                                   for i in range(bs.L_p)])]).T
                initial_classes = np.vstack([-np.ones((bs.L_n, 1)),
                                              rand_classes])
                initial_classes[np.nonzero(initial_classes == 0.0)] = 1.0

            # Setup SVM and QP
            if self.scale_C:
                C = self.C / float(len(bs.instances))
            else:
                C = self.C
            setup = self._setup_svm(bs.instances, initial_classes, C)
            K = setup[0]
            qp = IterativeQP(*setup[1:])

            class miSVMCCCP(CCCP):

                def bailout(cself, svm, classes):
                    return svm

                def iterate(cself, svm, classes):
                    cself.mention('Training SVM...')
                    D = spdiag(classes)
                    qp.update_H(D*K*D)
                    qp.update_Aeq(classes.T)
                    alphas, obj = qp.solve(cself.verbose)

                    # Construct SVM from solution
                    svm = SVM(kernel=self.kernel, gamma=self.gamma, p=self.p,
                              verbose=self.verbose, sv_cutoff=self.sv_cutoff)
                    svm._X = bs.instances
                    svm._y = classes
                    svm._alphas = alphas
                    svm._objective = obj
                    svm._compute_separator(K)
                    svm._K = K

                    cself.mention('Recomputing classes...')
                    p_conf = svm._predictions[-bs.L_p:]
                    pos_classes = np.vstack([_update_classes(part)
                                             for part in
                                             partition(p_conf, bs.pos_groups)])
                    new_classes = np.vstack([-np.ones((bs.L_n, 1)), pos_classes])

                    class_changes = round(np.sum(np.abs(classes - new_classes)/2))
                    cself.mention('Class Changes: %d' % class_changes)
                    if class_changes == 0:
                        return None, svm

                    return {'svm': svm, 'classes': new_classes}, None

            cccp = miSVMCCCP(verbose=self.verbose, svm=None,
                             classes=initial_classes, max_iters=self.max_iters)
            svm = cccp.solve()
            if svm is not None:
                obj = float(svm._objective)
                if obj < best_obj:
                    best_svm = svm
                    best_obj = obj

        if best_svm is not None:
            self._X = best_svm._X
            self._y = best_svm._y
            self._alphas = best_svm._alphas
            self._objective = best_svm._objective
            self._compute_separator(best_svm._K)

def _update_classes(x):
    classes = np.sign(x)
    # If classification happened to
    # be zero, make it 1.0
    classes[np.nonzero(classes == 0.0)] = 1.0
    # Guarantee that at least one
    # instance is positive
    classes[np.argmax(x)] = 1.0
    return classes.reshape((-1, 1))
