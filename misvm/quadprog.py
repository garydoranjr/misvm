from __future__ import print_function, division
from cvxopt import matrix as cvxmat, sparse, spmatrix
from cvxopt.solvers import qp, options
from sys import stderr
from itertools import count

from numpy import eye, vstack, matrix


class IterativeQP(object):
    """
    Iteratively solves QPs, allowing
    an update of parameters and using
    the previous solution as an initial solution
    """

    def __init__(self, H, f, Aeq, beq, lb, ub, fix_pd=False):
        """
        minimize:
                (1/2)*x'*H*x + f'*x
        subject to:
                Aeq*x = beq
                lb <= x <= ub
        """
        self.lb = lb
        (self.P, self.q, self.G,
         self.h, self.A, self.b) = _convert(H, f, Aeq, beq, lb, ub)
        self.last_results = None
        self.fix_pd = fix_pd

    def update_ub(self, ub):
        self.h = cvxmat(vstack([-self.lb, ub]))
        # Old results no longer valid
        self.last_results = None

    def update_H(self, H):
        self.P = cvxmat(H)

    def update_Aeq(self, Aeq):
        if Aeq is None:
            self.A = None
        else:
            self.A = cvxmat(Aeq)
        # Old results no longer valid
        self.last_results = None

    def _ensure_pd(self, epsilon):
        """
        Add epsilon times identity matrix
        to P to ensure numerically it is P.D.
        """
        n = self.P.size[0]
        self.P = self.P + cvxmat(epsilon * eye(n))

    def clear_results(self):
        self.last_results = None

    def solve(self, verbose=False):
        # Optimize
        old_settings = _apply_options({'show_progress': verbose})

        for i in count(-9):
            try:
                results = qp(self.P, self.q, self.G,
                             self.h, self.A, self.b,
                             initvals=self.last_results)
                break
            except ValueError as e:
                # Sometimes the hessian isn't full rank,
                # due to numerical error
                if self.fix_pd:
                    eps = 10.0 ** i
                    print('Rank error while solving, adjusting to fix...')
                    print('Using epsilon = %.1e' % eps)
                    self._ensure_pd(eps)
                else:
                    raise e

        _apply_options(old_settings)

        # Store results
        self.last_results = results

        # Check return status
        status = results['status']
        if not status == 'optimal':
            print('Warning: termination of qp with status: %s'
                  % status, file=stderr)

        # Convert back to NumPy matrix
        # and return solution
        xstar = results['x']
        obj = Objective((0.5 * xstar.T * self.P * xstar)[0], (self.q.T * xstar)[0])
        return matrix(xstar), obj


def quadprog(H, f, Aeq, beq, lb, ub, verbose=False, fix_pd=False):
    """
    minimize:
            (1/2)*x'*H*x + f'*x
    subject to:
            Aeq*x = beq
            lb <= x <= ub
    """
    qp = IterativeQP(H, f, Aeq, beq, lb, ub, fix_pd)
    return qp.solve(verbose)


def speye(n):
    """Create a sparse identity matrix"""
    r = range(n)
    return spmatrix(1.0, r, r)


def spzeros(r, c=1):
    """Create a sparse zero vector or matrix"""
    return spmatrix([], [], [], (r, c))


def _convert(H, f, Aeq, beq, lb, ub):
    """
    Convert everything to
    cvxopt-style matrices
    """
    P = cvxmat(H)
    q = cvxmat(f)
    if Aeq is None:
        A = None
    else:
        A = cvxmat(Aeq)
    if beq is None:
        b = None
    else:
        b = cvxmat(beq)

    n = lb.size
    G = sparse([-speye(n), speye(n)])
    h = cvxmat(vstack([-lb, ub]))
    return P, q, G, h, A, b


def _apply_options(option_dict):
    old_settings = {}
    for k, v in option_dict.items():
        old_settings[k] = options.get(k, None)
        if v is None:
            del options[k]
        else:
            options[k] = v
    return old_settings


class Objective(object):
    def __init__(self, quadratic, linear):
        self.objective = quadratic + linear
        self.quadratic = quadratic
        self.linear = linear

    def __float__(self):
        return float(self.objective)

    def __str__(self):
        return str(self.objective)
