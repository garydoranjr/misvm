"""
Implements standard code for problems that
require the Concave-Convex Procedure (CCCP),
or similar iteration.
"""
from sys import stderr


class CCCP(object):
    """
    Encapsulates the CCCP
    """
    TOLERANCE = 1e-6

    def __init__(self, verbose=True, max_iters=50, **kwargs):
        self.verbose = verbose
        self.max_iters = (max_iters + 1)
        self.kwargs = kwargs

    def mention(self, message):
        if self.verbose:
            print message

    def solve(self):
        """
        Called to solve the CCCP problem
        """
        for i in xrange(1, self.max_iters):
            self.mention('\nIteration %d...' % i)
            try:
                self.kwargs, solution = self.iterate(**self.kwargs)
            except Exception as e:
                if self.verbose:
                    print >> stderr, 'Warning: Bailing due to error: %s' % e
                return self.bailout(**self.kwargs)
            if solution is not None:
                return solution

        if self.verbose:
            print >> stderr, 'Warning: Max iterations exceeded'
        return self.bailout(**self.kwargs)

    def iterate(self, **kwargs):
        """
        Should perform an iteration of the CCCP,
        using values in kwargs, and returning the
        kwargs for the next iteration.

        If the CCCP should terminate, also return the
        solution; otherwise, return 'None'
        """
        pass

    def bailout(self, **kwargs):
        """
        Return a solution in the case that the
        maximum allowed iterations was exceeded.
        """
        pass

    def check_tolerance(self, last_obj, new_obj=0.0):
        """
        Compares objective values, or takes the first
        value as delta if no second argument is given.
        """
        if last_obj is not None:
            delta_obj = abs(float(new_obj) - float(last_obj))
            self.mention('delta obj ratio: %.2e' % (delta_obj / self.TOLERANCE))
            return delta_obj < self.TOLERANCE
        return False
