"""
Utility functions and classes
"""
from __future__ import print_function, division
import numpy as np
import scipy.sparse as sp
from itertools import chain
from random import uniform


def rand_convex(n):
    rand = np.matrix([uniform(0.0, 1.0) for i in range(n)])
    return rand / np.sum(rand)


def spdiag(x):
    n = len(x)
    return sp.spdiags(x.flat, [0], n, n)


def partition(items, group_sizes):
    """
    Partition a sequence of items
    into groups of the given sizes
    """
    i = 0
    for group in group_sizes:
        yield items[i: i + group]
        i += group


def slices(groups):
    """
    Generate slices to select
    groups of the given sizes
    within a list/matrix
    """
    i = 0
    for group in groups:
        yield i, i + group
        i += group


class BagSplitter(object):
    def __init__(self, bags, classes):
        self.bags = bags
        self.classes = classes

    def __getattr__(self, name):
        if name == 'pos_bags':
            self.pos_bags = [bag for bag, cls in
                             zip(self.bags, self.classes)
                             if cls > 0.0]
            return self.pos_bags
        elif name == 'neg_bags':
            self.neg_bags = [bag for bag, cls in
                             zip(self.bags, self.classes)
                             if cls <= 0.0]
            return self.neg_bags
        elif name == 'neg_instances':
            self.neg_instances = np.vstack(self.neg_bags)
            return self.neg_instances
        elif name == 'pos_instances':
            self.pos_instances = np.vstack(self.pos_bags)
            return self.pos_instances
        elif name == 'instances':
            self.instances = np.vstack([self.neg_instances,
                                        self.pos_instances])
            return self.instances
        elif name == 'inst_classes':
            self.inst_classes = np.vstack([-np.ones((self.L_n, 1)),
                                           np.ones((self.L_p, 1))])
            return self.inst_classes
        elif name == 'pos_groups':
            self.pos_groups = [len(bag) for bag in self.pos_bags]
            return self.pos_groups
        elif name == 'neg_groups':
            self.neg_groups = [len(bag) for bag in self.neg_bags]
            return self.neg_groups
        elif name == 'L_n':
            self.L_n = len(self.neg_instances)
            return self.L_n
        elif name == 'L_p':
            self.L_p = len(self.pos_instances)
            return self.L_p
        elif name == 'L':
            self.L = self.L_p + self.L_n
            return self.L
        elif name == 'X_n':
            self.X_n = len(self.neg_bags)
            return self.X_n
        elif name == 'X_p':
            self.X_p = len(self.pos_bags)
            return self.X_p
        elif name == 'X':
            self.X = self.X_p + self.X_n
            return self.X
        elif name == 'neg_inst_as_bags':
            self.neg_inst_as_bags = [inst for inst in chain(*self.neg_bags)]
            return self.neg_inst_as_bags
        elif name == 'pos_inst_as_bags':
            self.pos_inst_as_bags = [inst for inst in chain(*self.pos_bags)]
            return self.pos_inst_as_bags
        else:
            raise AttributeError('No "%s" attribute.' % name)
        raise Exception("Unreachable %s" % name)
