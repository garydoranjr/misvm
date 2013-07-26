MISVM: Multiple-Instance Support Vector Machines
================================================

by Gary Doran (<gary.doran@case.edu>)

Installation
------------

This package can be installed in two ways (the easy way):

    # If needed:
    # pip install numpy
    # pip install scipy
    # pip install cvxopt
    pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm

or by running the setup file manually

    git clone [the url for misvm]
    cd misvm
    python setup.py install

Note the code depends on the `numpy`, `scipy`, and `cvxopt` packages. So have those
installed first. The build will likely fail if it can't find them. For more information, see:

 + [NumPy](http://www.numpy.org/): Library for efficient matrix math in Python
 + [SciPy](http://www.scipy.org/): Library for more MATLAB-like functionality
 + [CVXOPT](http://cvxopt.org/): Efficient convex (including quadratic program) optimization

Contents
--------

The MISVM package currently implements the following support vector machine
(SVM) approaches for the multiple-instance (MI) learning framework:

### SIL
Single-Instance Learning (SIL) is a "naive" approach that assigns each instance
the label of its bag, creating a supervised learning problem but mislabeling
negative instances in positive bags. It works surprisingly well for many
problems.
> Ray, Soumya, and Mark Craven. **Supervised versus multiple instance learning:
> an empirical comparison.** Proceedings of the 22nd International Conference on
> Machine Learning. 2005.

### MI-SVM and mi-SVM
These approaches modify the standard SVM formulation so that the constraints on
instance labels correspond to the MI assumption that at least one instance in
each bag is positive. For more information, see:
> Andrews, Stuart, Ioannis Tsochantaridis, and Thomas Hofmann. **Support vector
> machines for multiple-instance learning.** _Advances in Neural Information
> Processing Systems._ 2002.

### NSK and STK
The normalized set kernel (NSK) and statistics kernel (STK) approaches use
kernels to map entire bags into a features, then use the standard SVM
formulation to find bag classifiers:
> Gärtner, Thomas, Peter A. Flach, Adam Kowalczyk, and Alex J. Smola.
> **Multi-instance kernels.** _Proceedings of the 19th International Conference on
> Machine Learning._ 2002.

### MissSVM
MissSVM uses a semi-supervised learning approach, treating the instances in
positive bags as unlabeled data:
> Zhou, Zhi-Hua, and Jun-Ming Xu. **On the relation between multi-instance
> learning and semi-supervised learning.** _Proceedings of the 24th
> International Conference on Machine Learning._ 2007.

### MICA
The "multiple-instance classification algorithm" (MICA) represents each bag
using a convex combinations of its instances. The optimization program is then
solved by iteratively solving a series of linear programs. In our formulation,
we use L2 regularization, so we solve alternating linear and quadratic programs.
For more information on the original algorithm, see:
> Mangasarian, Olvi L., and Edward W. Wild. **Multiple instance classification
> via successive linear programming.** _Journal of Optimization Theory and
> Applications_ 137.3 (2008): 555-568.

### sMIL, stMIL, and sbMIL
This family of approaches intentionally bias SVM formulations to handle the
assumption that there are very few positive instances in each positive bag. In
the case of sbMIL, prior knowledge on the "sparsity" of positive bags can be
specified or found via cross-validation:
> Bunescu, Razvan C., and Raymond J. Mooney. **Multiple instance learning for
> sparse positive bags.** _Proceedings of the 24th International Conference on
> Machine Learning._ 2007.

Questions and Issues
--------------------

If you find any bugs or have any questions about this code, please create an
issue on [GitHub](https://github.com/garydoranjr/misvm/issues), or contact Gary
Doran at <gary.doran@case.edu>. Of course, I cannot guarantee any support for
this software.
