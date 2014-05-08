try:
    from setuptools import setup

    setup  # quiet "redefinition of unused ..." warning from pyflakes
    # arguments that distutils doesn't understand
    setuptools_kwargs = {
        'install_requires': [
            'numpy',
            'scipy',
            'cvxopt',
        ],
        'provides': ['misvm'],
    }
except ImportError:
    from distutils.core import setup

    setuptools_kwargs = {}

setup(name='misvm',
      version="1.0",
      description=(
          """
          Implementations of various
          multiple-instance support vector machine approaches
          """
      ),
      author='Gary Doran',
      author_email='gary.doran@case.edu',
      url='https://github.com/garydoranjr/misvm.git',
      license="BSD compatable (see the LICENSE file)",
      packages=['misvm'],
      platforms=['unix'],
      scripts=[],
      **setuptools_kwargs)
