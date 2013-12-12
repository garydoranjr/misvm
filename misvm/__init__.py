"""
MISVM: An implementation of multiple-instance support vector machines

The following algorithms are implemented:

  SVM     : a standard supervised SVM
  SIL     : trains a standard SVM classifier after applying bag labels to each
            instance
  MISVM   : the MI-SVM algorithm of Andrews, Tsochantaridis, & Hofmann (2002)
  miSVM   : the mi-SVM algorithm of Andrews, Tsochantaridis, & Hofmann (2002)
  NSK     : the normalized set kernel of Gaertner, et al. (2002)
  STK     : the statistics kernel of Gaertner, et al. (2002)
  MissSVM : the semi-supervised learning approach of Zhou & Xu (2007)
  MICA    : the MI classification algorithm of Mangasarian & Wild (2008)
  sMIL    : sparse MIL (Bunescu & Mooney, 2007)
  stMIL   : sparse, transductive  MIL (Bunescu & Mooney, 2007)
  sbMIL   : sparse, balanced MIL (Bunescu & Mooney, 2007)
"""
__name__ = 'misvm'
__version__ = '1.0'
from svm import SVM
from sil import SIL
from nsk import NSK
from smil import sMIL
from misvm import miSVM, MISVM
from stk import STK
from stmil import stMIL
from sbmil import sbMIL
from mica import MICA
from misssvm import MissSVM
