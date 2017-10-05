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
from misvm.svm import SVM
from misvm.sil import SIL
from misvm.nsk import NSK
from misvm.smil import sMIL
from misvm.mi_svm import miSVM, MISVM
from misvm.stk import STK
from misvm.stmil import stMIL
from misvm.sbmil import sbMIL
from misvm.mica import MICA
from misvm.misssvm import MissSVM
