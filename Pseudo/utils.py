#!/usr/bin/python

############################################################################
##
## File:      utils.py
##
## Purpose:   Utility functions.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Apr. 1, 2021
##
############################################################################

import constants as const

from math                        import log,ceil,floor,sqrt,inf
from itertools                   import combinations
from matplotlib                  import colors
from sklearn.metrics             import roc_curve,RocCurveDisplay,auc,confusion_matrix,ConfusionMatrixDisplay

import matplotlib.pyplot         as plt
import numpy                     as np
import pandas                    as pd
import seaborn                   as sn
import tensorflow                as tf

import os

np.random.seed(12345)
tf.random.set_seed(12345)

############################################################################
##
## Purpose:   Unique list because np.unique returns strange results
##
############################################################################
def unique(l=[]):
    ret  = []
    if not (len(l) == 0):
        s    = l
        if type(s[0]) == type([]):
            s    = [tuple(t) for t in s]
        ret  = list(set(s))
        if type(s[0]) == type([]):
            ret  = [list(t) for t in ret]
    return np.asarray(ret)

############################################################################
##
## Purpose:   Permutations of a list of integers
##
############################################################################
def permute(dat=[],mine=True,l=const.constants.MAX_FEATURES):
    ret  = []
    sz   = len(dat)
    if not (sz == 0):
        if mine:
            # permute the array of indices beginning with the first element
            for j in range(0,sz+1):
                # all permutations of the array of indices
                jdat = list(dat[j:])
                jdat.extend(list(dat[:j]))
                for i in range(0,sz):
                    # only retain the sub arrays that are length >= 2
                    tmp = [list(x) for x in combinations(jdat,i+2)]
                    if len(tmp) > 0:
                        ret.extend(tmp)
        else:
            # permute the array of indices beginning with the first element
            lsz  = min(l,const.constants.MAX_FEATURES) if not (0 < l and l < min(const.constants.MAX_FEATURES,sz)) else l
            ret.extend(list(combinations(dat,lsz)))
    return unique(ret)

############################################################################
##
## Purpose:   Calculate the number of clusters to form using random clusters
##
############################################################################
def calcN(pts=None):
    ret = 2**2
    if (pts != None and type(pts) == type(0)):
        M    = ceil(sqrt(pts))
        for i in range(2,const.constants.MAX_FEATURES+1):
            N    = max(floor(sqrt(i)),2)
            if pts/(pts+(2*M*((N-1)^2))) <= 0.5:
                ret  = N**2
            else:
                break
    return ret

############################################################################
##
## Purpose:   Most frequently appearing element in a list
##
############################################################################
def most(l=[]):
    ret  = 0
    if (type(l) == type([]) or type(l) == type(np.asarray([]))):
        ll   = list(l)
        ret  = max(set(ll),key=ll.count)
    return ret

############################################################################
##
## Purpose:   Generate a list of hex colors for visual display of variance
##
############################################################################
def clrs(num=0):
    ret  = {}
    if not num <= 0:
        gc   = lambda n: list(map(lambda i: "#" + "%06x" % np.random.randint(0,0xFFFFFF),range(0,num)))
        clr  = gc(num)
        ret  = {i:clr[i] for i in range(0,num)}
    return ret

############################################################################
##
## Purpose:   Visual display of variance is plotted
##
############################################################################
def pclrs(mat=[],fl=None):
    ret  = {}
    if (type(mat) == type([]) or type(mat) == type(np.asarray([]))):
        if not len(mat) == 0:
            # need the data to be square if it is not already
            lm   = list(mat)
            dim  = int(ceil(sqrt(len(mat))))
            # extend the list of variance indicator
            if dim*dim > len(lm):
                lm.extend(np.full((dim*dim)-len(lm),0))
            # reshape the data for the heat map
            lm   = list(np.asarray(lm).reshape((dim,dim)))
            # now to map the variance indicators to colors
            disp = sn.heatmap(pd.DataFrame(lm))#,annot=True,annot_kws={"size":16})
            disp.plot()
            # save the image if requested
            if type(fl) == type(""):
                plt.savefig(fl)
            else:
                plt.show()
    return

############################################################################
##
## Purpose:   Plot the ROC curve
##
############################################################################
def roc(tgts=[],scrs=[],fl=None):
    if (type(tgts) == type([]) or type(tgts) == type(np.asarray([]))) and \
       (type(scrs) == type([]) or type(scrs) == type(np.asarray([]))):
        if not (len(tgts) == 0 or len(scrs) == 0):
            # the false and true positve rate and the threshold
            fpr,tpr,thres = roc_curve(tgts,scrs)
            rauc          = auc(fpr,tpr)
            disp          = RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=rauc)
            disp.plot()
            # save the image if requested
            if type(fl) == type(""):
                plt.savefig(fl)
            else:
                plt.show()
    return

############################################################################
##
## Purpose:   Plot the confusion matrix
##
############################################################################
def confusion(tgts=[],prds=[],fl=None):
    if (type(tgts) == type([]) or type(tgts) == type(np.asarray([]))) and \
       (type(prds) == type([]) or type(prds) == type(np.asarray([]))):
        if not (len(tgts) == 0 or len(prds) == 0):
            # the false and true positve rate and the threshold
            cm   = confusion_matrix(tgts,prds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            # save the image if requested
            if type(fl) == type(""):
                plt.savefig(fl)
            else:
                plt.show()
    return

############################################################################
##
## Purpose:   Utils class
##
############################################################################
class utils():
    @staticmethod
    def _unique(l=[]):
        return unique(l)
    @staticmethod
    def _permute(dat=[],mine=True,l=const.constants.MAX_FEATURES):
        return permute(dat,mine,l)
    @staticmethod
    def _calcN(pts=None):
        return calcN(pts)
    @staticmethod
    def _most(l=[]):
        return most(l)
    @staticmethod
    def _clrs(num=0):
        return clrs(num)
    @staticmethod
    def _pclrs(mat=[],fl=None):
        return pclrs(mat,fl)
    @staticmethod
    def _roc(tgts=[],scrs=[],fl=None):
        return roc(tgts,scrs,fl)
    @staticmethod
    def _confusion(tgts=[],prds=[],fl=None):
        return confusion(tgts,prds,fl)
