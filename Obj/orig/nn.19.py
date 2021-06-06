#!/usr/bin/python

############################################################################
##
## File:      nn.py
##
## Purpose:   Neural network logic.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Nov. 6, 2020
##
############################################################################

import sys

import constants as const

from roi                         import rp,roif,unique,restore3d,calcN

ver  = sys.version.split()[0]

if ver == const.VER:
    from            keras.layers         import Dense,BatchNormalization,Activation,Conv2D,Conv2DTranspose
    from            keras.models         import Sequential,load_model,Model
    from            keras.utils.np_utils import to_categorical
    from            keras.callbacks      import ModelCheckpoint
else:
    from tensorflow.keras.layers         import Dense,BatchNormalization,Activation,Conv2D,Conv2DTranspose
    from tensorflow.keras.models         import Sequential,load_model,Model
    from tensorflow.keras.utils          import to_categorical
    from tensorflow.keras.callbacks      import ModelCheckpoint

#from tensorflow.keras import backend as K

from joblib                      import Parallel,delayed
from math                        import log,ceil,floor,sqrt,inf
from itertools                   import combinations
from contextlib                  import closing
from matplotlib.pyplot           import imread,savefig
#from cv2                         import imread
from datetime                    import datetime
from PIL                         import Image
from scipy                       import stats
from glob                        import glob 

import multiprocessing       as mp
import numpy                 as np
import pandas                as pd
import tensorflow            as tf
import xml.etree.ElementTree as ET

import os
import cv2

np.random.seed(12345)

############################################################################
##
## Purpose:   Permutations of a list of integers
##
############################################################################
def permute(dat=[],mine=True,l=const.MAX_FEATURES):
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
            lsz  = min(l,const.MAX_FEATURES) if not (0 < l and l < min(const.MAX_FEATURES,sz)) else l
            ret.extend(list(combinations(dat,lsz)))
    return unique(ret)

############################################################################
##
## Purpose:   Deep belief network support function
##
############################################################################
def dbnlayers(model=None,outp=const.OUTP,shape=const.SHAPE,act=None,useact=False):
    if not (type(model) == type(None) or outp < const.OUTP or shape < const.OUTP or type(act) == type(None)):
        if not useact:
            # encode the input data using the rectified linear unit
            enc  = Dense(outp,input_shape=(shape,),activation=act)
            # add the input layer to the model
            model.add(enc)
        else:
            # encode the input data using the rectified linear unit
            enc  = Dense(outp,input_shape=(shape,),use_bias=False)
            # add the input layer to the model
            model.add(enc)
            # add batch normalization
            enc  = BatchNormalization()
            # add the input layer to the model
            model.add(enc)
            # add the activation
            enc  = Activation(act)
            # add the input layer to the model
            model.add(enc)
    return

############################################################################
##
## Purpose:   Deep belief network for classification or regression
##
############################################################################
def dbn(inputs=[]
       ,outputs=[]
       ,sfl=const.SFL
       ,splits=const.SPLITS
       ,props=const.PROPS
       ,clust=const.BVAL
       ,loss=const.LOSS
       ,optimizer=const.OPTI
       ,rbmact=const.RBMA
       ,dbnact=const.DBNA
       ,dbnout=const.DBNO
       ,epochs=const.EPO
       ,embed=const.EMB
       ,encs=const.ENCS
       ,useact=const.USEA
       ,verbose=const.VERB):
    model= None
    if inputs.any():
        # seed the random number generator for repeatable results
        np.random.seed(const.SEED)
        # linear stack of layers in the neural network (NN)
        model= Sequential()
        # add dense layers which are just densely connected typical artificial NN (ANN) layers
        #
        # at the input layer, there is a tendency for all NNs to act as imaging Gabor filters
        # where there's an analysis of the content of the inputs to determine whether there is
        # any specific content in any direction of the multi-dimensional inputs ... i.e. a Gabor filter
        # is a feature selector
        #
        # at the onset, the input level and second level match
        # first sibling level matches the number of features
        # all are deemed important and we will not allow feature selection as a result
        # this follows the writings in "Auto-encoding a Knowledge Graph Using a Deep Belief Network"
        #
        # the number of features, M, should be a function of len(inputs[0,0]) and the size of the kernel
        # as we will be building the model using the flattened kernels as inputs and reconstructing the input
        # data sets after retrieving the output from the model
        #
        # the kernel size will be determined using random cluster theory that won't be explained here, but
        # simply involves the projection of multidimensional data to a 2D space, while preserving class membership
        # of each data point ... the projection results in uniformly sized partitions in the 2D space, giving
        # the dimension of the kernel ... these kernels define the maximal region of disjoint subsets of
        # of data points, given the number of rows in the data set, such that all points in each disjoint
        # set are determinative of each other's states, according to random field theory ... this is precisely
        # what we want to model when filtering the data set to reveal its underlying equilibrium distribution
        #
        # recall that a distribution of data which exhibits the Markov property can be modeled as the sum of
        # a deterministic lower dimensional subspace plus additive noise ... it is the noise that we seek to filter
        # thereby revealing the underlying subspace
        ip   = inputs
        op   = outputs
        S    = splits
        if type(ip     ) == type(np.asarray([])) and \
           type(ip[0  ]) == type(np.asarray([])) and \
           type(ip[0,0]) == type(np.asarray([])):
            # without going into a lot of detail here, using a result based upon the random cluster model
            # we can estimate the number of classes to form as by assuming that we have N^2 classes containing
            # a total of M^2 data points, uniformly and evenly distributed throughout a bounded uniformly
            # partitioned unit area in the 2D plane
            #
            # then the probability of connection for any 2 data points in the bounded region is given by
            # M^2 / (M^2 + (2M * (N-1)^2)) and by the random cluster model, this probability has to equate
            # to 1/2, which gives a way to solve for N, the minimum number of clusters to form for a
            # data set consistinng of M^2 Gaussian distributed data points projected into 2D space
            #
            # each class contains M^2/N^2 data points ... the sqrt of this number is the size of the kernel we want
            #
            # note that the number of data points is M^2 in the calculations, different than M below
            kM   = max(2,ceil(sqrt(len(ip[0,0])/calcN(len(ip[0,0])))))
            # inputs have kM columns and any number of rows, while output has kM columns and any number of rows
            #
            # encode the input data using the scaled exponential linear unit
            # kernel size has to be 1 and strides 1 so that the dimensions of the data remains the same
            M    = len(ip[0,0,0])
            # convolution is really only the mean of a function of the center of the kernel window
            # with respect to a function of the remaining values in the kernel window
            #
            # convolution by its very nature guarantees stochastic separability, but it is better
            # for this separability to be found before we partition the image ... However,
            # we are still good by having it here
            #
            # we will convolve beginning with the lowest resolution to the highest resolution
            # then test the estimate of the blurred image against the original blurred image
            # in an auto-encoder setup
            a    = list(range(kM,(const.CONVS if hasattr(const,"CONVS") and const.CONVS >= kM else 10+kM)))
            a.reverse()
            for i in a:
                if ver == const.VER:
                    enc  = Conv2D(nb_filter=M,nb_row=i,nb_col=i,input_shape=ip[0].shape,subsample=1,activation='tanh',border_mode='same',init='random_normal')
                    #enc  = Conv2D(nb_filter=M,nb_row=1,nb_col=1,input_shape=ip[0].shape,subsample=1,activation='tanh',border_mode='same',init='random_normal')
                else:
                    enc  = Conv2D(filters=M,kernel_size=i,input_shape=ip[0].shape,strides=1,activation='selu',padding='same',kernel_initializer='random_normal')
                    #enc  = Conv2D(filters=M,kernel_size=1,input_shape=ip[0].shape,strides=1,activation='selu',padding='same',kernel_initializer='random_normal')
                model.add(enc)
        else:
            # add dense layers which are just densely connected typical artificial NN (ANN) layers
            #
            # at the input layer, there is a tendency for all NNs to act as imaging Gabor filters
            # where there's an analysis of the content of the inputs to determine whether there is
            # any specific content in any direction of the multi-dimensional inputs ... i.e. a Gabor filter
            # is a feature selector
            #
            # at the onset, the input level and second level match
            # first sibling level matches the number of features
            # all are deemed important and we will not allow feature selection as a result
            # this follows the writings in "Auto-encoding a Knowledge Graph Using a Deep Belief Network"
            #
            # if all possible levels, then M  = len(inputs[0])
            #M    = min(len(ip[0]),props)
            M    = len(ip[0])
            # inputs have M columns and any number of rows, while output has M columns and any number of rows
            #
            # encode the input data using the rectified linear unit
            dbnlayers(model,M,M,'relu',useact)
        # add other encodings that are being passed in the encs array
        if not (type(encs) == type(None)):
            if not (len(encs) == 0):
                for enc in encs:
                    model.add(enc)
        # if M > const.MAX_FEATURES, then we will embed the inputs in a lower dimensional space of dimension const.MAX_FEATURES
        #
        # embed the inputs into a lower dimensional space if M > min(const.MAX_FEATURES,props)
        if embed:
            p    = min(const.MAX_FEATURES,props)#if not op.any() else min(const.MAX_FEATURES,min(props,op.shape[len(op.shape)-1]))
            if M > p:
                dbnlayers(model,p,M,'tanh' if ver == const.VER else 'selu',useact)
                M    = max(int(ceil(log(p,S)/2.0)),p)
                # project the outputs into the lower dimensional subspace
                op   = np.asarray(op)[:,:(S**(2*p))]
        # add the rest of the layers according to the writings
        #
        # output dimension (odim) is initially S (number of splits in first level of hierarchy)
        odim = S
        # the loop variable counts the number of restricted boltzmann machines (RBM)
        # that are defined by the 2 extra layers that are added to the model
        # note that we should always have M RBMs, one for each property of the data
        for J in range(0,M):
            # the dimensionality will be computed in the loop as what's specified in
            # the writings works well for binary splits and 2 properties, but the loop
            # can be generalized to make the computation a bit better
            #
            # S^((J-2)/2)*M^(J/2)     for sibling levels in the writings
            # S^((J-1)/2)*M^((J-1)/2) for binary  levels in the writings
            #
            # for the sibling layers, the input is the old output dimension
            if J == 0:
                dim  = odim
            else:
                dim  = S * odim
            # next layers using the scaled exponential linear unit (Gibbs distribution) are siblings
            dbnlayers(model,dim,odim,'tanh' if ver == const.VER else 'selu',useact)
            # redefine the input and output dimensions for the binary layers
            odim = S * dim
            # next layers using the softmax are binary layers
            #
            # note that by using dense layers of ANNs we can use back propagation within each RBM and across
            # all RBMs to tune the weights in each layer so that the hidden layers have a structure
            # that results in the correct number of clusters in the output layer giving rise to
            # the hierarchy that is detailed in the writings and the slide deck ... without this tuning
            # through back propagation, there is no guarantee that the hidden layers' structure will match
            # what is expected
            #
            # also, Gibbs sampling is being performed in the hidden layers so that edges are open
            # and closed to give rise to the correct cluster structure in the hidden layers defined above
            # so that the tuning through back propagation leads to the equilibrium distribution that can be color
            # coded into distinct regions of connected clusters ... see the writings for an example
            if not (J == M - 1):
                #dbnlayers(model,odim,dim,'sigmoid',useact)
                dbnlayers(model,odim,dim,rbmact,useact)
            else:
                if not (type(dbnact) == type(None) or dbnout <= 0):
                    dbnlayers(model,odim,dim,'sigmoid',useact)
                else:
                    # add another layer to change the structure of the network if needed based on clusters
                    if not (clust <= 0):
                        dbnlayers(model,clust,dim,rbmact,useact)
                    else:
                        dbnlayers(model,odim ,dim,rbmact,useact)
        # add another layer for a different kind of model, such as a regression model
        if not (type(dbnact) == type(None) or dbnout <= 0):
            # preceding layers plus this layer now perform auto encoding
            dbnlayers(model,M,odim,'tanh' if ver == const.VER else 'selu',useact)
            if type(ip     ) == type(np.asarray([])) and \
               type(ip[0  ]) == type(np.asarray([])) and \
               type(ip[0,0]) == type(np.asarray([])):
                # without going into a lot of detail here, using a result based upon the random cluster model
                # we can estimate the number of classes to form as by assuming that we have N^2 classes containing
                # a total of M^2 data points, uniformly and evenly distributed throughout a bounded uniformly
                # partitioned unit area in the 2D plane
                #
                # then the probability of connection for any 2 data points in the bounded region is given by
                # M^2 / (M^2 + (2M * (N-1)^2)) and by the random cluster model, this probability has to equate
                # to 1/2, which gives a way to solve for N, the minimum number of clusters to form for a
                # data set consistinng of M^2 Gaussian distributed data points projected into 2D space
                #
                # each class contains M^2/N^2 data points ... the sqrt of this number is the size of the kernel we want
                #
                # note that the number of data points is M^2 in the calculations, different than M below
                kM   = max(2,ceil(sqrt(len(ip[0,0])/calcN(len(ip[0,0])))))
                # inputs have kM columns and any number of rows, while output has kM columns and any number of rows
                #
                # encode the input data using the scaled exponential linear unit
                # kernel size has to be 1 and strides 1 so that the dimensions of the data remains the same
                M    = len(ip[0,0,0])
                # convolution is really only the mean of a function of the center of the kernel window
                # with respect to a function of the remaining values in the kernel window
                #
                # convolution by its very nature guarantees stochastic separability, but it is better
                # for this separability to be found before we partition the image ... However,
                # we are still good by having it here
                #
                # deconvolution recovers a representation of the original values as samples
                # from an embedded subspace of the noisy sample space
                #
                # we will deconvolve beginning with the highest resolution to the lowest resolution
                # then test the estimate of the blurred image against the original blurred image
                # in an auto-encoder setup
                a    = list(range(kM,(const.CONVS if hasattr(const,"CONVS") and const.CONVS >= kM else 10+kM)))
                for i in a:
                    if ver == const.VER:
                        enc  = Conv2DTranspose(nb_filter=M,nb_row=i,nb_col=i,input_shape=ip[0].shape,subsample=1,activation='tanh',border_mode='same',init='random_normal')
                        #enc  = Conv2DTranspose(nb_filter=M,nb_row=1,nb_col=1,input_shape=ip[0].shape,subsample=1,activation='tanh',border_mode='same',init='random_normal')
                    else:
                        enc  = Conv2DTranspose(filters=M,kernel_size=i,input_shape=ip[0].shape,strides=1,activation='selu',padding='same',kernel_initializer='random_normal')
                        #enc  = Conv2DTranspose(filters=M,kernel_size=1,input_shape=ip[0].shape,strides=1,activation='selu',padding='same',kernel_initializer='random_normal')
                    model.add(enc)
            # requested model at the output layer of this RBM
            dbnlayers(model,dbnout,M,dbnact,useact)
        # optimize using the typical categorical cross entropy loss function with root mean square optimizer to find weights
        model.compile(loss=loss,optimizer=optimizer)
        if not (type(sfl) == type(None)):
            # construct the relaxed image name
            #
            # file name minus the type extension
            fln  = sfl[:sfl.rfind(".") ]
            # file type extension
            flt  = ".hdf5"
            # current date and time
            dt   = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # model checkpoint call back
            nsfl = fln + "_" + dt + flt
            chkpt= ModelCheckpoint(filepath=nsfl,save_weights_only=False,monitor='val_acc',mode='max')#,save_best_only=True)
            # we will allow for 100 iterations through the training data set to find the best sets of weights for the layers
            # fit the model using the flattened inputs and outputs
            vpct = 1.0 - (const.TRAIN_PCT if hasattr(const,"TRAIN_PCT") else 0.8)
            x    = ip.copy()
            y    = op.copy()
            lenxy= min(len(x),len(y))
            if not lenxy > 1:
                rows = max(2,max(len(x),len(y)))#const.MAX_ROWS if hasattr(const,"MAX_ROWS") else 10
                for i in range(0,rows-len(x)):
                    x    = np.vstack((x,x))
                for i in range(0,rows-len(y)):
                    y    = np.vstack((y,y))
            if ver == const.VER:
                model.fit(x=x,y=y,nb_epoch =epochs,verbose=verbose,callbacks=[chkpt],validation_split=vpct)
            else:
                model.fit(x=x,y=y,   epochs=epochs,verbose=verbose,callbacks=[chkpt],validation_split=vpct)
            # file type extension
            flt  = sfl[ sfl.rfind("."):]
            # save the model
            nsfl = fln + "_" + dt + flt
            model.save(nsfl)
        else:
            # we will allow for 100 iterations through the training data set to find the best sets of weights for the layers
            # fit the model using the flattened inputs and outputs
            if ver == const.VER:
                model.fit(x=ip,y=op,nb_epoch =epochs,verbose=verbose)
            else:
                model.fit(x=ip,y=op,   epochs=epochs,verbose=verbose)
    # return the model to the caller
    return model

############################################################################
##
## Purpose:   Define the outputs for a classification
##
############################################################################
def categoricals(rows=const.BVAL,splits=const.SPLITS,props=const.PROPS,clust=const.BVAL):
    ret  = []
    if not (rows <= const.BVAL or ((splits < const.SPLITS or props < const.PROPS) and clust <= const.BVAL)):
        # we need values to turn into labels when training
        # one-hot encode the integer labels as its required for the softmax
        s    = splits
        p    = min(const.MAX_FEATURES,props)
        if not (clust <= const.BVAL):
            nc   = clust
        else:
            nc   = s**(2*p)
        ret  = [np.random.randint(1,nc) for i in range(0,rows)]
        ret  = to_categorical(ret,nb_classes=nc) if ver == const.VER else to_categorical(ret,num_classes=nc)
    return ret

############################################################################
##
## Purpose:   Define the Kullback-Leibler divergence to measure the good-ness of fit
##
############################################################################
def kld(dat1=[],dat2=[]):
    ret  = 0.0
    if (type(dat1) == type([]) or type(dat1) == type(np.asarray([]))) and \
       (type(dat2) == type([]) or type(dat2) == type(np.asarray([]))):
        M,N,P = dat1.shape
        for i in range(0,M):
            for j in range(0,N):
                for k in range(0,P):
                    ret += (dat1[i][j][k] * np.log(dat1[i][j][k]/dat2[i][j][k]) if not (dat1[i][j][k] <= 0 or dat2[i][j][k] <= 0) else 0.0)
        ret /= (M*N*P)
    return ret

############################################################################
##
## Purpose:   Define the neural network that models the boundary of the ROI
##
############################################################################
def nn_roi(dat1=None,dat2=None):
    model= None
    if not (dat1 == None or dat2 == None):
        if (type(dat1) == type([]) or type(dat1) == type(np.asarray([]))) and \
           (type(dat2) == type([]) or type(dat2) == type(np.asarray([]))):
            if not (len(dat1) == 0 or len(dat2) == 0):
                roi  = [roif(dat1),roif(dat2)]
                lmin = min([len(d) for d in roi])
                # sample data points from the boundary, up to the max len
                ndat = [d[list(np.random.randint(0,len(d),lmin))] for d in roi] if not (lmin <= 1) else None
                if not (ndat == None):
                    ivals= ndat[0]
                    ovals= ndat[1]
                    model= dbn(ivals
                              ,ovals
                              ,loss='mean_squared_error'
                              ,optimizer='sgd'
                              ,rbmact='tanh'
                              ,dbnact='sigmoid'
                              ,dbnout=1)
    return model

############################################################################
##
## Purpose:   Main wrapper for the neural network logic
##
############################################################################
def nn(fl=None):
    # process a single file or a list of files being passed in
    ret  = None
    if not (fl == None):
        # number of cpu cores for multiprocessing
        nc   = const.CPU_COUNT if hasattr(const,"CPU_COUNT") else mp.cpu_count()
        if (type(fl) == type([]) or type(fl) == type(np.asarray([]))):
            ret  = Parallel(n_jobs=nc)(delayed(nn)(f) for f in fl)
        else:
            print(fl)
            ro   = const.RO if hasattr(const,"RO") else True
            ret  = {"fls":[],"roi":None,"kld":None,"swt":None}
            if (os.path.exists(fl) and os.path.getsize(fl) > 0):
                ivals= imread(fl)
                ivals= np.stack((ivals,)*3,axis=-1) if len(ivals.shape) is 2 else ivals
                ivals= ivals.reshape(np.asarray(ivals.shape)[sorted(range(0,len(ivals.shape)),reverse=True)])
                print(ivals.shape)
                # construct the relaxed image name
                #
                # file name minus the type extension
                fln  = fl[:fl.rfind(".") ]
                # file type extension
                flt  = fl[ fl.rfind("."):]
                # current date and time
                dt   = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                # relaxed image file name
                #rfl  = fln + (const.IMGR if hasattr(const,"IMGR") else "_relaxed") + "_" + dt + flt
                rfl  = fln + (const.IMGR if hasattr(const,"IMGR") else "_relaxed") + flt
                print(rfl)
                if not ro:
                    # diff image file name
                    #dfl  = fln + (const.IMGD if hasattr(const,"IMGD") else "_diff"   ) + "_" + dt + flt
                    dfl  = fln + (const.IMGD if hasattr(const,"IMGD") else "_diff"   ) + flt
                    print(dfl)
                    # RGB image file name
                    #bfl  = fln + (const.IMGB if hasattr(const,"IMGB") else "_rgb"    ) + "_" + dt + flt
                    bfl  = fln + (const.IMGB if hasattr(const,"IMGB") else "_rgb"    ) + flt
                    print(bfl)
                    # 3D image file name
                    #tfl  = fln + (const.IMG3 if hasattr(const,"IMG3") else "_3d"    ) + "_" + dt + flt
                    tfl  = fln + (const.IMG3 if hasattr(const,"IMG3") else "_3d"     ) + flt
                    print(tfl)
                    #ret["fls"].extend([fl,rfl,dfl,bfl,tfl])
                    ret["fls"].extend([rfl,dfl,bfl,tfl])
                else:
                    ret["fls"].extend([rfl])
            else:
                return ret
            print([[max(ivals[i].flatten()),ivals[i].shape] for i in range(0,len(ivals))])
            # number of data points, properties and splits
            m    = len(ivals[0])
            p    = len(ivals[0,0])
            # only one feature or property for the analysis
            # just return None ... can't do anything
            if not (m >= 2 or p >= 2):
                return ret
            s    = p
            # we will make the colors of the grayscale image to be relatively close in range
            # as we want to see which color is predicted most often when the colors differ
            #
            # later the permutations in the colums that are taken together by separability
            # due to the Markov property, will allow us a way to stochastically relax the entire image
            # while also not overly taxing the resources of the local machine
            #
            # keep the data at a manageable size for processing
            #
            # handle columns first
            blksc= 0
            if not (p <= const.MAX_COLS):
                # the number of images from excessive columns
                blksc= int(ceil(p/const.MAX_COLS))
                kp   = p
                remc = kp % const.MAX_COLS
                p    = const.MAX_COLS
                s    = p
                imgs = []
                for ival in ivals:
                    for blk in range(0,blksc):
                        if blk < blksc - 1:
                            imgs.append(np.asarray(ival)[:,range(blk*p,(blk+1)*p)])
                        else:
                            e    = (blk+1)*p if blk < blksc-1 else min((blk+1)*p,kp)
                            if remc == 0:
                                imgs.append(np.asarray(ival)[:,range(blk*p,(blk+1)*p)])
                            else:
                                # pad the image with a black strip to make the dimensions match ... cols in 2nd arg, no rows
                                imgs.append(np.pad(np.asarray(ival)[:,range(blk*p,e)],[(0,0),(0,const.MAX_COLS-remc)],'constant',constant_values=const.FILL_COLOR))
                ivals= np.asarray(imgs)
            # now handle rows
            blksr= 0
            if not (m <= const.MAX_ROWS):
                # the number of images from excessive rows
                blksr= int(ceil(m/const.MAX_ROWS))
                km   = m
                remr = km % const.MAX_ROWS
                m    = const.MAX_ROWS
                imgs = []
                for ival in ivals:
                    for blk in range(0,blksr):
                        if blk < blksr - 1:
                            imgs.append(np.asarray(ival)[range(blk*m,(blk+1)*m),:])
                        else:
                            e    = (blk+1)*m if blk < blksr-1 else min((blk+1)*m,km)
                            if remr == 0:
                                imgs.append(np.asarray(ival)[range(blk*m,(blk+1)*m),:])
                            else:
                                # pad the image with a black strip to make the dimensions match ... rows in 2nd arg, no cols
                                imgs.append(np.pad(np.asarray(ival)[range(blk*m,e),:],[(0,const.MAX_ROWS-remr),(0,0)],'constant',constant_values=const.FILL_COLOR))
                ivals= np.asarray(imgs)
            # the next steps are allowable as a result of the random cluster model and separable stochastic processes
            #
            # Use the permutations function to get sequential permutations of length MAX_FEATURES
            # Create a new data structure to hold the data for each feature indicated by the permutations.
            # Train the model using the created data structure then predict using the same data structure.
            # Modify the original data structure (image) using what’s been predicted.
            #
            # permutations of integer representations of the features in the image
            perm = permute(list(range(0,len(ivals[0,0]))),False,min(len(ivals[0,0]),const.MAX_FEATURES))
            print(ivals); print(perm)
            perms= []
            for j in range(0,len(perm)):
                if len(perm[j]) == min(len(ivals[0,0]),const.MAX_FEATURES) and \
                   list(perm[j]) == list(np.sort(list(range(min(perm[j]),max(perm[j])+1)))):
                    if j == 0:
                        perms.append(list(perm[j]))
                    else:
                        if not list(perm[j]) in list(perms):
                            perms.append(list(perm[j]))
            print(perms)
            # the number of properties most likely changed
            p    = len(perms[0])
            s    = p
            # the new data structure to hold the image in permutations of length MAX_FEATURES
            #
            # the idea is that the local structure determined by the Markov process and separability
            # allows us to only consider neighbors of each pixel ... as such, we will learn the local
            # structure as determined by the permutations of features and predict the most probable color of each center pixel
            # by considering the probability of each neighboring color ... here, we only create the structure
            # and train the model ... the probabilities will be obtained in a prediction by auto encoding the inputs
            # with the proviso that we are not looking to obtain outputs from the underlying subspace of the inputs
            # rather we will have the same number of outputs as inputs, only the outputs will be probabilities
            # indicating the likelihood of the color of the center pixel based upon the probabilities of colors of its neighbors
            if type(ivals     ) == type(np.asarray([])) and \
               type(ivals[0  ]) == type(np.asarray([])) and \
               type(ivals[0,0]) == type(np.asarray([])):
                nivals  = []
                for i in range(0,len(ivals)):
                    for j in range(0,len(perms)):
                        # length of the input feature space has to be consistent
                        # also, we want only contiguous integer feature columns, e.g. 0,1,2 or 3,4,5, etc.
                        nivals.append(ivals[i][:,perms[j]])
                nivals= np.asarray(nivals)
            else:
                nivals= ivals.copy()
            print(nivals[0]); print([max(nivals.flatten()),len(nivals),len(nivals[0]),len(nivals[0,0])])
            # test if tensorflow is using the GPU or CPU
            dump = None                                      if ver == const.VER else tf.compat.v1.disable_eager_execution()
            cfg  = tf.ConfigProto(log_device_placement=True) if ver == const.VER else tf.compat.v1.ConfigProto(log_device_placement=True)
            sess = tf.Session(config=cfg)                    if ver == const.VER else tf.compat.v1.Session(config=cfg)
            dev  = tf.device(const.DEV)                      if ver == const.VER else tf.compat.v1.device(const.DEV)
            with dev:
                a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
                b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
                c = tf.matmul(a, b)
            print(sess.run(c))
            # don't want to use a linear model for dbnact as that assumes that the
            # data are linearly separable, which results in 2 classes by default
            # as such, the resulting linear subspace will present values fairly close
            # to one another (we are filtering noise after all) and that is not desirable
            # in the long run ... thus we will use the sigmoid instead
            #
            # may also want to try a softmax with categorical cross entropy
            #
            # did some testing with other loss/optimizer/dbnact combinations
            # turns out that softmax and sigmoid in the final layer force
            # normalized values between 0 and 1 for the output
            #
            # returning to linear and will use the greatest changes between values in
            # the smoothed images and the original image as areas to concentrate upon
            # when identifying regions of interest
            #
            # after more testing it looks like a classification configuration will work best
            # in that we will get a probability for each color in each row of pixels which
            # is an indication of which color is more probable ... this is useful as we can
            # relax the image by changing all colors to the most probable one
            #
            # also, we can make use of permuations on each column sequence to make predictions
            # on the whole set of columns using the Markov property and separable processes
            #
            # construct the relaxed image name
            sfl  = const.SFL if hasattr(const,"SFL") else "models/obj.h5"
            #model = dbn(np.asarray([ivals]),np.asarray([ivals]),splits=s,props=p)
            model= dbn(np.asarray([nivals])
                      ,np.asarray([nivals])
                      ,sfl=sfl
                      ,splits=s
                      ,props=p
                      ,loss=const.LOSS
                      ,optimizer=const.OPTI
                      ,rbmact='tanh'
                      ,dbnact='tanh' if ver == const.VER else const.RBMA
                      ,dbnout=p)
            assert(type(model) != type(None))
            pvals= model.predict(np.asarray([nivals]))
            pvals= pvals[0] if not (len(pvals) == 0) else pvals
            # *********************************
            # replace the original image here and compute the differences between
            # the modified original image and the original image to identify
            # the regions of interest
            #
            # also note that by breaking the data up into a smaller number of features and
            # loading the data into the neural netowrk we are capturing the local characteristics
            # guaranteed by the Markov property and encoding those characteristics and changed states
            # for use afterward when other parts of the bounded region undergo the same relaxation process
            # *********************************
            print([pvals.shape,nivals.shape,ivals.shape])
            rivals= ivals.copy()
            def rivals_func(i):
                for j in range(0,len(perms)):
                    for k in range(0,len(rivals[i])):
                        # local receptive fields will be square of size len(perms)
                        # each partition of the image will have len(perms) columns
                        # the partitions are stacked on top of each other in the ordering of the perms
                        # so dividing by the number of images, we have all of the partitions for each image in stacked groupings
                        # we need the right stack and partition for each image to get the right receptive field for replacement
                        idx                  = list(pvals[len(perms)*i+j,k,range(0,len(perms[j]))]).index(max(pvals[len(perms)*i+j,k,range(0,len(perms[j]))]))
                        # copy the most probable color to all pixels in the kth row of the local receptive field
                        rivals[i,k,perms[j]] = np.full(len(perms[j]),ivals[i,k,perms[j][idx]])
                return None
            if nc > 1:
                dump = Parallel(n_jobs=nc)(delayed(rivals_func)(i) for i in range(0,len(rivals)))
            else:
                dump = []
                for i in range(0,len(rivals)):
                    dump.append(rivals_func(i))
            print(pvals[0,0]); print(nivals[0,0]); print(perms); print(rivals); print(rivals.shape)
            # diff the relaxed images with the original images ... numpy makes this easy
            #
            # zero elements will demarcate the boundaries of the regions of interest
            diffa = 2
            if hasattr(const,"DIFFA"):
                diffa= const.DIFFA if int(const.DIFFA) == const.DIFFA else diffa
            diffb = 1
            if hasattr(const,"DIFFB"):
                diffb= const.DIFFB if int(const.DIFFB) == const.DIFFB else diffb
            diffab= abs(diffa*rivals-diffb*ivals)
            diff  = abs(      rivals-      ivals)
            print(diffab); print([max(diffab.flatten()),diffab.shape,rivals.shape,ivals.shape,len(perms),len(perms[0]),blksc,blksr]);
            # obtain the Kullback Leibler divergence between rivals and the original ivals
            ret["kld"] = kld(rivals,ivals)
            # zeros in the diff image will correspond to the boundaries of the regions of interest
            #
            # it will be left to prove mathematically that the boundaries are formed by states at sites that are
            # drawn from the equilibrium distribution ... hence, no change to the states resulting in zero in the diff
            # then the boundary consisting of zeros ... is it also part of the boundary where the phase transition takes place ?
            #
            # each diff image
            z    = []
            for k in range(0,len(diffab)):
                d    = diffab[k]
                dz   = {'x':[],'y':None}
                # each row in a diff image
                for i in range(0,len(d)):
                    # each column in a diff image
                    for j in range(0,len(d[i])):
                        # possible zero sites in the immediate neighborhood of the ctr point
                        pts  = [[i-1,j  ]
                               ,[i+1,j  ]
                               ,[i  ,j-1]
                               ,[i  ,j+1]
                               ,[i-1,j+1]
                               ,[i+1,j+1]
                               ,[i-1,j-1]
                               ,[i+1,j-1]]
                        # track the zeros in the boundary of the regions of interest
                        if d[i,j] != 0 and ((0 <= i-1                                                   and d[i-1,j  ] != 0) or \
                                            (             i+1 < len(d)                                  and d[i+1,j  ] != 0) or \
                                            (                              0 <= j-1                     and d[i  ,j-1] != 0) or \
                                            (                                           j+1 < len(d[i]) and d[i  ,j+1] != 0) or \
                                            (0 <= i-1                  and              j+1 < len(d[i]) and d[i-1,j+1] != 0) or \
                                            (             i+1 < len(d) and              j+1 < len(d[i]) and d[i+1,j+1] != 0) or \
                                            (0 <= i-1                  and 0 <= j-1                     and d[i-1,j-1] != 0) or \
                                            (             i+1 < len(d) and 0 <= j-1                     and d[i+1,j-1] != 0)):
                            # after using random field theory (stochastically separable processes
                            # , local dynamics, conditional specifications)
                            #
                            # we have coordinates of data points that have been essentially projected
                            # into a lower dimensional space defined
                            #
                            # by the permutations stored in perms ... we need to project the coordinates
                            # of each data point back into the original space
                            #
                            # note also that each projected point in the lower dimensional space
                            # is simply a tuple of mostly zeros and a few non-zero
                            #
                            # values with each non-zero value corresponding to coordinates
                            # detailed by a tuple of coordinates in perms
                            #dz['x'].append([(k%const.MAX_ROWS)*(blksr-(0 if km/const.MAX_ROWS==0 else 1))+i,\
                                            #(k%const.MAX_COLS)*(blksc-(0 if kp/const.MAX_COLS==0 else 1))+j])
                            dz['x'].append([(k%const.MAX_ROWS)*blksr+i,(k%const.MAX_COLS)*blksc+j])
                if not (len(dz['x']) == 0):
                    # reorder the coordinates based upon the y-axis
                    dz['y'] = sorted(dz['x'],key=lambda x: x[-1])
                    # append the coordinates of the list of zeros for the current diff image
                    z.append(dz)
            print(z)
            # without going into a lot of detail here, using a result based upon the random cluster model
            # we can estimate the number of classes to form as by assuming that we have N^2 classes containing
            # a total of M^2 data points, uniformly and evenly distributed throughout a bounded uniformly
            # partitioned unit area in the 2D plane
            #
            # then the probability of connection for any 2 data points in the bounded region is given by
            # M^2 / (M^2 + (2M * (N-1)^2)) and by the random cluster model, this probability has to equate
            # to 1/2, which gives a way to solve for N, the minimum number of clusters to form for a
            # data set consistinng of M^2 Gaussian distributed data points projected into 2D space
            #
            # each class contains M^2/N^2 data points ... the sqrt of this number is the size of the kernel we want
            if type(ivals     ) == type(np.asarray([])) and \
               type(ivals[0  ]) == type(np.asarray([])) and \
               type(ivals[0,0]) == type(np.asarray([])):
                kM   = max(2,ceil(sqrt(len(ivals[0,0])/calcN(len(ivals[0,0])))))
            else:
                kM   = max(2,ceil(sqrt(len(ivals[0  ])/calcN(len(ivals[0  ])))))
            # For the regions of interest, only need to look at the matrix coordinates
            # of each of the zeros, as they will demarcate the boundaries of changed pixel intensities.
            #
            # Only need to look at zeros in the diff image
            # as random field theory will guarantee us all connected sites
            # in the boundary surrounding changed pixel intensities by simply looking at contiguous coordinates
            #
            # inner function for inner function for parallel processing
            def inner_inner(pt,ndz,polys):
                r    = rp(pt,ndz,kM)
                # keep only good regions of interest for the current image
                if not (r['poly'] == None):
                    if not (len(r['poly']) == 0):
                        polys.extend(r['poly'])
                        ndz  = [pt for pt in ndz if pt not in polys]
                return r['rp']
            # inner function for parallel processing
            def inner(dz):
                polys= []
                # check that the current pt has not been processed
                ndz  = [pt for pt in dz['x'] if pt not in polys]
                # the region of interest for the current image
                irp  = []
                for pt in ndz:
                    irp.append(inner_inner(pt,ndz,polys))
                return irp
            # For the regions of interest, only need to look at the matrix coordinates
            # of each of the zeros, as they will demarcate the boundaries of changed pixel intensities.
            #
            # Only need to look at zeros in the diff image
            # as random field theory will guarantee us all connected sites
            # in the boundary surrounding changed pixel intensities by simply looking at contiguous coordinates
            rps  = []
            for dz in z:
                rps.append(inner(dz))
            # we only want the good region proposals
            nrps = []
            for r1 in rps:
                nrp  = []
                for r in r1:
                    if not (r['xmin'] == None      or \
                            r['xmax'] == None      or \
                            r['ymin'] == None      or \
                            r['ymax'] == None      or \
                            r['xmin'] == r['xmax'] or \
                            r['ymin'] == r['ymax']):
                        nrp.append(r)
                if not (len(nrp) == 0):
                    nrps.append(nrp)
            rps  = nrps
            print(rps); print([len(z),len(rps)])
            # let's merge region proposals into regions of interest
            #
            # there will be a lot of overlap, as the region proposals are
            # done in parallel, resulting in much duplication / overlap
            nnrps= []
            for r in rps:
                nrps = []
                for r1 in r:
                    ps   = r1.copy()
                    if len(r) > 1:
                        # delete the first element which should correspond to the current region r1
                        del r[0]
                        # check which proposed regions should be merged with the current region
                        lr   = len(r)
                        i    = 0
                        while i < lr:
                            if ((r[i]['xmin'] >= r1['xmin'] and r[i]['xmin'] <= r1['xmax'])   or \
                                (r[i]['xmax'] >= r1['xmin'] and r[i]['xmax'] <= r1['xmax'])   or \
                                (r[i]['xmin'] <= r1['xmin'] and r[i]['xmax'] >= r1['xmax'])   or \
                                (r[i]['xmin'] <= r1['xmax'] and r[i]['xmax'] >= r1['xmax'])) and \
                               ((r[i]['ymin'] >= r1['ymin'] and r[i]['ymin'] <= r1['ymax'])   or \
                                (r[i]['ymax'] >= r1['ymin'] and r[i]['ymax'] <= r1['ymax'])   or \
                                (r[i]['ymin'] <= r1['ymin'] and r[i]['ymax'] >= r1['ymax'])   or \
                                (r[i]['ymin'] <= r1['ymax'] and r[i]['ymax'] >= r1['ymax'])):
                                ps   = {'xmin':max(r[i]['xmin'],ps['xmin']) \
                                       ,'xmax':max(r[i]['xmax'],ps['xmax']) \
                                       ,'ymin':max(r[i]['ymin'],ps['ymin']) \
                                       ,'ymax':max(r[i]['ymax'],ps['ymax'])}
                                # delete the current item and take note in the current (and total) counts
                                del r[i]
                                i    = 0 if i == 0 else i - 1
                                lr  -= 1
                            else:
                                # move to the next item
                                i   += 1
                    # lines with no area (or negative area) are unacceptable region of interest sets of probability measure zero
                    if not (ps['xmax']-ps['xmin'] <= 0 or ps['ymax']-ps['ymin'] <= 0):
                        nrps.append(ps)
                if not (len(nrps) == 0):
                    nnrps.append(nrps)
            ret["roi"] = nnrps
            print(ret["roi"])
            # reconstitute the relaxed input images
            #
            # first do the blocks of rows
            def recon(vals):
                ivals= list(vals)
                if not (blksr <= 1):
                    rblks= int(len(ivals)/blksr)
                    for blk in range(0,rblks):
                        ivals[blk*blksr] = list(ivals[blk*blksr])
                        for i in range(1,blksr):
                            ival = ivals[blk*blksr]
                            ival.extend(ivals[(blk*blksr)+i])
                            if not (i < blksr - 1):
                                if not (km % const.MAX_ROWS == 0):
                                    ival = np.asarray(ival)[range(0,len(ival)-(const.MAX_ROWS-(km%const.MAX_ROWS))),:]
                            ivals[blk*blksr] = ival
                    ival = []
                    for i in range(0,int(len(ivals)/blksr)):
                        if len(ivals) % blksr == 0:
                            ival.append(ivals[i*blksr])
                    ivals= np.asarray(ival)
                # now do the blocks of cols
                ivals= list(ivals)
                if not (blksc <= 1):
                    cblks= int(len(ivals)/blksc)
                    for blk in range(0,cblks):
                        ivals[blk*blksc] = list(ivals[blk*blksc])
                        for i in range(1,blksc):
                            ival = np.append(ivals[blk*blksc],ivals[(blk*blksc)+i],axis=1)
                            if not (i < blksc - 1):
                                if not (kp % const.MAX_COLS == 0):
                                    ival = np.asarray(ival)[:,range(0,len(ival[0])-(const.MAX_COLS-(kp%const.MAX_COLS)))]
                            ivals[blk*blksc] = ival
                    ival = []
                    for i in range(0,int(len(ivals)/blksc)):
                        if len(ivals) % blksc == 0:
                            ival.append(ivals[i*blksc])
                return ival
            # reconstitute the relaxed image
            #
            # need the original outputs of the first level in the network since convolution blurs the original
            # image ... thus, we will subtract the output after the first level from the output after deconvolution
            # to gauge which pixels were changed ... this difference will be subtracted from the original image
            #
            # reconstitute the original relaxed image for comparison
            rivals= np.asarray(recon(rivals))
            rivals= rivals.reshape(np.asarray(rivals.shape)[sorted(range(0,len(rivals.shape)),reverse=True)])
            print(rivals.shape)
            # define the relaxed image
            Image.fromarray(rivals.astype(np.uint8)).save((const.TMP_DIR if hasattr(const,"TMP_DIR") else "/tmp/")+"rivals.jpg")
            # the layers in question
            lyrs  = [l.output for l in np.asarray(model.layers)[[0,len(model.layers)-2]]]
            # the activations at the layers in question
            activs= Model(inputs=model.input,outputs=lyrs)
            # predictions based upon the layers in question
            preds = np.asarray([ivals])
            preds = np.reshape(preds,newshape=preds.shape)
            preds = np.asarray(activs.predict(preds))
            # reconstitute the diff image
            ivals = np.asarray(recon(ivals)).astype(np.uint8)
            ivals = ivals.reshape(np.asarray(ivals.shape)[sorted(range(0,len(ivals.shape)),reverse=True)])
            print(ivals.shape)
            # difference in the predictions of the layers in question to reveal the changed pixels
            preds = np.round(np.asarray(abs(np.asarray(recon(preds[0,0]))-np.asarray(recon(preds[1,0])))))
            preds = preds.reshape(np.asarray(preds.shape)[sorted(range(0,len(preds.shape)),reverse=True)])
            preds = np.where(ivals<=(const.PRED_L_SHIFT if hasattr(const,"PRED_L_SHIFT") else 10),0,preds-(const.PRED_L_SHIFT if hasattr(const,"PRED_L_SHIFT") else 10)      )
            preds = np.where(ivals>=(const.PRED_H_SHIFT if hasattr(const,"PRED_H_SHIFT") else 15),0,      (const.PRED_H_SHIFT if hasattr(const,"PRED_H_SHIFT") else 15)-preds)
            preds = preds.astype(np.uint8)
            # define the relaxed image
            Image.fromarray(ivals).save((const.TMP_DIR if hasattr(const,"TMP_DIR") else "/tmp/")+"ivals.jpg")
            Image.fromarray(preds).save((const.TMP_DIR if hasattr(const,"TMP_DIR") else "/tmp/")+"preds.jpg")
            rivals= ivals - preds
            print([rivals.shape,rivals.all()==ivals.all()])
            # save the relaxed image
            print([rfl,type(rfl)])
            Image.fromarray(rivals).save(rfl)
            if not ro:
                # reconstitute the diff image
                diffab= np.asarray(recon(diffab))
                diffab= diffab.reshape(np.asarray(diffab.shape)[sorted(range(0,len(diffab.shape)),reverse=True)])
                print(diffab.shape)
                # save the diff image
                print([dfl,type(dfl)])
                Image.fromarray(diffab).save(dfl)
                # reconstitute the rgb diff image
                diff  = np.asarray(recon(diff))
                diff  = diff.reshape(np.asarray(diff.shape)[sorted(range(0,len(diff.shape)),reverse=True)])
                print(diff.shape)
                # save the diff image
                print([bfl,type(bfl)])
                Image.fromarray(diff).save(bfl)
                # reconstitute the 3d image
                tivals= restore3d(ivals)
                # save the 3d image
                print([tfl,type(tfl)])
                Image.fromarray(tivals).save(tfl)
                # perform the Shapiro Wilks test on diff ... we want to know if the errors are normal
                ret["swt"] = { fl:stats.shapiro( ivals.flatten()                  ) \
                             ,rfl:stats.shapiro(rivals.flatten()                  ) \
                             ,dfl:stats.shapiro(diffab.flatten()                  ) \
                             ,bfl:stats.kstest(   diff.flatten(),'randint',(0,255))}
            else:
                ret["swt"] = {rfl:stats.shapiro(rivals.flatten()                  )}
    return ret

# *************** TESTING *****************

def nn_testing(M=500,N=2,fl="Almandari-Pre-OP.jpg"):
    # run against the images
    print(nn(fl))
    # number of data points, properties and splits
    m    = M
    p    = N
    if p > const.MAX_FEATURES:
        p    = const.MAX_FEATURES
    #s    = p + 1
    s    = p
    # uniformly sample values between 0 and 1
    #ivals= np.random.sample(size=(500,3))
    ivals= np.random.sample(size=(m,p))
    ovals= categoricals(M,s,p)
    # generate the clustering model for using the test values for training
    model = dbn(ivals,ovals,splits=s,props=p)
    if not (model == None):
        # generate some test data for predicting using the model
        ovals= np.random.sample(size=(int(m/10),p))
        # encode and decode values
        pvals= model.predict(ovals)
        # look at the original values and the predicted values
        print(ovals)
        print(pvals)
    else:
        print("Model 1 is null.")
    ovals= np.random.sample(size=(m,1))
    # generate the regression model for using the test values for training
    #model = dbn(ivals
                #,ovals
                #,splits=s
                #,props=p
                #,loss='mean_squared_error'
                #,optimizer='sgd'
                #,rbmact='sigmoid'
                #,dbnact='linear'
                #,dbnout=1)
    model = dbn(ivals
               ,ovals
               ,splits=s
               ,props=p
               ,loss='mean_squared_error'
               ,optimizer='sgd'
               ,rbmact='tanh'
               ,dbnact='linear'
               ,dbnout=1)
    if not (model == None):
        # generate some test data for predicting using the model
        ovals= np.random.sample(size=(int(m/10),p))
        # encode and decode values
        pvals= model.predict(ovals)
        # look at the original values and the predicted values
        print(ovals)
        print(pvals)
    else:
        print("Model 2 is null.")
    # generate the clustering model for using the test values for training
    # testing models of dimensions > 3
    m    = 50
    p    = 3
    s    = p
    ivals= np.random.sample(size=(m,p))
    # we need values to turn into labels when training
    # one-hot encode the integer labels as its required for the softmax
    ovals= categoricals(m,s,const.MAX_FEATURES)
    model = dbn(ivals,ovals,splits=s,props=p)
    if not (model == None):
        # generate some test data for predicting using the model
        ovals= np.random.sample(size=(int(m/10),p))
        # encode and decode values
        pvals= model.predict(ovals)
        # look at the original values and the predicted values
        print(ovals)
        print(pvals)
    else:
        print("Model 2 is null.")
