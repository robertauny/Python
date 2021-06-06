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
from matplotlib.pyplot           import savefig#imread,savefig
from cv2                         import imread
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
    # the idea is to perform blind image deconvolution where we don't have the
    # reference image to use during supervised learning so we have to improvise
    #
    # we will make an assumption that the noise resulting in image blur is the result of
    # a series of convolutions at different scales, where the scales are determined by
    # the size of the convolution kernel.
    #
    # to recover an estimate of the original image, we perform a series of deconvolutions
    # at different scales, followed by a deep belief network consisting of a sequence of
    # restricted Boltzmann machines for stochastically relaxation and noise resolution,
    # then follow that with a series of convolutional layers in the reverse scale used in
    # the deconvolutions
    #
    # the result is an auto-encoder that will attempt to recover the original blurred image
    # after which the output of the layer preceding the final sequence of convolutions can
    # be obtained as an estimate of the original image
    #
    # just to wrap up the exposition, we deconvolve the blurred image, stochastically relax
    # to remove other noise and blurring effects (this is the estimate of the original image)
    # and we need to know how close the estimate is to the original image (that we don't have)
    # thus, we apply convolutions in the reverse scale order as we deconvolve and test the resulting
    # estimate of the blurred image against the original blurred image ... if we are close, then
    # the estimate of the original image must be close to the original image
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
            # deconvolution recovers a representation of the original values as samples
            # from an embedded subspace of the noisy sample space
            #
            # we will deconvolve beginning with the lowest resolution to the highest resolution
            # then test the estimate of the de-blurred image against the original blurred image
            # by convolving the de-blurred estimate in an auto-encoder setup
            a    = list(range(kM,(const.CONVS if hasattr(const,"CONVS") and const.CONVS >= kM else 10+kM)))
            for i in a:
                if ver == const.VER:
                    enc  = Conv2DTranspose(nb_filter=M,nb_row=i,nb_col=i,input_shape=ip[0].shape,subsample=1,activation='tanh',border_mode='same',init='random_normal')
                else:
                    enc  = Conv2DTranspose(filters=M,kernel_size=i,input_shape=ip[0].shape,strides=1,activation='selu',padding='same',kernel_initializer='random_normal')
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
                # we will convolve beginning with the highest resolution to the lowest resolution
                # then test the estimate of the blurred image against the original blurred image
                # in an auto-encoder setup
                a    = list(range(kM,(const.CONVS if hasattr(const,"CONVS") and const.CONVS >= kM else 10+kM)))
                a.reverse()
                for i in a:
                    if ver == const.VER:
                        enc  = Conv2D(nb_filter=M,nb_row=i,nb_col=i,input_shape=ip[0].shape,subsample=1,activation='tanh',border_mode='same',init='random_normal')
                    else:
                        enc  = Conv2D(filters=M,kernel_size=i,input_shape=ip[0].shape,strides=1,activation='selu',padding='same',kernel_initializer='random_normal')
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
            sfl  = const.SFL if hasattr(const,"SFL") else "models/dconv.h5"
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
            # also note that by breaking the data up into a smaller number of features and
            # loading the data into the neural netowrk we are capturing the local characteristics
            # guaranteed by the Markov property and encoding those characteristics and changed states
            # for use afterward when other parts of the bounded region undergo the same relaxation process
            # *********************************
            print([pvals.shape,nivals.shape,ivals.shape])
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
            kM   = max(2,ceil(sqrt(len(ivals[0,0])/calcN(len(ivals[0,0])))))
            a    = list(range(kM,(const.CONVS if hasattr(const,"CONVS") and const.CONVS >= kM else 10+kM)))
            # the layers in question
            lyrs  = np.asarray(model.layers)[len(model.layers)-(len(a)+1)]
            # the activations at the layers in question
            activs= Model(inputs=model.input,outputs=lyrs.output)
            # predictions based upon the layers in question
            preds = np.asarray([ivals])
            preds = np.reshape(preds,newshape=preds.shape)
            preds = np.asarray(activs.predict(preds))
            # we will take the estimate of the original image after de-blurring, which we captured from the model
            # and use random field theory to perform an annealing process to find its associated equilibrium distribution
            rivals= preds[0].copy()
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
                        rivals[i,k,perms[j]] = np.full(len(perms[j]),preds[0][i,k,perms[j][idx]])
                return None
            if nc > 1:
                dump = Parallel(n_jobs=nc)(delayed(rivals_func)(i) for i in range(0,len(rivals)))
            else:
                dump = []
                for i in range(0,len(rivals)):
                    dump.append(rivals_func(i))
            print(pvals[0,0]); print(nivals[0,0]); print(perms); print(rivals); print(rivals.shape)
            # obtain the Kullback Leibler divergence between rivals and the original ivals
            ret["kld"] = kld(rivals,preds[0])
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
            # reconstitute the original relaxed (de-blurred) image for comparison
            rivals= np.asarray(recon(rivals))
            rivals= rivals.reshape(np.asarray(rivals.shape)[sorted(range(0,len(rivals.shape)),reverse=True)])
            print(rivals.shape)
            Image.fromarray(rivals.astype(np.uint8)).save(rfl)
            ret["swt"] = {rfl:stats.shapiro(rivals.flatten())}
            # reconstitute the original blurred image
            ivals = np.asarray(recon(ivals))
            ivals = ivals.reshape(np.asarray(ivals.shape)[sorted(range(0,len(ivals.shape)),reverse=True)])
            print(ivals.shape)
            Image.fromarray(ivals.astype(np.uint8)).save((const.TMP_DIR if hasattr(const,"TMP_DIR") else "/tmp/")+"ivals.jpg")
    return ret

# *************** TESTING *****************

def nn_testing(M=500,N=2,fl="Almandari-Pre-OP.jpg"):
    # run against the images
    nn(fl)
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
