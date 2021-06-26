#!/usr/bin/python

############################################################################
##
## File:      roi.py
##
## Purpose:   Rregion of Interest support used in the Glance app.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Mar. 6, 2021
##
############################################################################

from math                        import inf,ceil,sqrt,floor
from cv2                         import imread                     \
                                       ,cvtColor                   \
                                       ,imwrite                    \
                                       ,COLOR_BGR2GRAY             \
                                       ,IMREAD_GRAYSCALE           \
                                       ,getOptimalNewCameraMatrix  \
                                       ,undistort                  \
                                       ,UMat                       \
                                       ,TERM_CRITERIA_EPS          \
                                       ,TERM_CRITERIA_MAX_ITER     \
                                       ,findChessboardCorners      \
                                       ,cornerSubPix               \
                                       ,calibrateCamera
from joblib                      import Parallel,delayed
from torch                       import from_numpy,float32,save
#from pytorch3d.ops               import cubify
#from pytorch3d.io                import save_obj
from scipy.io                    import savemat,loadmat
from scipy.interpolate           import interp1d

import multiprocessing       as mp
import numpy                 as np
import xml.etree.ElementTree as ET

import os

import constants             as const

np.random.seed(12345)

############################################################################
##
## Purpose:   Define the polygons demarcating the region proposals using random field theory
##
############################################################################
def rp(ctr=[],sites=[],kM=0):
    ret  = {'rp':{'xmin':None,'xmax':None,'ymin':None,'ymax':None},'poly':None}
    if not (len(ctr) == 0 or len(sites) == 0 or kM == 0):
        poly = []
        # possible zero sites in the immediate neighborhood of the ctr point
        pts  = [[ctr[0]-kM,ctr[1]   ]
               ,[ctr[0]+kM,ctr[1]   ]
               ,[ctr[0]   ,ctr[1]-kM]
               ,[ctr[0]   ,ctr[1]+kM]
               ,[ctr[0]-kM,ctr[1]+kM]
               ,[ctr[0]+kM,ctr[1]+kM]
               ,[ctr[0]-kM,ctr[1]-kM]
               ,[ctr[0]+kM,ctr[1]-kM]]
        nsites= sites
        for pt in pts:
            if pt in nsites:
                nsites= [p for p in nsites if not (p == pt or p == ctr)]
                # add the current point to the list demarcating the region proposal
                poly.append(pt)
        # get the max and min of all x and y coordinates of the polygon defining the region proposal
        if not (len(poly) == 0):
            # the final region proposal for Detectron2
            ret['rp']['xmin'] = min(np.asarray(poly)[:,0])
            ret['rp']['xmax'] = max(np.asarray(poly)[:,0])
            ret['rp']['ymin'] = min(np.asarray(poly)[:,1])
            ret['rp']['ymax'] = max(np.asarray(poly)[:,1])
            # all data demarcating the region proposal
            ret['poly']       = poly
    return ret

############################################################################
##
## Purpose:   Construct new polygons defined using random field theory
##
############################################################################
def pts(dat=[]):
    ret  = ""
    if (type(dat) == type([]) or type(dat) == type(np.asarray([]))):
        print(dat)
        if not (len(dat) == 0):
            xmin =  inf
            xmax = -inf
            ymin =  inf
            ymax = -inf
            for i in range(0,len(dat)):
                for j in range(0,len(dat[i])):
                    ret  += "<pt>"
                    ret  += ("<x>" + str(dat[i][j]["xmin"]) + "</x>")
                    ret  += ("<y>" + str(dat[i][j]["ymax"]) + "</y>")
                    ret  += "</pt>"
                    xmin  = dat[i][j]["xmin"] if dat[i][j]["xmin"] <= xmin else xmin
                    xmax  = dat[i][j]["xmax"] if dat[i][j]["xmax"] >= xmax else xmax
                    ymin  = dat[i][j]["ymin"] if dat[i][j]["ymin"] <= ymin else ymin
                    ymax  = dat[i][j]["ymax"] if dat[i][j]["ymax"] >= ymax else ymax
            ret  += ("<xmin>" + str(xmin) + "</xmin>")
            ret  += ("<xmax>" + str(xmax) + "</xmax>")
            ret  += ("<ymin>" + str(ymin) + "</ymin>")
            ret  += ("<ymax>" + str(ymax) + "</ymax>")
    return ret

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
## Purpose:   Calculate the number of clusters to form using random clusters
##
############################################################################
def calcN(pts=None):
    ret = 2**2
    if (pts != None and type(pts) == type(0)):
        M    = ceil(sqrt(pts))
        for i in range(2,const.MAX_FEATURES+1):
            N    = max(floor(sqrt(i)),2)
            if pts/(pts+(2*M*((N-1)^2))) <= 0.5:
                ret  = N**2
            else:
                break
    return ret

############################################################################
##
## Purpose:   Generate a hounsfield unit for each pixel in the image
##
############################################################################
def hounsfield(fl=None):
    ret  = None
    if not (fl is None):
        img  = imread(fl,IMREAD_GRAYSCALE)
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
        kM   = max(2,ceil(sqrt(len(img[0])/calcN(len(img[0])))))
        # return the hounsfield units in a numpy array
        # that has the same shape as the original image
        #
        # the Hounsfield unit is a normalized measure of attenutation of radiation as a result
        # of the tissue, bone, fluid, etc., that is being irradiated.
        # normal values for things like water = 0, air < -1000 (depends on air quality, etc.)
        # allow for the computation of values for other structures like bone, cartilage, muscle, etc.
        # yet, in the computation, we need to know the structure being considered, which won't be known
        # at run-time ... yet all is not lost, as the Hounsfield unit is linear and computed as
        # 1000 * (mean(substance)-mean(water))/(mean(water)-mean(air)) ... where mean(substance)
        # is the mean attenuation of our substance, with similar meaning for water and air
        #
        # since mean(water) = 0 (assumed to be in the center of the scale) and pixel intensities
        # are between 0 and 255 for grayscale images, then we can use intensity 128 for water
        #
        # now, air causes less attenuation so that we can use intensity 192 for air ... we could
        # use 255, but there are other factors resulting in slightly more attenuation in air, so 192 should
        # be good to use as the mean ... likewise, darker structures indicate more attenuation which
        # will be indicated by values between 0 and 128 on the pixel intensity scale
        #
        # this will form the basis of our pseudo Hounsfield unit in our voxel
        ret  = np.full(img.shape,0)
        if img.shape[0] > kM and img.shape[1] > kM:
            for i in range(0,img.shape[0]):
                if i <= img.shape[0]-kM:
                    for j in range(0,img.shape[1]):
                        if j <= img.shape[1]-kM:
                            ret[i,j] = 1000 * ((np.mean(img[i:(i+kM),j:(j+kM)].flatten())-128)/(128-192))
                        else:
                            # because of the kM x kM window being used to compute the pseudo Hounsfield unit
                            # we won't be able to directly compute values for the remaining columns ... however,
                            # the values for the remaining columns are highly correlated to the values of the
                            # last column for which we have a value as a direct result of our Gaussian random field
                            ret[i,j] = ret[i,img.shape[1]-kM]
                else:
                    # because of the kM x kM window being used to compute the pseudo Hounsfield unit
                    # we won't be able to directly compute values for the remaining rows ... however,
                    # the values for the remaining rows are highly correlated to the values of the
                    # last row for which we have values as a direct result of our Gaussian random field
                    ret[i,:] = ret[img.shape[0]-kM,:]
        ret  = ret
    return ret

############################################################################
##
## Purpose:   Create the XML output file for the new images obtained
##
############################################################################
def xmlf(ofl=None,fl=None,dat=None):
    ret  = None
    if not (ofl == None or fl == None or dat == None):
        if (os.path.exists(ofl) and os.path.getsize(ofl) > 0) and \
           (os.path.exists( fl) and os.path.getsize( fl) > 0):
            # idea is to look at the original region of interest
            # that is included in the annotations ... then augment
            # the original region with extensions that are predicted
            # using random field theory
            if (type(dat) == type([]) or type(dat) == type(np.asarray([]))):
                if not (len(dat) == 0):
                    d    = dat.copy()
                    tree = ET.parse(ofl)
                    root = tree.getroot()
                    ndat = []
                    objn = []
                    idn  = []
                    user = []
                    date = []
                    # the hounsfield unit for every pixel and the offset
                    # that determines the range of equivalence for other
                    # regions of interest
                    houns= np.nan_to_num(hounsfield(fl),posinf=1000.0,neginf=1000.0)
                    houn = const.HOUN_OFFSET if hasattr(const,"HOUN_OFFSET") else 10
                    # for every original region of interest
                    for obj in root.findall('object'):
                        name = obj.find('name').text
                        if name.isnumeric():
                            continue
                        objn.extend([name])
                        idn.extend( [int( idn.text) for idn  in  obj.findall('id'      )])
                        date.extend([    date.text  for date in  obj.findall('date'    )])
                        for poly in obj.findall('polygon'):
                            user.extend([user.text  for user in poly.findall('username')])
                            allx = []
                            ally = []
                            for pt in poly.findall('pt'):
                                allx.append(int(pt.find('x').text))
                                ally.append(int(pt.find('y').text))
                                xmin = min(allx)
                                xmax = max(allx)
                                ymin = min(ally)
                                ymax = max(ally)
                                # the hounsfield units for each pixel in the bounding box of the region of interest
                                ohoun= np.nan_to_num(np.median(np.asarray(houns)[xmin:xmax,ymin:ymax].flatten())
                                                    ,posinf=1000.0
                                                    ,neginf=1000.0)
                                # look at the predicted regions and augment the original region
                                #
                                # note, we are defining our dictionary in this way to preserve
                                # a particular order for comparison to see if we already have
                                # augmented our region with a prediction
                                ndat.append([{"xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax}])
                                # list to hold the regions that have already been added
                                drmv = []
                                for i in range(0,len(d)):
                                    for j in range(0,len(d[i])):
                                        #if (xmin >= d[i][j]["xmin"]  and
                                           # xmin <= d[i][j]["xmax"])  or \
                                           #(xmax >= d[i][j]["xmin"]  and
                                           # xmax <= d[i][j]["xmax"])  or \
                                           #(ymin >= d[i][j]["ymin"]  and
                                           # ymin <= d[i][j]["ymax"])  or \
                                           #(ymax >= d[i][j]["ymin"]  and
                                           # ymax <= d[i][j]["ymax"])  or \
                                        #if (d[i][j]["xmin"] >= xmin  and
                                            #d[i][j]["xmin"] <= xmax) and \
                                           #(d[i][j]["xmax"] >= xmin  and
                                            #d[i][j]["xmax"] <= xmax) and \
                                           #(d[i][j]["ymin"] >= ymin  and
                                            #d[i][j]["ymin"] <= ymax) and \
                                           #(d[i][j]["ymax"] >= ymin  and
                                            #d[i][j]["ymax"] <= ymax):
                                        if (d[i][j]["xmin"] >= xmin  and
                                            d[i][j]["xmin"] <= xmax)  or \
                                           (d[i][j]["xmax"] >= xmin  and
                                            d[i][j]["xmax"] <= xmax)  or \
                                           (d[i][j]["ymin"] >= ymin  and
                                            d[i][j]["ymin"] <= ymax)  or \
                                           (d[i][j]["ymax"] >= ymin  and
                                            d[i][j]["ymax"] <= ymax):
                                            # the idea here ... only add to the region of interest
                                            # if the median hounsfield unit for the region to be added
                                            # is within a small range of the median hounsfield unit
                                            # of the original bounded box of the region of interest
                                            nhoun=np.nan_to_num(np.median(np.asarray(houns)[d[i][j]["xmin"]:d[i][j]["xmax"]
                                                                                           ,d[i][j]["ymin"]:d[i][j]["ymax"]]
                                                                                           .flatten())
                                                               ,posinf= 1000.0
                                                               ,neginf=-1000.0)
                                            if nhoun in range(int(floor(ohoun-houn)),int(ceil(ohoun+houn))):
                                                # if the prediction is not in our augmentations
                                                # then it is ok to add it now
                                                ndat.append(d[i])
                                                # remove the current one because we don't want dups
                                                drmv.append(i)
                                                break
                                # remove the regions indicated to prevent dups
                                d    = [d[i] for i in range(0,len(d)) if i not in drmv]
                    # add the final augmentations and write the xml file for the new image
                    pdat = pts(ndat)
                    if not (len(pdat) == 0):
                        ndat = ET.fromstring("<object>"                                          + \
                                                 "<name>"        +objn[0]         +"</name>"     + \
                                                 "<deleted>0</deleted>"                          + \
                                                 "<verified>0</verified>"                        + \
                                                 "<verifieduser />"                              + \
                                                 "<occluded />"                                  + \
                                                 "<attributes />"                                + \
                                                 "<parts><hasparts /><ispartof /></parts>"       + \
                                                 "<date>"        +unique(date)[0] +"</date>"     + \
                                                 "<id>"          +str(max(idn)+1) +"</id>"       + \
                                                 "<polygon>"                                     + \
                                                     "<username>Random Field Theory</username>"  + \
                                                     pdat                                        + \
                                                 "</polygon>"                                    + \
                                            "</object>")
                        root.append(ndat)
                        for p in root:
                            if p.tag == "filename":
                                p.text =  fl[: fl.rfind(".")]                                   + p.text[p.text.rfind("."):]
                                ret    = ofl[:ofl.rfind("/")] + fl[fl.rfind("/"):fl.rfind(".")] +    ofl[   ofl.rfind("."):]
                                with open(ret,"w") as f1:
                                    f1.write(ET.tostring(root,encoding="unicode"))
                                break
    return ret

############################################################################
##
## Purpose:   Create the final regions of interest from the intermediate ones
##
############################################################################
def roif(dat=None):
    ret  = []
    if not (dat == None):
        if (type(dat) == type([]) or type(dat) == type(np.asarray([]))):
            if not (len(dat) == 0):
                # collection of region proposals
                for d in dat:
                    xmin =  inf
                    xmax = -inf
                    ymin =  inf
                    ymax = -inf
                    # collection of subregion proposals ... merge
                    for r in d:
                        xmin = r["xmin"] if r["xmin"] < xmin else xmin
                        ymin = r["ymin"] if r["ymin"] < ymin else ymin
                        xmax = r["xmax"] if r["xmax"] > xmax else xmax
                        ymax = r["ymax"] if r["ymax"] > ymax else ymax
                    ret.extend([[xmin,ymax],[xmax,ymin]])
                # For the interpolation, we can apply a canonical ordering on the data points comprising the regions of interest
                # and find a monotone function, which by definition is order preserving, so that we can always map data points
                # to their associated data points in other layers.
                # In essence, choosing the ordering, e.g. x_1 <= x_2 <= x_3 <= ... <= x_n,
                # doesn't change the region of interest nor the location of any of the points in its boundary.
                # The ordering only changes the starting point and ending points defining the boundary of the regions of interest.
                # Thus, we will always know how data points in the boundary of the region of interest in one slice are mapped to
                # the associated boundary in the next slice, allowing the regions to be tracked through slices.
                ret  = unique(sorted(ret))
    return ret

############################################################################
##
## Purpose:   Generate a mask for the region of interest
##
############################################################################
def mask(fl=None,bbox=None,poly=None,ann=None):
    ret  = None
    if not (fl is None or bbox is None or poly is None or ann is None):
        # get the file name for outputting the tmp image
        ret  = fl[(fl.rfind("/")+1):fl.rfind(".")] + "_" + str(ann) + fl[fl.rfind("."):]
        ret  = (const.TMP_DIR if hasattr(const,"TMP_DIR") else "/tmp/") + ret
        # open the image and read the data
        # image data is immutable so make a copy
        img  = imread(fl,IMREAD_GRAYSCALE)
        # mathematically, what we want to accomplish is to find the interior
        # of a bounded region ... from a computer science perspective, all we
        # want is to highlight the region of interest, setting it apart from
        # the remainder of the image ... this can be done by shifting pixel
        # intensities ... everything in the interior of the region will be
        # white and all other pixel intensities will be black
        #
        # first set all intensities outside the bounding box to black
        #
        # easily, for each pixel coordinate, if its x-coord <= xmin, or
        # x-coord >= xmax, or y-coord <= ymin, or y-coord >= ymax, then
        # change the intensity of that pixel to 0 (black)
        #
        # support functions
        def l(m,v,c):
            # if we are outside the region of interest then change the intensity
            # otherwise just return what we originally were passed
            if True in v:
                m[v*len(m)] = c
            return m
        def o(m,v,c,p):
            # if we are inside the region of interest then change the intensity
            # black out everything first, then find a range if it exists to
            # make everything else white ... a mask is like a negative image
            if True in v:
                om   = m.copy()
                m[range(0,len(m))] = np.full(len(m),0)
                if not (len(p) == 0):
                    rng    = range(min(p),max(p))
                    m[rng] = np.full(len(rng),c)
            return m
        # now perform the masking
        nc   = const.CPU_COUNT if hasattr(const,"CPU_COUNT") else mp.cpu_count()
        if nc > 1:
            if bbox["xmax"]-bbox["xmin"] > bbox["ymax"]-bbox["ymin"]:
                # rows above and below the bounding box
                img  = np.asarray(img)
                img  = Parallel(n_jobs=nc)(delayed(l)(img[i,:]
                                                     ,[(i<=bbox["ymin"])or(i>=bbox["ymax"])]
                                                     ,0  ) for i in range(0,len(img   )))
                # rows intersecting the bounding box
                img  = np.asarray(img)
                img  = Parallel(n_jobs=nc)(delayed(o)(img[i,:]
                                                     ,[(i>bbox["ymin"])and(i<bbox["ymax"])]
                                                     ,255
                                                     ,[poly[j][1] for j in range(0,len(poly)) if poly[j][0] == i]) for i in range(0,len(img   )))
            else:
                # columns left and right of the bounding box
                img  = np.asarray(img)
                img  = Parallel(n_jobs=nc)(delayed(l)(img[:,i]
                                                     ,[(i<=bbox["xmin"])or(i>=bbox["xmax"])]
                                                     ,0  ) for i in range(0,len(img[0])))
                # columns intersecting the bounding box
                img  = np.asarray(img)
                img  = Parallel(n_jobs=nc)(delayed(o)(img[:,i]
                                                     ,[(i>bbox["xmin"])and(i<bbox["xmax"])]
                                                     ,255
                                                     ,[poly[j][0] for j in range(0,len(poly)) if poly[j][1] == i]) for i in range(0,len(img[0])))
        else:
            if bbox["xmax"]-bbox["xmin"] > bbox["ymax"]-bbox["ymin"]:
                # rows above and below the bounding box
                img  = np.asarray(img)
                for i in range(0,len(img   )):
                    img[i,:] = l(img[i,:]
                                ,[(i<=bbox["ymin"])or(i>=bbox["ymax"])]
                                ,0  )
                # rows intersecting the bounding box
                img  = np.asarray(img)
                for i in range(0,len(img   )):
                    img[i,:] = o(img[i,:]
                                ,[(i>bbox["ymin"])and(i<bbox["ymax"])]
                                ,255
                                ,[poly[j][1] for j in range(0,len(poly)) if poly[j][0] == i])
            else:
                # columns left and right of the bounding box
                img  = np.asarray(img)
                for i in range(0,len(img[0])):
                    img[:,i] = l(img[:,i]
                                ,[(i<=bbox["xmin"])or(i>=bbox["xmax"])]
                                ,0  )
                # columns intersecting the bounding box
                img  = np.asarray(img)
                for i in range(0,len(img[0])):
                    img[:,i] = o(img[:,i]
                                ,[(i>bbox["xmin"])and(i<bbox["xmax"])]
                                ,255
                                ,[poly[j][0] for j in range(0,len(poly)) if poly[j][1] == i])
        # save the mask
        s    = const.SHIFT if hasattr(const,"SHIFT") else 128
        img  = np.asarray(img)
        for i in range(0,len(img)):
            for j in range(0,len(img[i])):
                img[i,j] = 0 if img[i,j] <= s else 255
        imwrite(ret,np.asarray(img))
    return ret

############################################################################
##
## Purpose:   Generate a voxel (higher dimensional pixel) for each 2D pixel
##
############################################################################
def voxel(fl=None,bbox=None,poly=None):
    ret  = None
    if not (fl is None or bbox is None or poly is None):
        ret  = {"vox":None,"obj":None,"voxel":None}
        sz   = list(imread(fl).shape)
        sz.reverse()
        imgs = np.full(sz,0)
        # for these purposes, each part of RGB is treated as an image
        for img in imgs:
            # rows intersecting the bounding box occupy the mesh
            for i in range(0,len(img   )):
                if i > bbox["ymin"] and i < bbox["ymax"]:
                    cols        = []
                    cols.extend(range(bbox["xmin"]+1,bbox["xmax"]))
                    img[i,cols] = np.full(len(cols),1)
            # columns intersecting the bounding box occupy the mesh
            for i in range(0,len(img[0])):
                if i > bbox["xmin"] and i < bbox["xmax"]:
                    rows        = []
                    rows.extend(range(bbox["ymin"]+1,bbox["ymax"]))
                    img[rows,i] = np.full(len(rows),1)
        # file name before extension
        fln        = os.path.join((const.TMP_DIR if hasattr(const,"TMP_DIR") else "/tmp/")+fl[(fl.rfind("/")+1):fl.rfind(".")])
        # create and save a numpy array as our voxel for this image
        #imgs.flags.writeable = True
        vox        = from_numpy(imgs).to(dtype=float32,device="cpu")
        vox        = vox.unsqueeze(0)
        ret["vox"] = fln + ".vox.zip"
        save(vox,ret["vox"])
        # create and save the mesh
        try:
            # the model used at run time
            #mesh       = cubify(vox,0.5)
            ret["obj"] = fln + ".obj"
            #save_obj(ret["obj"],mesh.verts_list()[0],mesh.faces_list()[0])
        except:
            print(fl)
        # this will be the voxel used at run time
        #
        # first create a mask from the original image
        flm  = mask(fl,bbox,poly)
        # read the mask that was just created and
        # convert all intensities to probabilities
        # this is just a matter of substituting 1 for 255
        img  = imread(flm,IMREAD_GRAYSCALE)
        img  = np.asarray(imgs)
        # voxel size and empty creation
        v    = const.VSIZE if   hasattr(const,"VSIZE")          and \
                                const.VSIZE > 0                 and \
                                img.shape[0] % const.VSIZE == 0 and \
                                img.shape[1] % const.VSIZE == 0     \
                           else 8
        vox  = np.full((v,v,v),0)
        # divide the image space into low resolution areas
        # each of which being occupied if it contains a non-black pixel
        for i in range(0,int(len(img)/v)):
            for j in range(0,int(len(img[i])/v)):
                vox[i,j] = 1 if max(img[(i*v):((i+1)*v),(j*v):((j+1)*v)].flatten()) > 0 else 0
        ret["mat"] = fln + ".mat"
        savemat(ret["mat"],{"voxel":vox})
    return ret

############################################################################
##
## Purpose:   3D reconstruction and crop for original ROI, as black pixels added
##
############################################################################
def restore3d(img=None,crop=False):
    ret  = None
    # camera intrinsic matrix
    c    = [[  2.13333333e-01       ,  0.0                  ,  0.0                 ,0.0]
           ,[  0.0                  ,  2.13333333e-01       ,  0.0                 ,0.0]
           ,[  0.0                  ,  0.0                  ,  6.13496933e-04      ,0.0]]
    c    = UMat(np.asarray((const.IM if hasattr(const,"IM") else c),dtype=np.uint8)[:min(len(c),3),:min(len(c[0]),3)])
    if not (img is None or len(img) == 0 or len(img[0]) == 0):
        # termination criteria
        crit = (TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER,30,0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        #
        # these are the chessboard inner corners
        objp = np.zeros((6*7,3),np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        # use the chessboard image to obtain the instrinsic matrix
        # and the distortion matrix coefficients for recovery of the 3D image
        cimg = const.CIMG if hasattr(const,"CIMG") else "/C137_working_dir/data/2020_01_24/images/chess.jpg"
        #assert os.path.exists(cimg) and os.path.getsize(cimg)
        if os.path.exists(cimg) and os.path.getsize(cimg) > 0:
            gray = imread(cimg,IMREAD_GRAYSCALE)
            # Find the chess board corners
            r,corn = findChessboardCorners(gray,(7,6),None)
            # If found, add object points, image points (after refining them)
            if r == True:
                cornerSubPix(gray,corn,(11,11),(-1,-1),crit)
                # mtx is the camera intrinsic matrix to use for this processing
                #
                # we could use our intrinsic matrix "c" if we knew the distortion coefficients
                # however, we will always attempt a recovery of each image using the same
                # methodology, which should be good enough, so long as the same methodology is
                # used for all images
                #
                # "r" in the next line is not the same as "r" that follows
                r,mtx,d,rvecs,tvecs = calibrateCamera([objp],[corn],gray.shape[::-1],None,None)
                # only want the first 2 coordinates of the shape for processing
                h,w  = img.shape[:2]
                # new camera intrinsic matrix is c ... roi is r
                nc,r = getOptimalNewCameraMatrix(mtx,d,(w,h),1)
                # recover the 3D image
                ret  = undistort(img,mtx,d,None,nc)
                # crop the image if requested
                if crop:
                    ret  = ret[r[1]:(r[1]+h),r[0]:(r[0]+w)]
        else:
            ret  = img
    return ret
