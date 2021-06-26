#!/usr/bin/python

############################################################################
##
## File:      constants.py
##
## Purpose:   General configuration constants
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Mar. 6, 2021
##
############################################################################

# miscellaneous constants
CONFIG         = "imaging.json"
TEST           = "test.json"
BVAL           = -1
MAX_CLUSTERS   = 3#10
MAX_FEATURES   = 3 #>= 3
MAX_SPLITS     = MAX_FEATURES
MAX_COLS       = MAX_FEATURES
MAX_ROWS       = 10
V              = "vertices"
E              = "edges"
MDL            = "model"
LBL            = "label"
SEP            = "-"
SEED           = 12345
INST           = 0
FILL_COLOR     = 0
ERR            = "ERROR"
VER            = "3.5.2"
DEV_CPU        = "/cpu:0"
DEV_GPU        = "/gpu:0"
DEV            = DEV_CPU
IMGR           = "_relaxed"
IMGD           = "_diff"
IMGB           = "_rgb"
IMG3           = "_3d"
DIFFA          = 2
DIFFB          = 1
IMS_PER_BATCH  = 2#0
LOCAL_DEVS     = ["rUbuntu","WF-1023591-L","nvidia-chris","nvidia-john"]
MASK_INTENSITY = 20
TMP_DIR        = "/tmp/"
XML_JSON_FILE  = "glance.json"
XML_BEG_LOOP   = -1000 #-150 # -100
IMAGE_ROOT     = "/C137_working_dir/data/2020_01_24/images/"
THING_COLORS   = [[255, 255,  25]
                 ,[230,  25,  75]
                 ,[250, 190, 190]
                 ,[ 60, 180,  75]
                 ,[230, 190, 255]
                 ,[  0, 130, 200]
                 ,[245, 130,  48]
                 ,[ 70, 240, 240]
                 ,[210, 245,  60]]
EVAL_TYPE      = None
PMO            = False
RO             = True
CONFIG_YAML    = "/C137_working_dir/notebooks/meshrcnn/meshrcnn/configs/pix3d/meshrcnn_R50_FPN.yaml"
#CONFIG_YAML    = "/C137_working_dir/repos/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
SHIFT          = 200#150
PRED_L_SHIFT   = 10#103
PRED_H_SHIFT   = 11#114
VSIZE          = 8
HOUN_OFFSET    = 10
CPU_COUNT      = 1
# neural network constants
OUTP           = 1
SHAPE          = 3
SPLITS         = 3
PROPS          = 3
LOSS           = "categorical_crossentropy"
OPTI           = "rmsprop"
RBMA           = "softmax"
METRICS        = ["accuracy"]
DBNA           = None
DBNO           = 0
EPO            = 10
EMB            = False
ENCS           = None
USEA           = False
VERB           = 0
BSZ            = 64
LDIM           = 256
VSPLIT         = 0.2
SFL            = "models/lstm.h5"
TRAIN_PCT      = 0.8
MAX_PREDS      = 100#0
BASE_LR        = 0.0002
MAX_ITER       = 30000
DETECTIONS     = 4#1
K              =  [  6.35000000e+01       ,  6.35000000e+01       ,  0.0                     ]
TRANS_MAT      =  [  0.0                  ,  0.0                  ,  6.13496933e-01          ]
ROT_MAT        = [[  0.0                  ,  2.13333333e-01       ,  0.0                     ]
                 ,[  0.0                  ,  0.0                  , -2.13333333e-01          ]
                 ,[ -6.13496933e-04       ,  0.0                  ,  0.0                     ]]
IM             = [[  2.13333333e-01       ,  0.0                  ,  0.0                 ,0.0]
                 ,[  0.0                  ,  2.13333333e-01       ,  0.0                 ,0.0]
                 ,[  0.0                  ,  0.0                  ,  6.13496933e-04      ,0.0]]
DST            = [0.5,0.5,0.5,0.5,0.5]
# chessboard image for camera calibration
CIMG           = "/C137_working_dir/data/2020_01_24/images/chess.jpg"
# default pdfs
PDFS           = ["/home/robert/data/files/kg.pdf"]
# default imgs
IMGS           = ["/home/robert/data/files/IMG_0569.jpeg","/home/robert/data/files/IMG_0570.jpg"]
# global stem variables for some NLP
#
# tokenization
TOKS           = "toks"
# entity extraction
ENTS           = "ents"
# concepts
CONS           = "cons"
# part of speech
POST           = "post"
# stemming
STEM           = "stem"
# all stems but only using vertices for now
STEMS          = [V]
# extension used when storing the knowledge graph files
EXT            = ".xml"
# optical character recognition
OCR            = "ocr"
# image processing
IMG            = "ocri"
# object detection
OBJ            = "objd"
# shape detection
SHP            = "objs"
# shape detection
WIK            = "wiki"
# entity extraction
EE             = "keyPhrases"
# sentiment
SENT           = "sentiment"
