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
## Date:      Apr. 1, 2021
##
############################################################################

class constants():
    # miscellaneous constants
    CONFIG         = "pseudo.json"
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
    DIFFA          = 2
    DIFFB          = 1
    IMS_PER_BATCH  = 2#0
    MASK_INTENSITY = 20
    TMP_DIR        = "/tmp/"
    EVAL_TYPE      = None
    PMO            = False
    RO             = True
    SHIFT          = 200#150
    PRED_L_SHIFT   = 10
    PRED_H_SHIFT   = 15
    VSIZE          = 8
    HOUN_OFFSET    = 10
    CPU_COUNT      = 1
    TARGETS        = ["REASONCODE"]
    DATES          = ["DATE"]
    DROP           = ["Id"
                     ,"PATIENT"
                     ,"START"
                     ,"STOP"
                     ,"PAYER"
                     ,"CODE"
                     ,"SSN"
                     ,"BIRTHPLACE"
                     ,"ADDRESS"
                     ,"CITY"
                     ,"STATE"
                     ,"COUNTY"
                     ,"ZIP"
                     ,"LAT"
                     ,"LON"
                     ,"DRIVERS"
                     ,"PASSPORT"
                     ,"PREFIX"
                     ,"FIRST"
                     ,"LAST"
                     ,"SUFFIX"
                     ,"MAIDEN"
                     ,"PAYER_COVERAGE"
                     ,"DISPENSES"
                     ,"TOTALCOST"
                     ,"ENCOUNTER"
                     ,"CODE"
                     ,"DESCRIPTION"
                     ,"BASE_COST"
                     ,"BIRTHDATE"
                     ,"DEATHDATE"
                     ,"REASONDESCRIPTION"]
    MATCH_ON       = ["Id","PATIENT"]
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
    SFL            = "models/pseudo.h5"
    TRAIN_PCT      = 0.8
    MAX_PREDS      = 100#0
    BASE_LR        = 0.0002
    MAX_ITER       = 30000
