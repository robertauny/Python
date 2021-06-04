#!/usr/bin/python

############################################################################
##
## File:      config.py
##
## Purpose:   Read and parse the JSON config file.
##
## Parameter: N/A
##
## Creator:   Robert A. Murphy
##
## Date:      Apr. 1, 2021
##
############################################################################

import json
import os

import constants as const

############################################################################
##
## Purpose:   Read a configuration file
##
############################################################################
def cfg(fl=const.constants.CONFIG):
    ret= None
    if( os.path.exists(fl) and os.stat(fl).st_size > 0 ):
        try:
            with open(fl) as f:
                ret  = json.load(f)
                f.close()
        except:
            ret  = ret.msg
    # return the parsed config to the caller
    return ret

############################################################################
##
## Purpose:   Write a configuration file
##
############################################################################
def wcfg(dat=None,fl=const.constants.CONFIG):
    ret= None
    if( len(dat) > 0 ):
        try:
            with open(fl,'w') as f:
                json.dump(dat,f)
                f.close()
        except:
            ret  = 'Unable to write to ' + dat
    # return the parsed config to the caller as a check
    return ret

############################################################################
##
## Purpose:   Config class
##
############################################################################
class config():
    @staticmethod
    def _cfg(fl=const.constants.CONFIG):
        return cfg(fl)
    @staticmethod
    def _wcfg(fl=const.constants.CONFIG,ofl="/tmp/tempus.json"):
        dat  = config._cfg(fl)
        return wcfg(dat,ofl)

############################################################################
##
## Purpose:   Testing
##
############################################################################
def config_testing():
    cfg  = config()
    ret  = cfg._cfg(const.constants.CONFIG)
    print(ret)
    ret  = cfg._wcfg(const.constants.CONFIG,"/tmp/tempus.json")
    print(ret)
