#!/usr/bin/env python
# coding: utf-8
"""
    Change hard-coded amisim files you wish to convert to mirage-y (MAST) format
"""
"""
    read in amisim cube of data, 2D or 3D cube
    graft mirage-like headers on the data part
"""

import glob
import os, sys, time
from astropy.io import fits
import numpy as np

print(sys.argv[0])

######## EDIT ME
from pathlib import Path
datadir = str(Path.home())+"/Downloads/asoulain_arch2019.12.07/Simulated_data/"
amisimfns = ("t_dsk_100mas__F430M_81_flat_x11__00",  # Target first
             "c_dsk_100mas__F430M_81_flat_x11__00",  # One calibrator (for now)
             )

######## END EDIT
mirext = "_mir"
mirexample = str(Path.home()) + \
             "/gitsrc/ImPlaneIA/example_data/" + \
             "/jw00793001001_01101_00001_nis_cal.fits"


for fname in amisimfns:
    fobj_sim = fits.open(datadir+fname+".fits")
    #rint(fobj_sim[0].data.shape)
    if len(fobj_sim[0].data.shape) == 2:    #make single slice of data into a 1 slice cube
        d = np.zeros((1, fobj_sim[0].data.shape[0], fobj_sim[0].data.shape[1]))
        d[0,:,:] = fobj_sim[0].data
        fobj_sim[0].data = d

    mirobj = fits.open(mirexample) # read in sample mirage file
    
    # make cube of data for mirage from input ami_sim file... even a 1-deep cube.
    mirobj[1].data = np.zeros((fobj_sim[0].data.shape[0], #"slices of cube of integrations"
                                 mirobj[1].data.shape[0], 
                                 mirobj[1].data.shape[1]))
    
    mirobj[1].data = mirobj[1].data.astype(np.float32)
    mirobj[1].data = fobj_sim[0].data # replace with ami_sim data
    mirobj[1].data = mirobj[1].data[:,:80,:80]

    print("    TARGNAME =", mirobj[0].header["TARGNAME"], 
             " Output cube shape", mirobj[1].data.shape)
    mirobj.writeto(datadir+fname+mirext+".fits", overwrite=True)
