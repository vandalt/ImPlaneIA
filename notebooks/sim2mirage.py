#!/usr/bin/env python
# coding: utf-8


import glob
import os, sys, time
from astropy.io import fits
import numpy as np

datadir = "../example_data/noise/"
filt1 = "PSF_MASK_NRM_F430M_x11_0.82_ref_det"  # root name target
mirext = "_mir"


datadir = "simulatedpsfs/"

mirexample = "jw00793001001_01101_00001_nis_cal.fits"


for fname in (filt1,):
    fobj_sim = fits.open(datadir+fname+".fits")
    print(fobj_sim[0].data.shape)
    if len(fobj_sim[0].data.shape) == 2:    #make single slice of data into a 1 slice cube
        d = np.zeros((1, fobj_sim[0].data.shape[0], fobj_sim[0].data.shape[1]))
        d[0,:,:] = fobj_sim[0].data
        fobj_sim[0].data = d

    mirobj = fits.open(datadir+mirexample) # read in sample mirage file
    #import pdb;pdb.set_trace()
    
    # make cube of data for mirage from input ami_sim file...
    mirobj[1].data = np.zeros((fobj_sim[0].data.shape[0], #"slices of data cube of integrations"
                                     mirobj[1].data.shape[0], 
                                     mirobj[1].data.shape[1]))
    
    mirobj[1].data = mirobj[1].data.astype(np.float32)
    mirobj[1].data = fobj_sim[0].data # replace with ami_sim data
    mirobj[1].data = mirobj[1].data[:,:80,:80]
    print(mirobj[1].data.shape)

    mirobj.writeto(datadir+fname+mirext+".fits", overwrite=True)
    



