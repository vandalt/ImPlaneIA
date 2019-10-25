#!/usr/bin/env python
# coding: utf-8


import glob
import os, sys, time
from astropy.io import fits
import numpy as np

datadir = "../example_data/noise/"
tr = "t_disk_small2_0__PSF_MASK_NRM_F430M_x11_0.82_ref__00"  # root name target
cr =       "c_disk3_4__PSF_MASK_NRM_F430M_x11_0.82_ref__00"  # root name calibrator
mirext = "_mir"

mirexample = "jw00793001001_01101_00001_nis_cal.fits"


for fname in (tr, cr):
    fobj = fits.open(datadir+fname+".fits")
    mirobj = fits.open(datadir+mirexample) # read in sample mirage file
    print(mirobj[1].data.shape)
    mirobj[1].data[:,:] = 0.0 # zero out mirage scihdr data
    print(mirobj[1].data.shape)

    mirobj[1].data[:77,:77] = fobj[0].data[0,:,:] # replace with ami_sim data
    print(mirobj[1].data.shape)
    print(fobj[0].data.shape)
    d = fobj[0].data
    print(d.shape, type(d))

    samples = d[0,73:,:]
    print(samples.shape, samples.mean())

    #import pdb; pdb.set_trace()
    mirobj[1].data[77:,:] = samples.mean()
    mirobj[1].data[:,77:] = samples.mean()

    mirobj.writeto(datadir+fname+mirext+".fits", overwrite=True)
    



