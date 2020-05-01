#!/usr/bin/env python
# coding: utf-8
"""
    Convert hard-coded [ami]sim files to mirage-y (MAST) format
"""
"""
    Read in [ami]sim cube of data, 2D or 3D cube and graft mirage-like headers on the data part, 
    Also cube the data if incoming data is 2D.
    Input files are in the list variable simfile, so edit this as needed.
    MIRAGE template file is in the string variable mirexample
"""

import glob
import os, sys, time
from astropy.io import fits
import numpy as np

def amisim2mirage(datadir, amisimfns, mirexample, filt):
    """
    datadir: where input amisim files (cubes or 2d) are located
    amisimfns: list or tuple of simultaed data files
    mirexample: fullpath/filename of mirage example to use for header values
    *** WARNING *** sim2mirage.py:
        The MIRAGE fits file that provides the structure to wrap your simulated
        data to look like MAST data is for the F480M filter.  If you use it to
        convert any  other other filter's simulated data into mirage format, change
        this with e.g., 'mirobj[0].header["FILTER"] = "F430M"' or other filter used
        to create the simulated data.
    """
    mirext = "_mir"
    for fname in amisimfns:
        print(fname+':', end='')
        fobj_sim = fits.open(datadir+fname+".fits")
        if len(fobj_sim[0].data.shape) == 2:    #make single slice of data into a 1 slice cube
            d = np.zeros((1, fobj_sim[0].data.shape[0], fobj_sim[0].data.shape[1]))
            d[0,:,:] = fobj_sim[0].data
            fobj_sim[0].data = d

        try:
            mirobj = fits.open(mirexample) # read in sample mirage file
        except FileNotFoundError:
            sys.exit("*** ERROR sim2mirage.py:  mirage example {:s} file not found ***".format(mirexample))
        
        # make cube of data for mirage from input ami_sim file... even a 1-deep cube.
        mirobj[1].data = np.zeros((fobj_sim[0].data.shape[0], #"slices of cube of integrations"
                                     mirobj[1].data.shape[0], 
                                     mirobj[1].data.shape[1]))
        
        mirobj[1].data = mirobj[1].data.astype(np.float32)
        mirobj[1].data = fobj_sim[0].data # replace with ami_sim data
        mirobj[1].data = mirobj[1].data[:,:80,:80]  # trim ends of cols, rows (if needed) to 80 x 80

        # Transfer non-conflicting keywords from sim data to mirage file header
        mirkeys = list(mirobj[0].header.keys())

        for kwd in fobj_sim[0].header.keys():
            if  kwd not in mirkeys and 'NAXIS' not in kwd:
                mirobj[0].header[kwd] = (fobj_sim[0].header[kwd], fobj_sim[0].header.comments[kwd])
       
        # change the example mirage header to the correct filter name
        # that amisim used... the nmae must be an allowd ami filter
        # actul bandpasses can be manipulated when creating the simulation.
        mirobj[0].header["FILTER"] = filt
        print(" TARGNAME =", mirobj[0].header["TARGNAME"],  " ",
              mirobj[0].header["FILTER"],
              " Input", fobj_sim[0].data.shape,
              " Output", mirobj[1].data.shape)

        # write out miragized sim data
        mirobj.writeto(datadir+fname+mirext+".fits", overwrite=True)

    mirexample = str(Path.home()) + \
                 "/gitsrc/ImPlaneIA/example_data/example_niriss/" + \
                 "jw00793001001_01101_00001_nis_cal.fits" 


if __name__ == "__main__":
    from pathlib import Path
    amisim2mirage(
        str(Path.home())+"/Downloads/asoulain_arch2019.12.07/Simulated_data/",
        ("t_dsk_100mas__F430M_81_flat_x11__00",
         "c_dsk_100mas__F430M_81_flat_x11__00",
        ),
        str(Path.home()) + \
        "/gitsrc/ImPlaneIA/example_data/example_niriss/" + \
        "jw00793001001_01101_00001_nis_cal.fits" ,
        "F430M"
    )
