#! /usr/bin/env python

import os
import sys
import numpy as np
from astropy.io import fits
from astropy import units as u
import time
from nrm_analysis.fringefitting.LG_Model import NRM_Model
from nrm_analysis.misctools import utils  # AS LG++
from nrm_analysis.misctools.utils import amisim2mirage
from nrm_analysis import nrm_core, InstrumentData


home = os.path.expanduser('~')
datadir = home+"/gitsrc/ImPlaneIA/notebooks/implaneia_tests/"
tr = "perfect_wpistons_mir"
test_tar = datadir + tr + ".fits"
oversample=3
filt="F430M"

mirexample = os.path.expanduser('~') + \
             "/ImplaneIA/notebooks/simulatedpsfs/" + \
             "jw00793001001_01101_00001_nis_cal.fits" 
mirexample = os.path.expanduser('~') + \
        "/gitsrc/ImPlaneIA/example_data/example_niriss/" + \
        "jw00793001001_01101_00001_nis_cal.fits"


"""
F480M A0V:
    Total of PSF is: 0.14983928265312782
    Peak pixel is:  0.0023813565623978152
    Central pixel fraction is:  0.015892738674614215
F430M_A0V:
    Total of PSF is: 0.15029221331182707
    Peak pixel is:  0.0029221329659187313
    Central pixel fraction is:  0.01944300973102229
F380M_A0V
    Total of PSF is: 0.15064757333183165
    Peak pixel is:  0.0035334896463460517
    Central pixel fraction is:  0.023455337302797623
F377W_A0V:
    Total of PSF is: 0.1515569249392621
    Peak pixel is:  0.005893341903346654
    Central pixel fraction is:  0.038885335696197766
"""
def verify_pistons(arg):

    """
    Create simulated data with pistons,
    1. analyze the data by calling fit_image
    Do not use fringefitter since the data is perfecrly centered (the if loop).
    Check if input and output pistons match.
    
    2. analyze the data by calling fit_image via fringefitter
    Fringefitter finds the bestcenter and makes model with bestcenter.
    Since the data is perfecrly centered fringefitter should not affect output pistons.
    Check if input and output pistons match.
    """
    
    jw = NRM_Model(mask='jwst',holeshape="hex")
    pixelscale_as=0.0656
    arcsec2rad = u.arcsec.to(u.rad)

    jw.set_pixelscale(pixelscale_as*arcsec2rad)

    np.random.seed(100)

    #lambda/14 ~ 80% strehl ratio
    phi = np.random.normal(0,1.0, 7) / 14.0 # waves
    print("phi", phi, "varphi", phi.var(), "waves")
    phi = phi - phi.mean()

    print("phi_nb stdev/w", phi.std())
    print("phi_nb stdev/r", phi.std()*2*np.pi)
    print("phi_nb mean/r", phi.mean()*2*np.pi)
    pistons = phi *4.3e-6 #OR

    print("/=====input pistons/m=======/\n",pistons)
    print("/=====input pistons/r=======/\n",pistons*(2*np.pi)/4.3e-6)

    jw.set_pistons(pistons)
    jw.simulate(fov=81, bandpass=4.3e-6, over=oversample)
    fits.PrimaryHDU(data=jw.psf).writeto("implaneia_tests/perfect_wpistons.fits",overwrite=True)

    if arg == ("no_fringefitter"):
        
        jw.make_model(fov=81, bandpass=4.3e-6, over=oversample)
        jw.fit_image(jw.psf, modelin=jw.model)

        pos = np.get_printoptions()
        np.set_printoptions(precision=4, formatter={'float': lambda x: '{:+.4e}'.format(x)},
                            linewidth=80)
        print("Residual/psfpeak array center:", jw.psf.shape)
        print((jw.residual/jw.psf.max())[36:-36,36:-36])
        np.set_printoptions(pos)

        fits.PrimaryHDU(data=jw.residual).writeto("residual_pistons_no_ff.fits",overwrite=True)
        #return jw
        #print("output pistons/r",jw.fringepistons)
        #print("output pistons/w",jw.fringepistons/(2*np.pi))
        #print("output pistons/m",jw.fringepistons*4.3e-6/(2*np.pi))
        #print("input pistons/m ",jw.phi)  
    
            
        
    elif arg == ("use_fringefitter"):
       
        fits.PrimaryHDU(data=jw.psf).writeto(datadir+"/perfect_wpistons.fits",overwrite=True)
   
        amisim2mirage( datadir, ("perfect_wpistons",), mirexample, filt)

        niriss = InstrumentData.NIRISS(filt)
        ff_t = nrm_core.FringeFitter(niriss, datadir=datadir, savedir=datadir, 
                                     oversample=oversample, interactive=False)
        print(test_tar)
        ff_t.fit_fringes(test_tar)

        print("Residual:")
        #print(ff_t.nrm.residual)
        print("Residual/psfpeak array center:", ff_t.nrm.reference.shape)
        pos = np.get_printoptions()
        np.set_printoptions(precision=3, formatter={'float': lambda x: '{:+.3e}'.format(x)},
                            linewidth=80)
        print((ff_t.nrm.residual/jw.psf.max())[36:-36,36:-36])
        np.set_printoptions(pos)

        fits.PrimaryHDU(data=ff_t.nrm.residual).writeto(datadir+\
                        "/residual_pistons_with_ff.fits",overwrite=True)
    
        utils.compare_pistons(jw.phi*2*np.pi/4.3e-6, ff_t.nrm.fringepistons, str="ff_t")

        #print("output pistons/r",ff_t.nrm.fringepistons)
        #print("output pistons/w",ff_t.nrm.fringepistons/(2*np.pi))
        #print("output pistons/m",ff_t.nrm.fringepistons*4.3e-6/(2*np.pi))
        #print("input pistons/m ",jw.phi)   
        

if __name__ == "__main__":

    
    #verify_pistons("no_fringefitter")
    verify_pistons("use_fringefitter")
