#! /usr/bin/env python

import os
import numpy as np
from astropy.io import fits
from astropy import units as u
import sys
from nrm_analysis.fringefitting.LG_Model import NRM_Model
from nrm_analysis.misctools import utils  
from nrm_analysis.misctools.utils import mas2rad, amisim2mirage
from nrm_analysis import nrm_core, InstrumentData
from pathlib import Path
from nrm_analysis.misctools.utils import avoidhexsingularity
from nrm_analysis.misctools.utils import Affine2d


np.set_printoptions(precision=6,linewidth=160)
home = os.path.expanduser('~')
fitsimdir = home+"/data/implaneia/niriss_verification/test_offset_residuals/"
if not os.path.exists(fitsimdir):  
    os.makedirs(fitsimdir)

tr = "offset_data_mir"
test_tar = fitsimdir + tr + ".fits"

#mirexample = os.path.expanduser('~') + \
#        "/gitsrc/ImPlaneIA/example_data/example_niriss/" + \
#        "jw00793001001_01101_00001_nis_cal.fits"
mirexample = os.path.expanduser('~') + \
             "/ImplaneIA/notebooks/simulatedpsfs/" + \
             "jw00793001001_01101_00001_nis_cal.fits" 

filt="F430M"
oversample=11
bandpass = np.array([(1.0, 4.3e-6),])
psf_offsets = (0.48,0.0)

def test_offset_residuals():
    """
    Use same affine object for simulation and analysis
    """ 
    pixelscale_as=0.0656
    arcsec2rad = u.arcsec.to(u.rad)

    #psf_offsets = ((0,0), (1.0,0), )
    
    #psf_offsets = ((0,0), (1.0,0), (0, 1.0), (1.0,1.0))
    
    bandpass = np.array([(1.0, 4.3e-6),]) 
    # Loop over psf_offsets
    
    jw = NRM_Model(mask='jwst', holeshape='hex')
    jw.set_pixelscale(pixelscale_as*arcsec2rad)
            
    jw.simulate(fov=35, bandpass=bandpass, over=oversample, psf_offset=psf_offsets)
    fits.writeto(fitsimdir+"offset_data.fits",jw.psf,overwrite=True)
    
    jw.make_model(fov=35, bandpass=bandpass, over=oversample, psf_offset=psf_offsets )

    jw.fit_image(jw.psf, modelin=jw.model)
    fits.writeto(fitsimdir+"residual_offset.fits",jw.residual,overwrite=True)
    

def test_offset_residuals_with_offset_measured():
    """
    Simulate data with known offsets, analyze data using fringefitter.
    """ 

    pixelscale_as=0.0656
    arcsec2rad = u.arcsec.to(u.rad)
    PIXELSCALE_r =  pixelscale_as * arcsec2rad

    jw0 = NRM_Model(mask='jwst', holeshape="hex")
    jw0.set_pixelscale(pixelscale_as*arcsec2rad)

    jw0.simulate(fov=35, bandpass=bandpass, over=oversample, psf_offset=(0.0,0.0))
    fits.PrimaryHDU(data=jw0.psf).writeto(fitsimdir+"no_offset_data.fits",overwrite=True)
    
    jw = NRM_Model(mask='jwst', holeshape="hex")
    jw.set_pixelscale(pixelscale_as*arcsec2rad)
    
    jw.simulate(fov=35, bandpass=bandpass, over=oversample, psf_offset=psf_offsets)
    fits.PrimaryHDU(data=jw.psf).writeto(fitsimdir+"offset_data.fits",overwrite=True)
    fits.PrimaryHDU(data=(jw.psf-jw0.psf)/jw0.psf.max()).writeto(fitsimdir+"diff_from_center.fits",overwrite=True)
   
    amisim2mirage( fitsimdir, ("offset_data",), mirexample, filt)

    niriss = InstrumentData.NIRISS(filt,bandpass=bandpass)
    ff_t = nrm_core.FringeFitter(niriss, datadir=fitsimdir, savedir=fitsimdir, 
                                     oversample=oversample, interactive=False)
    print(test_tar)
    ff_t.fit_fringes(test_tar)

    jw_m = NRM_Model(mask='jwst', holeshape="hex")
    jw_m.set_pixelscale(pixelscale_as*arcsec2rad)

    #Look at measured offsets in the screen output and feed them to simulate to compare with simulated data created with input offsets.
    #nrm.bestcenter (0.4799802988666451, 6.984734412937637e-05)  nrm.xpos 0.4799802988666451  nrm.ypos 6.984734412937637e-05
    jw_m.simulate(fov=35, bandpass=bandpass, over=oversample, psf_offset=(ff_t.nrm.xpos, ff_t.nrm.ypos))
    
    fits.PrimaryHDU(data=jw_m.psf).writeto(fitsimdir+"m_offset_data.fits",overwrite=True)
    fits.PrimaryHDU(data=(jw.psf-jw_m.psf)/jw0.psf.max()).writeto(fitsimdir+"m_diff_of_offsets.fits",overwrite=True)
    
    print("Residual:")
    #print(ff_t.nrm.residual)
    print("Residual/psfpeak array center:", ff_t.nrm.reference.shape)
    pos = np.get_printoptions()
    np.set_printoptions(precision=3, formatter={'float': lambda x: '{:+.3e}'.format(x)},
                            linewidth=80)
    print((ff_t.nrm.residual/jw.psf.max())[36:-36,36:-36])
    np.set_printoptions(pos)

    fits.PrimaryHDU(data=ff_t.nrm.residual).writeto(fitsimdir+\
                        "residual_offsets_with_ff.fits",overwrite=True)
    #fits.PrimaryHDU(data=ff_t.nrm.residual/jw.psf).writeto(fitsimdir+\
                       # "n_residual_offsets_with_ff.fits",overwrite=True)

 

if __name__ == "__main__":
    """
    Comment one of these lines to switch between using simulated vs measured rotation 
    """
    #test_offset_residuals()
    test_offset_residuals_with_offset_measured()

    print("Examine the offset data in", fitsimdir)
    print("Examine the residuals in", fitsimdir+"/offset_data_mir/")
    print("ds9 -zoom 8 ../offset_data.fits n_centered_0.fits n_modelsolution_00.fits n_residual_00.fits&")
    
