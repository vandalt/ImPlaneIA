#! /usr/bin/env python

import os
import numpy as np
from astropy.io import fits
from astropy import units as u
import sys
import string
from nrm_analysis.fringefitting.LG_Model import NRM_Model
from nrm_analysis.misctools import utils  
from nrm_analysis.misctools.utils import mas2rad, amisim2mirage
from nrm_analysis import nrm_core, InstrumentData
from pathlib import Path
from nrm_analysis.misctools.utils import avoidhexsingularity
from nrm_analysis.misctools.utils import Affine2d




np.set_printoptions(precision=6,linewidth=160)
home = os.path.expanduser('~')
fitsimdir = home+"/data/implaneia/niriss_verification/test_all_residuals/"
if not os.path.exists(fitsimdir):  
    os.makedirs(fitsimdir)

tr = "all_effects_data_mir"
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

#datafiles = ("""fitsimdir+'all_effects_data_mir.fits' fitsimdir+'all_effects_data_mir_copy.fits """).split( )
#print(datafiles)


#datafiles = ("""all_effects_data_mir.fits all_effects_data_mir_copy.fits""").split()
datafiles = (fitsimdir+'all_effects_data_mir.fits',)# all_effects_data_mir_copy.fits""").split()

#datafiles=['all_effects_data_mir.fits','all_effects_data_mir_copy.fits']


def simulate_data(rot_deg=0.0,psf_offsets_det = (0.0,0.0),phi_waves = None):

    pixelscale_as=0.0656
    arcsec2rad = u.arcsec.to(u.rad)
    PIXELSCALE_r =  pixelscale_as * arcsec2rad
    holeshape='hex'

    #*************ADD ROTATION TO PERFECT DATA***********************************************
    
    rot = avoidhexsingularity(rot_deg) # in utils
    affine_rot = Affine2d(rotradccw=np.pi*rot/180.0, name="{0:.0f}".format(rot)) # in utils
    aff = affine_rot
    print("***********simulated aff**************")
    aff.show()
    
    #*************ADD OFFSETS TO PERFECT DATA************************************************

    #Use psf_offset from the function call

    #************ADD PISTONS TO PERFECT DATA************************************************
    np.random.seed(100)

    #lambda/14 ~ 80% strehl ratio
    #phi = np.random.normal(0,1.0, 7) / 14.0 # waves
    
    phi = phi_waves - phi_waves.mean()
    print("phi", phi, "varphi", phi.var(), "waves")
    
    print("phi_nb stdev/w", phi.std())
    print("phi_nb stdev/r", phi.std()*2*np.pi)
    print("phi_nb mean/r", phi.mean()*2*np.pi)
    pistons = 0.0*phi *4.3e-6 #zero or non-zero pistons
    #pistons = phi *4.3e-6   #meters

    print("/=====input pistons/m=======/\n",pistons)
    print("/=====input pistons/r=======/\n",pistons*(2*np.pi)/4.3e-6)

    

    #***********ADD ALL EFFECTS TO SIMULATE**************************************************
    jw = NRM_Model(mask='jwst', holeshape="hex", affine2d=aff)
    jw.set_pixelscale(pixelscale_as*arcsec2rad)
    jw.set_pistons(pistons)
    jw.simulate(fov=35, bandpass=bandpass, over=oversample,psf_offset=psf_offsets_det )
    fits.PrimaryHDU(data=jw.psf).writeto(fitsimdir+"all_effects_data.fits",overwrite=True)

    #**********Convert simulated data to mirage format***************************************

    amisim2mirage( fitsimdir, ("all_effects_data",), mirexample, filt)

    
def analyze_data(test_tar, affine2d = None, set_center=(0.0,0.0), rotsearch_d = None, set_pistons = None):
    
    pixelscale_as=0.0656
    arcsec2rad = u.arcsec.to(u.rad)
    PIXELSCALE_r =  pixelscale_as * arcsec2rad
    holeshape='hex'
    print(test_tar)
    
    data = fits.getdata(test_tar)
    print(data.shape)
    
    
    from nrm_analysis import find_affine2d_parameters as FAP
    mx, my, sx,sy, xo,yo, = (1.0,1.0, 0.0,0.0, 0.0,0.0)
    #rotsearch_d = (8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0)
    #rotsearch_d = (-1.0, 0.0,1.0,2.0,3.0,4.0, 5.0)
    #rotsearch_d = (-3.0, -2.0,-1.0,0.0,1.0,2.0, 3.0)
    
    psf_offset = set_center

    if rotsearch_d: 
        affine2d = FAP.find_rotation(data[0,:,:], psf_offset,
                      rotsearch_d, mx, my, sx, sy, xo,yo,
                      PIXELSCALE_r, 35, bandpass, oversample, holeshape, outdir=fitsimdir)
    
        print("***********measured aff: aff_rot_measured**************")
        affine2d.show()
    
    
    niriss = InstrumentData.NIRISS(filt,bandpass=bandpass,affine2d=affine2d) #aff_rot_measured
    ff_t = nrm_core.FringeFitter(niriss, datadir=fitsimdir, savedir=fitsimdir, 
                                     oversample=oversample, interactive=False)
    print(test_tar)
    ff_t.fit_fringes(test_tar)

    pos = np.get_printoptions()
    np.set_printoptions(precision=3, formatter={'float': lambda x: '{:+.3e}'.format(x)},
                            linewidth=80)
    print((ff_t.nrm.residual/data.max())[36:-36,36:-36])
    print("*****  measured pistons ******/r", ff_t.nrm.fringepistons)
    print("output pistons/r",ff_t.nrm.fringepistons)
    print("output pistons/w",ff_t.nrm.fringepistons/(2*np.pi))
    print("output pistons/m",ff_t.nrm.fringepistons*4.3e-6/(2*np.pi))
    #print("input pistons/m ",jw.phi)
    return affine2d, (ff_t.nrm.xpos, ff_t.nrm.ypos),  ff_t.nrm.fringepistons,
    print("done")
    ##np.set_printoptions(pos)
    
    

if __name__ == "__main__":

    rot = 0.0
    rot = avoidhexsingularity(rot) # in utils
    aff = Affine2d(rotradccw=np.pi*rot/180.0, name="{0:.0f}".format(rot)) # in utils
    _rotsearch_d = (-3.0, -2.0,-1.0,0.0,1.0,2.0, 3.0)
   
    
    for df in datafiles:
        print("analyzing", df)

        """
        Find rotation and use for analysis
        analyze_data(df,affine2d = None, set_center=(0.0,0.0), rotsearch_d = _rotsearch_d, set_pistons = None)
        OR
        affine object (aff) created using user supplied rotation
        analyze_data(df,affine2d = aff, set_center=(0.0,0.0), rotsearch_d = None, set_pistons = None)
        """

        _aff, _psf_offset, _pistons = analyze_data(df,affine2d = None, set_center=(0.0,0.0), rotsearch_d = _rotsearch_d, set_pistons = None)
        #_aff, _psf_offset, _pistons = analyze_data(file,affine2d = aff, set_center=(0.48,0.0), rotsearch_d = None, set_pistons = None)
        
        print("/====measured affine=========/")
        _aff.show()
        print("/=====measured psf_offset====/\n",_psf_offset)
        print("/=====output pistons/r=======/\n",_pistons)
        
        
        """
        Work in progress
        aff_list =[ ]
        i=0
        while(i<= 4):
           
           _aff, _psf_offset, _pistons = analyze_data(file,affine2d = aff, set_center=(0.48,0.0), rotsearch_d = None, set_pistons = None)
           aff_list.append(_aff)
           print(type(_aff))
           affine2d = _aff
           set_center = _psf_offset
           print(_psf_offset, _pistons), 
           _aff.show()  #ERROR  AttributeError: 'tuple' object has no attribute 'show'
           i = i+1 
        """
         
    print("Examine the residuals in", fitsimdir, "\n")
        
       
