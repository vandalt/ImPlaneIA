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

fov = 80
filt="F430M"
oversample=11
bandpass = np.array([(1.0, 4.3e-6),])


datafiles = (fitsimdir+'all_effects_data_mir.fits',)# all_effects_data_mir_copy.fits""").split()
np.random.seed(100)
#datafiles=['all_effects_data_mir.fits','all_effects_data_mir_copy.fits']


def simulate_data(rot_deg=0.0, psf_offsets_det = (0.0,0.0), phi_waves = np.zeros((7,))):
    """
    This function should not be a part of the calwebb_ami3 pipeline. It is used to simulate data for ImPlaneIA testing and verification.
    """
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
    """
    Use psf_offset from the function call
    """
    #************ADD PISTONS TO PERFECT DATA************************************************

    #lambda/14 ~ 80% strehl ratio
    #Doing tests with phi = 0.5*np.random.normal(0,1.0, 7) / 14.0 in the function call# waves
    
    phi = phi_waves - phi_waves.mean()
    print("phi", phi, "varphi", phi.var(), "waves")
    
    print("phi_nb stdev/w", phi.std())
    print("phi_nb stdev/r", phi.std()*2*np.pi)
    print("phi_nb mean/r", phi.mean()*2*np.pi)

    pistons = phi *4.3e-6   #meters
    #Temporarily overwrite with smaller pistons measured from CV3 data by un-commenting the line below.
    #pistons = np.array([5.25e-13, -2.054e-13, -4.300e-14, -2.38e-13,6.21e-15, -2.036e-13,  1.60e-13])  #meters. too small

    print("/=====input pistons/m=======/\n",pistons)
    print("/=====input pistons/r=======/\n",pistons*(2*np.pi)/4.3e-6)
    

    #***********ADD ALL EFFECTS TO SIMULATE**************************************************
    jw = NRM_Model(mask='jwst', holeshape="hex", affine2d=aff)
    jw.set_pixelscale(pixelscale_as*arcsec2rad)
    jw.set_pistons(pistons)
    jw.simulate(fov=fov, bandpass=bandpass, over=oversample,psf_offset=psf_offsets_det)
    fits.PrimaryHDU(data=jw.psf).writeto(fitsimdir+"all_effects_data.fits",overwrite=True)

    #**********Convert simulated data to mirage format.***************************************

    amisim2mirage( fitsimdir, ("all_effects_data",), mirexample, filt)
    return aff, psf_offsets_det, pistons,

    
def analyze_data(test_tar, affine2d = None, psf_offset_find_rotation = (0.0,0.0), psf_offset_ff = None, rotsearch_d = None, set_pistons = None):
    """
    This function can be used for analyzing any NIRISS AMI data in MIRAGE format. Use this for ami_analyze part of the calwebb_ami3 pipeline.
    """
    pixelscale_as=0.0656
    arcsec2rad = u.arcsec.to(u.rad)
    PIXELSCALE_r =  pixelscale_as * arcsec2rad
    holeshape='hex'
    print(test_tar)
    
    data = fits.getdata(test_tar)
    print(data.shape[1])
    dim = data.shape[1]       #Do not use 'fov' for parameter name. It's specific to the simulation which we may or may not do.

    from nrm_analysis import find_affine2d_parameters as FAP
    mx, my, sx,sy, xo,yo, = (1.0,1.0, 0.0,0.0, 0.0,0.0)
    
    psf_offset = psf_offset_find_rotation

    if affine2d is None:
        print("Finding affine2d...")
        affine2d = FAP.find_rotation(data[0,:,:], psf_offset,
                      rotsearch_d, mx, my, sx, sy, xo,yo,
                      PIXELSCALE_r, dim, bandpass, oversample, holeshape, outdir=fitsimdir)
        print("***** Using measured affine2d *****")
    else:
        affine2d = aff
        print("***** Using affine2d created with user supplied rotation *****")
    affine2d.show()

    niriss = InstrumentData.NIRISS(filt,bandpass=bandpass,affine2d=affine2d)
    ff_t = nrm_core.FringeFitter(niriss, psf_offset_ff=psf_offset_ff, datadir=fitsimdir, savedir=fitsimdir,
                                 oversample=oversample, interactive=False)
    
    print(test_tar)
    ff_t.fit_fringes(test_tar)

    data=fits.getdata(test_tar)

    n_residual_max = (ff_t.nrm.residual/data.max()).max()
    n_residual_min = (ff_t.nrm.residual/data.max()).min()

    print("***** Mean CP ",ff_t.nrm.redundant_cps.mean(), " *****")
    print("***** Sigma CP ",ff_t.nrm.redundant_cps.std(), " *****")

    pos = np.get_printoptions()
    np.set_printoptions(precision=3, formatter={'float': lambda x: '{:+.3e}'.format(x)},
                            linewidth=80)
    print("***** n_residual[36:-36,36:-36] *****")
    print((ff_t.nrm.residual/data.max())[36:-36,36:-36])
    print("***** n_residual_max ", n_residual_max, " *****")
    print("***** n_residual_min ", n_residual_min, " *****")

    return affine2d, ff_t.nrm.bestcenter,  psf_offset, ff_t.nrm.fringepistons,
    print("done")
    np.set_printoptions(pos)
    
    

if __name__ == "__main__":

    rot = 2.0
    rot = avoidhexsingularity(rot) # in utils
    aff = Affine2d(rotradccw=np.pi*rot/180.0, name="{0:.0f}".format(rot)) # in utils
    _rotsearch_d = (-3.00, -2.00,-1.0,0.0,1.0,2.0, 3.0,4.0, 5.0, 6.0)

    #Uncomment to create one of the simulations below
    #affine2d_in,psf_offsets_in,pistons_in = simulate_data(rot_deg=rot,psf_offsets_det= (0.0,0.0))                                                  # perfect data when rot =0.0
    #affine2d_in,psf_offsets_in,pistons_in = simulate_data(rot_deg=rot,psf_offsets_det= (0.48,0.0))                                                 # a. rotation (zero on non-zero defined by rot), non zero psf_offsets, zero pistons
    #affine2d_in,psf_offsets_in,pistons_in = simulate_data(rot_deg=rot,psf_offsets_det= (0.0,0.0), phi_waves=0.5*np.random.normal(0,1.0, 7)/14.0)   # b. rotation (zero on non-zero defined by rot),no psf_offsets, non-zero pistons
    affine2d_in,psf_offsets_in,pistons_in = simulate_data(rot_deg=rot,psf_offsets_det= (0.48,0.0), phi_waves=0.5*np.random.normal(0,1.0, 7)/14.0)   # c. rotation (zero on non-zero defined by rot), non-zero psf_offsets, non-zero pistons

    
    for df in datafiles:
        print("analyzing", df)
        datafile = fits.getdata(df)

        """
        *********  Note: When affine2d = None use rotsearch_d **********************************************
        ********** when affine2d is created with user suppiled rotation, psf_offset_find_rotation and rotsearch_d are not used to find rotation  ************
        ********** Put pistons back? TBD. set_pistons is a placeholder **************************************

        Use forced psf_offset_ff with caution. Think whether the FOV is even or odd, where the brightest pixel is. If in the simulation you introduced no rotation,
        no pistons, and an offset of 0.48 for odd FOV the center of the PSF stays inside the same pixel at 0.48 pixels from the center of the pixel. Fringefitter
        uses offset from the center of the brightest pixel. Introducing a rotation (e.g 2 degrees) and some pistons moves the brightest pixel to the neoghboring pixel
        on the right while keeping the PSF center at 0.48 in the original pixel. The correct offset for fringefitter is then the one measured from the center of
        the brightest pixel on the right. This is -0.52.
        """

        # ***** THESE ARE DIFFERENT (NOT ALL) WAYS TO ANALYZE DATA THAT IS ROTATED BY SOME ANGLE (defined by rot for simulation in main), HAS PISTONS AND PSF_OFFSET OF (0.48, 0.0) *****
        # ***** Use [1], [3], [4], [5], [6], [7] to analyze data odd fov data simulated by 'c'.
        #If FOV is even and you introduced an offset of (0.48,0.0) in the simulation psf_offset_ff=(-0.5199,0.0) will not work. Update if you use even fov."

        #[1]
        #_aff, _psf_offset_ff, _psf_offset_r,_pistons = analyze_data(df, affine2d = None, psf_offset_find_rotation=(0.0,0.0), psf_offset_ff=None, rotsearch_d = _rotsearch_d, set_pistons = None)

        #[2]
        #_aff, _psf_offset_ff, _psf_offset_r, _pistons = analyze_data(df, affine2d = None, psf_offset_find_rotation=(0.0,0.0), psf_offset_ff=(0.48,0.0), rotsearch_d = _rotsearch_d, set_pistons = None)

        #[3]
        #_aff, _psf_offset_ff, _psf_offset_r, _pistons = analyze_data(df, affine2d = None, psf_offset_find_rotation=(0.0,0.0), psf_offset_ff=(-0.5199,0.0), rotsearch_d = _rotsearch_d, set_pistons = None)

        #[4]
        _aff, _psf_offset_ff, _psf_offset_r, _pistons = analyze_data(df, affine2d = None, psf_offset_find_rotation=(0.48,0.0), psf_offset_ff=None, rotsearch_d = _rotsearch_d, set_pistons = None)

        #[5]
        #_aff, _psf_offset_ff, _psf_offset_r, _pistons = analyze_data(df, affine2d = aff, psf_offset_find_rotation=(0.0,0.0), psf_offset_ff=None, rotsearch_d = None, set_pistons = None)

        #[6]
        #_aff, _psf_offset_ff, _psf_offset_r, _pistons = analyze_data(df, affine2d = None, psf_offset_find_rotation=(0.48,0.0), psf_offset_ff=(-0.5199,0.0), rotsearch_d = _rotsearch_d, set_pistons = None)

        #[7]
        #_aff, _psf_offset_ff, _psf_offset_r, _pistons = analyze_data(df, affine2d = aff, psf_offset_find_rotation=(0.48,0.0), psf_offset_ff=(-0.5199,0.0), rotsearch_d = _rotsearch_d, set_pistons = None)

        print("/====affine used=========/")
        _aff.show_rot()

        print("/=====psf_offset used to find rotation====/")
        print("***** NOTE: psf offset used to find rotation is not used when affine2d is provided by the user. *****")
        print(_psf_offset_r,"\n")
        print("/=====psf_offset used by fringefitter====/\n",_psf_offset_ff)
        
        
        
    print("/=====input affine==========/")
    affine2d_in.show_rot()
    print("/=====input psf_offset=======/\n", psf_offsets_in)
    #print("/=====input pistons/r=======/\n",pistons_in*2*np.pi/4.3e-6)
    utils.compare_pistons(pistons_in*2*np.pi/4.3e-6,_pistons)
    print("Examine the residuals in", fitsimdir, "\n")
