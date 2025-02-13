#! /usr/bin/env python

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
import sys
import time
from scipy.integrate import simps 
from nrm_analysis import nrm_core, InstrumentData, fringefitting


""" Hokey but in webbpsf_driver_NIRISS_nojitter.py output for A0V on a 255x255 detector we see:   

filter      sampling  psftot    psfpeak       cpf
F430_G2V    detector    0.1503   2.6923e-5   0.01945

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
#use A0V for photometry for now...

filt_psftot = {"F480M":0.1498,
               "F430M":0.1503,
               "F380M":0.1506,
               "F277W":0.1516
               }
filt_cpf    = {"F480M":0.002381,
               "F430M":0.002922,
               "F380M":0.023455,
               "F277W":0.038885,
               }


def applyspectrum(jwinstr, filt, spt, wave, weight):
    """ fill in with pysynphot """
    return weight

webbpsffiledir = "./webbpsffiles_for_filters/"
def get_webbpsffilters(sptype=None):
    """ 
        read in psf headers and scoop out filter bandpasses 
        returns bpd, cwd, ewd, betad dictionaries keyed by filter name
        filter bandpass dictionary values are np array[npoints,2] with
           WGHT = filt_bpd[filt][:,0]
           WAVE = filt_bpd[filt][:,1] 
        central wavelength dictionary values weighted central wavelength / m
        equivalent width dictionary values in wavelength / m
        fractional bandpass dictionary values dimensionless
        sptype, or flat spec (in counts) assumed.  eg A0V  G2V strings
    """
    m = 1.0
    um = 1.0e-6 * m

    filters = ("F480M", "F430M", "F380M", "F277W")
    pupil = "MASK_NRM"
    sptype = "A0V" # just to match file name of webbpsf psf files

    # number of points used by webbpsf to calc the psf
    # /Users/_moi_/anaconda3/envs/astroconda/share/webbpsf-data/NIRISS/filters.tsv
    filt_n = {"F480M":19, "F430M":19, "F380M":19, "F277W":41}

    filt_bpd = {"F480M":np.zeros((filt_n["F480M"],2)), 
                "F430M":np.zeros((filt_n["F430M"],2)), 
                "F380M":np.zeros((filt_n["F380M"],2)), 
                "F277W":np.zeros((filt_n["F277W"],2))
                }
    filt_cw = {}  # weighted central wavelength / m
    filt_ew = {}  # equiv width / m
    filt_beta = {}  # fractional bandpass
    for filt in filters:
        print(filt, filt_n[filt], "points")
        hdr  = fits.getheader(webbpsffiledir+"PSF_det_%s_%s_%s.fits"%(pupil, filt, sptype))

        # read in filter weights & wavelengths (meters)
        # order of 0 & 1 matches use in simulateG7S6_f430m.py
        for ii in range(filt_n[filt]):
            filt_bpd[filt][ii,1] = hdr["WAVE{:d}".format(ii)]
            filt_bpd[filt][ii,0] = hdr["WGHT{:d}".format(ii)]
        WGHT = filt_bpd[filt][:,0]
        WAVE = filt_bpd[filt][:,1]

        # multiply by target spectrun in counts per wavelength interval: 
        if sptype is not None:
            WGHT = applyspectrum('NIRISS', filt, sptype, WAVE, WGHT)

        wavelen = (WAVE*WGHT).sum()/WGHT.sum() # Weighted mean wavelength in meters
        area = simps(WGHT, WAVE)
        ew = area / WGHT.max()
        filt_cw[filt] = wavelen
        filt_ew[filt] = ew
        filt_beta[filt] = ew/wavelen
        print("\tctr wavelength {:.4e} um".format(wavelen/um))
        print("\tequiv width {:.4e} um".format(ew/um))
        print("\tfrac bandpass {:.2e}".format(ew/wavelen))
        print()

    # target spectrum from webbpsf_data:


    return filt_bpd, filt_cw, filt_ew, filt_beta


def psf(filt, fbp, cw, ew, beta, data_dir,
              oversample = 11, 
              n_image = 81, 
              pixelscale_as=0.0656, 
              f2f = 0.82, 
              saveover=False):
    
    arcsec2rad = u.arcsec.to(u.rad)

    WGHT = fbp[:,0]
    WAVE = fbp[:,1]

    # setup parameters for simulation
    verbose = 1
    overwrite = 1

    mask = 'MASK_NRM'

    # directory containing the test data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    name_seed = 'PSF_%s_%s_x%d_%.2f'%(mask,filt,oversample,f2f) # original
    # only det sampling needed here...
    name_seed = '%s_%d_%s_det.fits'%(filt, n_image, "A0V")  # for ami_sim input photometry in NIRISSami_apt_calcPSF
    name_seed = '%s_%d_%s_x%d.fits'%(filt, n_image, "flat", oversample)  # for ami_sim PSF to give to driver_scene
    name_seed_det = '%s_%d_%s_det.fits'%(filt, n_image, "flat")  # for ami_sim PSF for examining by eye 
    print(name_seed)
    psf_image = os.path.join(data_dir,name_seed)
    psf_image_without_oversampling = os.path.join(data_dir, name_seed_det)

    from nrm_analysis.fringefitting.LG_Model import NRM_Model
    jw = NRM_Model(mask='jwst',holeshape="hex")
    jw.set_pixelscale(pixelscale_as*arcsec2rad)
    jw.simulate(fov=n_image, 
        bandpass=fbp, 
        over=oversample)
    print("simulateG7S6psf: simulation oversampling is", oversample)
    (year, month, day, hour, minute, second, weekday, DOY, DST) =  time.gmtime()

    # oversampled psf is used as the fine psf to create jitter and dither errors. 
    # ami_sim driver scene expects ODD numbers of pixels to a side.
    # We oversize to 81 x 81 pixels here.  
    # When adding mirage headers to ami_sim 'data' we should trim to 80x80.
    # oversampled psf is used as the fine psf to create jitter and dither errors. 

    if saveover:  # oversampled psf is used as the fine psf to createjitter and dither errors. 
        # PSF on oversampled pixels
        psf_over_n =  filt_psftot[filt] * jw.psf_over/jw.psf_over.sum()  # wp psf total ~15%
        fits.writeto(psf_image, psf_over_n, overwrite=True)
        header = fits.getheader(psf_image)
        header['PIXELSCL'] = pixelscale_as/oversample, "arcsec per pixel"
        header['FILTER'] = filt
        header['PUPIL'] = mask 
        header["OVER"] = ( oversample, "oversample")
        header["SPTYPE"] = "FLAT",  "source spectral type unspecified"
        header["NWAVES"] = fbp.shape[0]
        header["WAVELEN"] = cw, "Weighted mean wavelength in meters "
        header["FILT_EW"] = ew, "Filter equiv width"
        header["FILT_BP"] = beta, "Fractional bandpass = WL/EqW"
        for i in range(len(WAVE)):
            header["WAVE%d"%i] = WAVE[i], "Wavelength %d"%i
            header["WGHT%d"%i] = WGHT[i], "Wavelength weight %d"%i
        header["FT"] = ( "analytical", "hexee * fringes")
        header['NRM_GEOM'] =  'G7S6', 'Beaulieu, PGT, AS, active ctrs'
        header['F2F'] = jw.d, "flat2flat hole size m"
        header['NRM_X_A1'] = jw.ctrs[0,0], 'X coordinate (m) of NRM sub-aperture 0'          
        header['NRM_Y_A1'] = jw.ctrs[0,1], 'Y coordinate (m) of NRM sub-aperture 0'         
        header['NRM_X_A2'] = jw.ctrs[1,0], 'X coordinate (m) of NRM sub-aperture 1'          
        header['NRM_Y_A2'] = jw.ctrs[1,1], 'Y coordinate (m) of NRM sub-aperture 1'          
        header['NRM_X_A3'] = jw.ctrs[2,0], 'X coordinate (m) of NRM sub-aperture 2'          
        header['NRM_Y_A3'] = jw.ctrs[2,1], 'Y coordinate (m) of NRM sub-aperture 2'          
        header['NRM_X_A4'] = jw.ctrs[3,0], 'X coordinate (m) of NRM sub-aperture 3'          
        header['NRM_Y_A4'] = jw.ctrs[3,1], 'Y coordinate (m) of NRM sub-aperture 3'          
        header['NRM_X_A5'] = jw.ctrs[4,0], 'X coordinate (m) of NRM sub-aperture 4'          
        header['NRM_Y_A5'] = jw.ctrs[4,1], 'Y coordinate (m) of NRM sub-aperture 4'          
        header['NRM_X_A6'] = jw.ctrs[5,0], 'X coordinate (m) of NRM sub-aperture 5'          
        header['NRM_Y_A6'] = jw.ctrs[5,1], 'Y coordinate (m) of NRM sub-aperture 5'          
        header['NRM_X_A7'] = jw.ctrs[6,0], 'X coordinate (m) of NRM sub-aperture 6'          
        header['NRM_Y_A7'] = jw.ctrs[6,1], 'Y coordinate (m) of NRM sub-aperture 6'   
        header['NORMALIZ'] = 'first', 'PM psf.sum=1, this.sum = mask throughput'
        header['PSFTOT'] = filt_psftot[filt], "sum of webbpsf sim done at 11x over"
        header['PSFPEAK'] = psf_over_n.max()
        header["SRC"] = ( "simulateG7S6psf.py", "ImplaneIA/notebooks/")
        header['AUTHOR'] = '%s@%s' % (os.getenv('USER'), os.getenv('HOST')), 'username@host for calculation'
        header['DATE'] = '%4d-%02d-%02dT%02d:%02d:%02d' %  (year, month, day, hour, minute, second), 'Date of calculation'
        header['F2F'] = (0.82, "flat2flat hole size m")
        fits.update(psf_image, psf_over_n, header=header)

    # PSF on detector pixels
    psf_det_n = filt_psftot[filt] * jw.psf/jw.psf.sum()
    fits.writeto(psf_image_without_oversampling, psf_det_n, overwrite=True)
    header = fits.getheader(psf_image_without_oversampling)
    header['PIXELSCL'] = pixelscale_as, "arcsec per pixel"
    header['FILTER'] = filt
    header['PUPIL'] = mask 
    header["OVER"] = ( oversample, "oversample pre-rebin")
    header["SPTYPE"] = "FLAT", "source spectral type"
    header["NWAVES"] = fbp.shape[0]
    header["WAVELEN"] = cw, "Weighted mean wavelength in meters "
    header["FILT_EW"] = ew, "Filter equiv width"
    header["FILT_BP"] = beta, "Fractional bandpass = WL/EqW"
    for i in range(len(WAVE)):
        header["WAVE%d"%i] = WAVE[i], "Wavelength %d"%i
        header["WGHT%d"%i] = WGHT[i], "Wavelength weight %d"%i
    header["FT"] = ( "analytical", "hexee * fringes")
    header['NRM_GEOM'] =  'G7S6', 'Beaulieu, PGT, AS active ctrs'
    header['F2F'] = jw.d, "flat2flat hole size m"
    header['NRM_X_A1'] = jw.ctrs[0,0], 'X coordinate (m) of NRM sub-aperture 0'          
    header['NRM_Y_A1'] = jw.ctrs[0,1], 'Y coordinate (m) of NRM sub-aperture 0'         
    header['NRM_X_A2'] = jw.ctrs[1,0], 'X coordinate (m) of NRM sub-aperture 1'          
    header['NRM_Y_A2'] = jw.ctrs[1,1], 'Y coordinate (m) of NRM sub-aperture 1'          
    header['NRM_X_A3'] = jw.ctrs[2,0], 'X coordinate (m) of NRM sub-aperture 2'          
    header['NRM_Y_A3'] = jw.ctrs[2,1], 'Y coordinate (m) of NRM sub-aperture 2'          
    header['NRM_X_A4'] = jw.ctrs[3,0], 'X coordinate (m) of NRM sub-aperture 3'          
    header['NRM_Y_A4'] = jw.ctrs[3,1], 'Y coordinate (m) of NRM sub-aperture 3'          
    header['NRM_X_A5'] = jw.ctrs[4,0], 'X coordinate (m) of NRM sub-aperture 4'          
    header['NRM_Y_A5'] = jw.ctrs[4,1], 'Y coordinate (m) of NRM sub-aperture 4'          
    header['NRM_X_A6'] = jw.ctrs[5,0], 'X coordinate (m) of NRM sub-aperture 5'          
    header['NRM_Y_A6'] = jw.ctrs[5,1], 'Y coordinate (m) of NRM sub-aperture 5'          
    header['NRM_X_A7'] = jw.ctrs[6,0], 'X coordinate (m) of NRM sub-aperture 6'          
    header['NRM_Y_A7'] = jw.ctrs[6,1], 'Y coordinate (m) of NRM sub-aperture 6'   
    header['NORMALIZ'] = 'first', 'PM psf.sum=1, this.sum = mask throughput'
    header['PSFTOT'] = psf_det_n.sum()
    header['PSFPEAK'] = psf_det_n.max()
    header['CPF'] = psf_det_n.max() / psf_det_n.sum(), 'Central pixel fraction'
    header["SRC"] = ( "simulateG7S6psf_for_ami_sim.py", "ImplaneIA/notebooks/")
    header['AUTHOR'] = '%s@%s' % (os.getenv('USER'), os.getenv('HOST')), 'username@host for calculation'
    header['DATE'] = '%4d-%02d-%02dT%02d:%02d:%02d' %  (year, month, day, hour, minute, second), 'Date of calculation'
    fits.update(psf_image_without_oversampling, psf_det_n, header=header)


if __name__ == "__main__":
    bpd, cwd, ewd, betad = get_webbpsffilters()
    filters = ("F480M", "F430M", "F380M", "F277W")
    for ff in filters:

        # activate this line in code above:
        #      name_seed = '%s_%d_%s_det.fits'%(filt, n_image, "A0V")  
        #      # for ami_sim input photometry in NIRISSami_apt_calcPSF
        #sf(ff, bpd[ff], cwd[ff], ewd[ff], betad[ff],  './simulatedpsfs/', oversample=3, n_image=255, saveover=False)
        #sf(ff, bpd[ff], cwd[ff], ewd[ff], betad[ff],  './simulatedpsfs/', oversample=3, n_image=31,  saveover=False)


        # activate this line in code above:
        #      name_seed = '%s_%d_%s_x%d.fits'%(filt, n_image, "flat", oversample)  
        #      # for ami_sim PSF to give to driver_scene
        psf(ff, bpd[ff], cwd[ff], ewd[ff], betad[ff],  './simulatedpsfs_for_ami_sim/', oversample=11, n_image=81, saveover=True)
