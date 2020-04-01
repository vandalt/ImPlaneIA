#!/usr/bin/env python
# coding: utf-8


import os, sys, time
import numpy as np
from astropy.io import fits
from astropy import units as u


import nrm_analysis.misctools.utils as utils
#from nrm_analysis.misctools.utils import Affine2d
#from nrm_analysis.misctools.utils import avoidhexsingularity
from nrm_analysis.fringefitting.LG_Model import NRM_Model

from nrm_analysis import nrm_core, InstrumentData
print(InstrumentData.__file__)

arcsec2rad = u.arcsec.to(u.rad)
um = 1.0e-6
PIXELSCALE_r =  0.0656 * arcsec2rad
FILT = "F430M"
MONOF430M = np.array([(1.0, 4.3e-6),])

def image2oi(mirfile, oitxtdir, oversample=3, 
         filt="F430M", bp=None, affine2d=None,
         prec=4, debug=True, verbose=False):
    """
    function to extract text observables from list of fits data files
    in mirage format (with expected keywords, and a second HDU that is a
    datacube of slices (eg number of exposures) that come out of second
    stage STScI pipeline processing).
    input:
        mirfile (str or list of strs): full path(s) of input image fits file to be analyzed
        oitxtdir (str) : directory where text observables are to be  written, 
                         a sub_dir_str of the object rootname is created for the 
                         output txt oi observables, one per input data slice
        oversample (int): pixel oversampling for implaneia fringe fitting
        prec (int) optional precision for float numpy printout
    """

    np.set_printoptions(precision=prec)

    # Create InstrumentData instantiation for NIRISS...
    # convenient BANDPASS variables are defined here: use what you need.
    #f debug:
    #   bp = np.array([(1.0, 4.3e-6),]) # for speedy development, exact simulations
    niriss = InstrumentData.NIRISS(filt, bandpass=bp, affine2d=affine2d)

    # * FringeFitter(..., save_txt_only=True, ...) switch off fits file writing
    # * set interactive=False unless you are a beginner
    ff_t = nrm_core.FringeFitter(niriss, datadir="nonsense", savedir=oitxtdir,
                                 oversample=oversample, interactive=False) 
                                                            
    # To use the parallelization option, set threads=n_threads in fit_fringes()
    # fit_fringes calls InstrumentData.read_data() which assigns the sub_dir_string used 
    # to write out the per slice  observables' text files under the object name (without
    # ".fits") sub directory.
    # fit_fringes works for fn or [fns]
    ff_t.fit_fringes(mirfile)

    print("oversample {:d} used in modelling the data".format(oversample))
    print("observables text files in ", oitxtdir)
    print("observables text files by image file root/* under ",  oitxtdir)
    ###print("cal observables in subdir", csavedir)
    if debug:
        print("Current working directory is ", os.getcwd())
        print("InstrumentData is file: ", InstrumentData.__file__)



def create_data(imdir, rot, ov):
    """ imdir: directory for simulated fits image data
        rot: pupil rotation in degrees
        ov: oversample for simulation
        Writes sim data to fitsimdir
    """
    npix = 81
    wave = 4.3e-6  # SI 
    fnfmt = '/psf_nrm_{2:.1f}_{0}_{1}_rot{3:.3f}d.fits' # expects strings of  %(npix, holeshape, wave/um, rot_d)

    rot = utils.avoidhexsingularity(rot) # in utils
    affine_rot = utils.Affine2d(rotradccw=np.pi*rot/180.0, name="rot{0:.3f}d".format(rot)) # in utils

    jw = NRM_Model(mask='jwst', holeshape='hex', affine2d=affine_rot)
    jw.set_pixelscale(PIXELSCALE_r)
    jw.simulate(fov=81, bandpass=MONOF430M, over=ov)

    psffn = fnfmt.format(npix, 'hex', wave/um, rot)
    fits.writeto(imdir+psffn, jw.psf, overwrite=True)
    header = fits.getheader(imdir+psffn)
    header = utils.affinepars2header(header, affine_rot)
    fits.update(imdir+psffn, jw.psf, header=header)
    del jw

    return psffn  # filename only


def rotsim_mir(rotdegs=0.0, ov=1):
    """ This function sets up directory and file names, writes a 2D file,
        and then mirage-izes it to disk.
        Data is written to a hardcoded ddata directory at the momet:
            ~/data/niriss_verification/*"
        A subdirectory labelled with oversampling is created for implaneia analysis output.
        rotdegs: pupil rotation in degrees
    """
    home = os.path.expanduser('~')
    import getpass
    username = getpass.getuser()

    # Mirage header needed from:
    mirext = "_mir"
    if (username=="anand"):
        mirexample = home + \
                "/gitsrc/ImPlaneIA/example_data/example_niriss/"+\
                "jw00793001001_01101_00001_nis_cal.fits" 
    else:
        mirexample = home + \
             "/ImplaneIA/notebooks/simulatedpsfs/" + \
             "jw00793001001_01101_00001_nis_cal.fits" 

    ov = 1 # oversampling for simulation of psf

    # create simdata and output observables directories data
    fitsimdir = home+"/data/niriss_verification/"
    if not os.path.exists(fitsimdir):
        os.mkdir(fitsimdir)
        print("Created image simulation data directory:\n\t", fitsimdir)

    oidir = fitsimdir + "ov{:d}".format(ov)
    if not os.path.exists(oidir):
        os.mkdir(oidir)
        print("Created oi text output file directory:\n\t", oidir)
    else:
        print("Using existing oi text output file directory:\n\t", oidir)

    simfn = create_data(fitsimdir, rotdegs, ov) 
    # get back miragefile.fits string (not full path)
    mirfn = utils.amisim2mirage(fitsimdir, simfn.replace(".fits",""), mirexample, FILT)
    if mirfn[0] == '/': mirfn=mirfn[1:]
    print("mirage-like file", mirfn, 'written in', fitsimdir)
    
    simdata = fits.getdata(fitsimdir+mirfn, 1)[0,:,:]
    print("simdata.shape", simdata.shape)

    from nrm_analysis import find_affine2d_parameters as FAP
    mx, my, sx,sy, xo,yo, = (1.0,1.0, 0.0,0.0, 0.0,0.0)
    rotsearch_d = (8.0, 9.0, 10.0, 11.0, 12.0)
    wave = 4.3e-6  # SI  - we won't know from the data what lam_c -  comes from outside?
                   # real-life - use apprpriate bandpass array
    aff_rot_measured = FAP.find_rotation(simdata,
                      rotsearch_d, mx, my, sx,sy, xo,yo,
                      PIXELSCALE_r, 80, wave, ov, 'hex', outdir=fitsimdir+"/rotdir/")

    image2oi(fitsimdir+mirfn, oidir, oversample=ov, filt=FILT, bp=MONOF430M, prec=4, affine2d=aff_rot_measured)

    aff_rot_measured.show()


if __name__ == "__main__":
    rotsim_mir(rotdegs=10.0, ov=1.0)
