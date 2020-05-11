#!/usr/bin/env python
# coding: utf-8

# ImPlaneIA fringe fitting on Anthony Soulains's two science 
# targets, two calibrators. 2020

import glob
import os, sys, time
from astropy.io import fits
import numpy as np

from nrm_analysis import nrm_core, InstrumentData
from nrm_analysis.misctools import utils
print(InstrumentData.__file__)

def main(mirfile, oitxtdir, oversample=3, 
         filt="F430M", bp=None,
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
    if debug:
        bp = np.array([(1.0, 4.3e-6),]) # for speedy development, exact simulations
    niriss = InstrumentData.NIRISS(filt, bandpass=bp)

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
    print("observables text files by image file root/*.txt under ",  oitxtdir)
    if debug:
        print("Current working directory is ", os.getcwd())
        print("InstrumentData is file: ", InstrumentData.__file__)

if __name__ == "__main__":
    """ 
        to get the flow of the program correct, or when debugging, use 
        bpmono, oversample=1
    """
    home = os.path.expanduser('~')

    input_fitsimdir = home+"/Downloads/asoulain_arch2019.12.07/Simulated_data/"

    # Create full path and filename or list of the same, 'imfile_:
    mirfits_fn = [
                 "t_dsk_100mas__F430M_81_flat_x11__00_mir.fits",
                 "c_dsk_100mas__F430M_81_flat_x11__00_mir.fits",
                 "c_bin_s=120mas__F430M_81_flat_x11__00_mir.fits",
                 "t_bin_s=120mas__F430M_81_flat_x11__00_mir.fits",
                 ]
    imfile_ = []
    for mirfn in mirfits_fn:
        imfile_.append(input_fitsimdir + mirfn) # full path

    # Construct your output OI txext files' top directory for all extractions in this run"
    oversample_ = 1
    oitxtdir_ = input_fitsimdir + "ov{:d}".format(oversample_)
    if not os.path.exists(oitxtdir_):
        os.mkdir(oitxtdir_)
        print("Created oi text output file directory:\n\t", oitxtdir_)
    else:
        print("Using existing oi text output file directory:\n\t", oitxtdir_)
    
    main(imfile_, oitxtdir_, oversample=oversample_, filt="F430M", prec=3, debug=True)
