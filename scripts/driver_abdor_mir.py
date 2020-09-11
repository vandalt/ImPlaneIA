#! /usr/bin/env python

import os
import numpy as np
from astropy.io import fits
from astropy import units as u
import sys
import string
import matplotlib.pylab as plot
import nrm_analysis
from nrm_analysis.fringefitting.LG_Model import NRM_Model
from nrm_analysis.misctools import utils  
from nrm_analysis import nrm_core, InstrumentData
from nrm_analysis import find_affine2d_parameters as FAP
from pathlib import Path
from nrm_analysis.misctools.utils import Affine2d


np.set_printoptions(precision=4, linewidth=160)

def examine_residuals(ff, trim=36):
    """ input: FringeFitter instance after fringes are fit """

    print("\nExamine_residuals, standard deviations & variances of *independent* CP's and CAs:")
    print("   Closure phase mean {:+.4f}  std dev {:.2e}  var {:.2e}".format(ff.nrm.redundant_cps.mean(),
                                               np.sqrt(utils.cp_var(ff.nrm.N,ff.nrm.redundant_cps)),
                                                      utils.cp_var(ff.nrm.N, ff.nrm.redundant_cps)))

    print("   Closure amp   mean {:+.4f}  std dev {:.2e}  var {:.2e}".format(ff.nrm.redundant_cas.mean(),
                                               np.sqrt(utils.cp_var(ff.nrm.N,ff.nrm.redundant_cas)),
                                                      utils.cp_var(ff.nrm.N, ff.nrm.redundant_cas)))

    print("    Fringe amp   mean {:+.4f}  std dev {:.2e}  var {:.2e}".format(ff.nrm.fringeamp.mean(),
                                                                             ff.nrm.fringeamp.std(),
                                                                             ff.nrm.fringeamp.var()))

    np.set_printoptions(precision=3, formatter={'float': lambda x: '{:+.1e}'.format(x)}, linewidth=80)
    print(" Normalized residuals central 6 pixels")
    tlo, thi = (ff.nrm.residual.shape[0]//2 - 3, ff.nrm.residual.shape[0]//2 + 3)
    print((ff.nrm.residual/ff.datapeak)[tlo:thi,tlo:thi])
    print(" Normalized residuals max and min: {:.2e}, {:.2e}".format( ff.nrm.residual.max() / ff.datapeak,
                                                                      ff.nrm.residual.min() / ff.datapeak))
    utils.default_printoptions()


def analyze_data(fitsfn=None, fitsimdir=None, affine2d=None,
                         psf_offset_find_rotation = (0.0,0.0),
                         psf_offset_ff = None, 
                         rotsearch_d=None,
                         set_pistons=None,
                         oversample=3,
                         mnem='',
                         firstfew=None,
                         verbose=False):
    """ 
        returns: affine2d (measured or input), 
        psf_offset_find_rotation (input),
        psf_offset_ff (input or found),
        fringe pistons/r (found)
    """

    if verbose: print("analyze_data: input", fitsimdir+fitsfn)
    if verbose: print("analyze_data: oversample", oversample)

    fobj = fits.open(fitsimdir+fitsfn)

    if verbose: print(fobj[0].header['FILTER'])
    niriss = InstrumentData.NIRISS(fobj[0].header['FILTER'], 
                                   bpexist=False,
                                   firstfew=firstfew,# read_data truncation to only read first few slices...
                                   )

    
    ff = nrm_core.FringeFitter(niriss, 
                                 datadir=fitsimdir, 
                                 savedir=fitsimdir,
                                 oversample=oversample,
                                 oifprefix="ov{:d}_".format(oversample)+mnem,
                                 interactive=False,
                                 save_txt_only=True)

    ff.fit_fringes(fitsimdir+fitsfn)
    examine_residuals(ff)

    np.set_printoptions(formatter={'float': lambda x: '{:+.2e}'.format(x)}, linewidth=80)
    if verbose: print("analyze_data: fringepistons/rad", ff.nrm.fringepistons)
    utils.default_printoptions()
    return affine2d, psf_offset_find_rotation, ff.nrm.psf_offset, ff.nrm.fringepistons


def main(fitsimdir, ifn, oversample=3, mnem='', firstfew=None, verbose=False):
    """ 
    fitsimdir: string: dir containing data file
    ifn: str inout file name

    """
    
    np.set_printoptions(formatter={'float': lambda x: '{:+.2e}'.format(x)}, linewidth=80)
    if verbose: print("main: ", ifn)
    if verbose: print("main: fitsimdir", fitsimdir)

    aff, psf_offset_r, psf_offset_ff, fringepistons = analyze_data(ifn, fitsimdir, 
                                                                   oversample=oversample,
                                                                   mnem=mnem,
                                                                   firstfew=firstfew,
                                                                   verbose=verbose)
    del aff
    del psf_offset_r
    del psf_offset_ff
    del fringepistons
    #plot.show()


if __name__ == "__main__":

    # MIRAGE VERSION
    """
    jw01093001001_01101_00001_nis_cal.fits
    jw01093002001_01101_00001_nis_cal.fits
    jw01093001001_01101_00001_nis_calints.fits
    jw01093002001_01101_00001_nis_calints.fits
    """
    mirdatafiles = ['jw01093001001_01101_00001_nis_calints.fits',
                    'jw01093002001_01101_00001_nis_calints.fits',
                   ]

    FIRSTFEW = None # eg 5, or None to analyze all slices
    FIRSTFEW = 5 # eg 5, or None to analyze all slices
    OVERSAMPLE = 7
    print('FIRSTFEW', FIRSTFEW, 'OVERSAMPLE', OVERSAMPLE)

    datasuperdir = os.path.expanduser('~') + '/data/implaneia/mirage_nis019/'

    count = 0
    for fnmir in mirdatafiles:
        print('\nAnalyzing\n   ',  count, fnmir.replace('.fits',''), end=' ')
        hdr = fits.getheader(datasuperdir+fnmir)
        print(hdr['FILTER'], end=' ')
        print(hdr['TARGNAME'], end=' ')
        print(hdr['TARGPROP'])
        # next line for convenient use in oifits writer which looks up target online
        catname = hdr['TARGPROP'].replace('-','') # for target lookup on-line, otherwise UNKNOWN used
        fits.setval(datasuperdir+fnmir, 'TARGNAME', value=catname)
        fits.setval(datasuperdir+fnmir, 'TARGPROP', value=catname)
        main(fitsimdir=datasuperdir,
             ifn=fnmir, 
             oversample=OVERSAMPLE, 
             mnem='',
             firstfew=FIRSTFEW,
             verbose=False)
        plot.close()
        count += 1
