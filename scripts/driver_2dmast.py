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
    print(" Normalized residuals trimmed by {:d} pixels from each edge".format(trim))
    print((ff.nrm.residual/ff.datapeak)[trim:-trim,trim:-trim])
    print(" Normalized residuals max and min: {:.2e}, {:.2e}".format( ff.nrm.residual.max() / ff.datapeak,
                                                                      ff.nrm.residual.min() / ff.datapeak))
    utils.default_printoptions()


def analyze_data(fitsfn=None, fitsimdir=None, affine2d=None,
                         psf_offset_find_rotation = (0.0,0.0),
                         psf_offset_ff = None, 
                         rotsearch_d=None,
                         set_pistons=None,
                         oversample=3):
    """ 
        returns: affine2d (measured or input), 
        psf_offset_find_rotation (input),
        psf_offset_ff (input or found),
        fringe pistons/r (found)
    """

    print("analyze_data: input file", fitsfn)
    print("analyze_data: oversample", oversample)

    data = fits.getdata(fitsfn)
    fobj = fits.open(fitsfn)

    print(fobj[0].header['FILTER'])
    niriss = InstrumentData.NIRISS(fobj[0].header['FILTER'], bpexist=False)
    ff_t = nrm_core.FringeFitter(niriss, 
                                 datadir=fitsimdir, 
                                 savedir=fitsimdir,
                                 oversample=oversample,
                                 oifprefix="ov{:d}_".format(oversample),
                                 interactive=False)
    ff_t.fit_fringes(fitsfn)
    examine_residuals(ff_t)

    np.set_printoptions(formatter={'float': lambda x: '{:+.2e}'.format(x)}, linewidth=80)
    print("analyze_data: fringepistons/rad", ff_t.nrm.fringepistons)
    utils.default_printoptions()
    return affine2d, psf_offset_find_rotation, ff_t.nrm.psf_offset, ff_t.nrm.fringepistons


def main(fitsimdir, ifn, oversample=3):
    """ 
    fitsimdir: string: dir containing data file
    ifn: str: inout data file name, 2d cal or 3d calint MAST header fits file
    """
    
    fitsimdir = os.path.expanduser('~')+"/data/implaneia/niriss_development/2dinput/"
    if not os.path.exists(fitsimdir):  
        os.makedirs(fitsimdir)
    df = fitsimdir+'niscal_mir.fits'

    np.set_printoptions(formatter={'float': lambda x: '{:+.2e}'.format(x)}, linewidth=80)
    print("__main__: analyzing", ifn)

    aff, psf_offset_r, psf_offset_ff, fringepistons = analyze_data(df, fitsimdir, oversample=oversample)
    print("implaneia output in: ", fitsimdir, "\n")

    plot.show()

if __name__ == "__main__":

    main(fitsimdir=os.path.expanduser('~')+"/data/implaneia/niriss_development/2dinput/", 
         ifn='niscal_mir.fits',
         oversample=5
         )
