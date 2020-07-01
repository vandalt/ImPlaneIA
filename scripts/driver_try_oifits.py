#! /usr/bin/env python

import os
import numpy as np
import scipy
from astropy.io import fits
from astropy import units as u
import sys
import string
import nrm_analysis
from nrm_analysis.fringefitting.LG_Model import NRM_Model
from nrm_analysis.misctools import utils  
from nrm_analysis import nrm_core, InstrumentData
from nrm_analysis import find_affine2d_parameters as FAP
from pathlib import Path
from nrm_analysis.misctools.utils import Affine2d


np.set_printoptions(precision=4, linewidth=160)

home = os.path.expanduser('~')
fitsimdir = home+"/data/implaneia/niriss_verification/test_all_residuals/"
if not os.path.exists(fitsimdir):  
    os.makedirs(fitsimdir)

mirexample = os.path.expanduser('~') + '/gitsrc/' +\
        "/ImPlaneIA/example_data/example_niriss/" + \
        "jw00793001001_01101_00001_nis_cal.fits"

fov = 79
filt="F430M"
lamc = 4.3e-6
oversample=3
bandpass = np.array([(1.0, lamc),])

pixelscale_as=0.0656
arcsec2rad = u.arcsec.to(u.rad)
PIXELSCALE_r =  pixelscale_as * arcsec2rad
holeshape='hex'

datafiles = (fitsimdir+'all_effects_data_mir.fits',)
np.random.seed(100)

def default_printoptions():
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8,
    suppress=False, threshold=1000, formatter=None)

def cp_var(nh, cps):
    """ True standard deviation given non-independent closure phases """
    return  ((cps - cps.mean())**2).sum() / scipy.special.comb(nh-1,2)
def ca_var(nh, cas):
    """ True standard deviation given non-independent closure amplitudes """
    return  ((cas - cas.mean())**2).sum() / scipy.special.comb(nh-3, 2)

def examine_residuals(ff, trim=36):
    """ input: FringeFitter instance after fringes are fit """

    print("\n\texamine_residuals: FIT QUALITY:")
    print(" Standard deviation & variance take into acount reduced DOF of all CP's and CAs")
    print("   Closure phase mean {:+.4f}  std dev {:.2e}  var {:.2e}".format(ff.nrm.redundant_cps.mean(),
                                                  np.sqrt(cp_var(ff.nrm.N,ff.nrm.redundant_cps)),
                                                         cp_var(ff.nrm.N, ff.nrm.redundant_cps)))
    print("   Closure ampl  mean {:+.4f}  std dev {:.2e}  var {:.2e}".format(ff.nrm.redundant_cas.mean(),
                                                  np.sqrt(cp_var(ff.nrm.N,ff.nrm.redundant_cas)),
                                                         cp_var(ff.nrm.N, ff.nrm.redundant_cas)))

    np.set_printoptions(precision=3, formatter={'float': lambda x: '{:+.1e}'.format(x)}, linewidth=80)
    print(" Normalized residuals trimmed by {:d} pixels from each edge".format(trim))
    print((ff.nrm.residual/ff.datapeak)[trim:-trim,trim:-trim])
    print(" Normalized residuals max and min: {:.2e}, {:.2e}".format( ff.nrm.residual.max() / ff.datapeak,
                                                                      ff.nrm.residual.min() / ff.datapeak))
    default_printoptions()


def analyze_data(fitsfn, observables_dir="", affine2d=None,
                         psf_offset_find_rotation = (0.0,0.0),
                         psf_offset_ff = None, 
                         rotsearch_d=None,
                         set_pistons=None):
    """ 
        returns: affine2d (measured or input), 
        psf_offset_find_rotation (input),
        psf_offset_ff (input or found),
        fringe pistons/r (found)
    """

    print("analyze_data: input file", fitsfn)

    data = fits.getdata(fitsfn)
    dim = data.shape[1]

    mx, my, sx,sy, xo,yo, = (1.0,1.0, 0.0,0.0, 0.0,0.0)

    niriss = InstrumentData.NIRISS(filt, bandpass=bandpass, affine2d=affine2d)
    ff_t = nrm_core.FringeFitter(niriss, psf_offset_ff=psf_offset_ff, datadir=fitsimdir, savedir=fitsimdir+observables_dir,
                                 oversample=oversample, interactive=False)
    ff_t.fit_fringes(fitsfn)
    examine_residuals(ff_t)

    np.set_printoptions(formatter={'float': lambda x: '{:+.2e}'.format(x)}, linewidth=80)
    print("analyze_data: fringepistons/rad", ff_t.nrm.fringepistons)
    default_printoptions()
    return affine2d, psf_offset_find_rotation, ff_t.nrm.psf_offset, ff_t.nrm.fringepistons


def simulate_data(affine2d=None, psf_offset_det=None, pistons_w=None):

    np.set_printoptions(precision=4, formatter={'float': lambda x: '{:+.1e}'.format(x)}, linewidth=80)
    default_printoptions()


    #***********ADD EFFECTS TO SIMULATE*******************
    jw = NRM_Model(mask='jwst', holeshape="hex", affine2d=affine2d)
    jw.set_pixelscale(pixelscale_as*arcsec2rad)
    jw.simulate(fov=fov, bandpass=bandpass, over=oversample)
    fits.PrimaryHDU(data=jw.psf).writeto(fitsimdir+"all_effects_data.fits",overwrite=True)

    #**********Convert simulated data to mirage format.*******
    utils.amisim2mirage(fitsimdir, ("all_effects_data",), mirexample, filt)


if __name__ == "__main__":

    simulate_data()

    args_odd_fov = [[None, (0.0,0.0), None, None], ]
    args_even_fov= [[None, (0.0,0.0), None, None], ]

    np.set_printoptions(formatter={'float': lambda x: '{:+.2e}'.format(x)}, linewidth=80)

    for df in datafiles:
        print("__main__: ")
        np.set_printoptions(formatter={'float': lambda x: '{:+.2e}'.format(x)}, linewidth=80)
        print("           analyzing", df)
        data = fits.getdata(df)

        if (data.shape[1] % 2) == 0:
            args = args_even_fov
        else:
            args = args_odd_fov

        #Simple case with one set of parameters.
        #
        #_aff, _psf_offset_r, _psf_offset_ff, fringepistons = \
        #                                analyze_data(df, affine2d=None,
        #                                                 psf_offset_find_rotation = (0.0,0.0), 
        #                                                 psf_offset_ff = None,
        #                                                 rotsearch_d=_rotsearch_d)


        #Analyze data with multiple sets of parameters
        for iarg,arg in enumerate(args):

            print("\nanalyze_data arguments:", "set", iarg, ":",  end=' ')

            _aff, _psf_offset_r, _psf_offset_ff, fringepistons = \
                                        analyze_data(df, "obs%d/"%iarg)

            np.set_printoptions(formatter={'float': lambda x: '{:+.2e}'.format(x)}, linewidth=80)
            np.set_printoptions(formatter={'float': lambda x: '{:+.3f}'.format(x)}, linewidth=80)
            print("implaneia output in: ", fitsimdir, "\n")
