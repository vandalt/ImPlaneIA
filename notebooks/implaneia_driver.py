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

#mirexample = os.path.expanduser('~') + \
#        "/gitsrc/ImPlaneIA/example_data/example_niriss/" + \
#        "jw00793001001_01101_00001_nis_cal.fits"
mirexample = os.path.expanduser('~') + \
         "/ImplaneIA/notebooks/simulatedpsfs/" + \
         "jw00793001001_01101_00001_nis_cal.fits"

fov = 79
filt="F430M"
lamc = 4.3e-6
oversample=11
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


def analyze_data(fitsfn, subdirout="", affine2d=None, psf_offset_find_rotation = (0.0,0.0), psf_offset_ff = None, rotsearch_d=None, set_pistons=None):
    """ returns: affine2d (measured or input), psf_offset_find_rotation (input), psf_offset_ff (input or found), fringe pistons/r (found)
    """

    print("analyze_data: input file", fitsfn, end="")

    data = fits.getdata(fitsfn)
    dim = data.shape[1]

    mx, my, sx,sy, xo,yo, = (1.0,1.0, 0.0,0.0, 0.0,0.0)

    if affine2d is None:
        print(" analyze_data: Finding affine2d...")
        affine2d = FAP.find_rotation(data[0,:,:], psf_offset_find_rotation,
                      rotsearch_d, mx, my, sx, sy, xo,yo,
                      PIXELSCALE_r, dim, bandpass, oversample, holeshape, outdir=fitsimdir)
        print("analyze_data:  Using measured affine2d...", affine2d.name)
    else:
        print("analyze_data:  Using incoming affine2d ", affine2d.name)

    niriss = InstrumentData.NIRISS(filt, bandpass=bandpass, affine2d=affine2d)
    ff_t = nrm_core.FringeFitter(niriss, psf_offset_ff=psf_offset_ff, datadir=fitsimdir, savedir=fitsimdir+subdirout,
                                 oversample=oversample, interactive=False)

    ff_t.fit_fringes(fitsfn)
    examine_residuals(ff_t)

    np.set_printoptions(formatter={'float': lambda x: '{:+.2e}'.format(x)}, linewidth=80)
    print("analyze_data: fringepistons/rad", ff_t.nrm.fringepistons)
    default_printoptions()
    return affine2d, psf_offset_find_rotation, ff_t.nrm.bestcenter, ff_t.nrm.fringepistons


def simulate_data(affine2d=None, psf_offset_det=None, pistons_w=None):

    print(" simulate_data: ", affine2d.name)

    #************ADD PISTONS TO PERFECT DATA******************
    pistons_w = pistons_w - pistons_w.mean()
    pistons_m =  pistons_w * lamc
    pistons_r =  pistons_w * 2.0 * np.pi

    utils.show_pistons_w(pistons_w, str="simulate_data: input pistons")

    np.set_printoptions(precision=4, formatter={'float': lambda x: '{:+.1e}'.format(x)}, linewidth=80)
    print(" simulate_data: pistons_um", pistons_m/1e-6)
    print(" simulate_data: pistons_r", pistons_r)
    print(" simulate_data: pistons_w stdev/w", round(pistons_w.std(),4))
    print(" simulate_data: pistons_w stdev/r", round(pistons_w.std()*2*np.pi,4))
    print(" simulate_data: pistons_w mean/r",  round(pistons_w.mean()*2*np.pi,4))
    default_printoptions()


    #***********ADD EFFECTS TO SIMULATE*******************
    jw = NRM_Model(mask='jwst', holeshape="hex", affine2d=affine2d)
    jw.set_pixelscale(pixelscale_as*arcsec2rad)
    jw.set_pistons(pistons_m)
    print(" simulate: ", psf_offset_det, type(psf_offset_det))
    jw.simulate(fov=fov, bandpass=bandpass, over=oversample, psf_offset=psf_offset_det)
    fits.PrimaryHDU(data=jw.psf).writeto(fitsimdir+"all_effects_data.fits",overwrite=True)

    #**********Convert simulated data to mirage format.*******
    utils.amisim2mirage(fitsimdir, ("all_effects_data",), mirexample, filt)


if __name__ == "__main__":

    identity = utils.Affine2d(rotradccw=utils.avoidhexsingularity(0.0),
                              name="affrot_{0:+.3f}deg".format(0.0))
    no_pistons = np.zeros((7,)) * 1.0
    _psf_offset_det = (0.48, 0.0)
    no_psf_offset = (0.0, 0.0)

    rot = 2.0
    rot = utils.avoidhexsingularity(rot)
    aff = utils.Affine2d(rotradccw=np.pi*rot/180.0, name="{0:.0f}".format(rot))
    _rotsearch_d = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    _rotsearch_d = np.arange(-3, 3.1, 1)

    #std dev 1, 7 holes, diffraction-limited @ 2um we're at 4um
    _pistons_w = 0.5 * np.random.normal(0,1.0, 7) / 14.0

    simulate_data(affine2d=aff, psf_offset_det=_psf_offset_det, pistons_w=_pistons_w)

    """
    Implaneia uses the center of the brightest pixel as the coordinate system to calculate psf offsets for fringefitting.
    With non-zero pistons and slight rotation, the offsets used to generate the verificaton data have “true center” that is not inside the brightest pixel.
    Hence a psf_offset (-0.52, 0.0) in implaneia’s local centroid-finding algorithm places the center in the pixel to the left of the brightest pixel.
    which is the correct result.
    Tests below are specific to analyzing data simulated with rot=2.0 deg, psf offsets (0.48, 0.0) and pistons in waves = pistons_w - pistons_w.mean() where pistons_w = 0.5 * np.random.normal(0,1.0, 7) / 14.0
    """


    args_odd_fov = [[None, (0.0,0.0), None, _rotsearch_d],
            [None, (0.0,0.0), (-0.5199,0.0), _rotsearch_d],
            [None, (0.48,0.0), None, _rotsearch_d],
            [aff,(0.0,0.0),None, None],
            [None, (0.48,0.0), (-0.5199,0.0), _rotsearch_d],
            [aff, (0.48,0.0), (-0.5199,0.0), _rotsearch_d],
            ]

    args_even_fov= [[None, (0.0,0.0), None, _rotsearch_d],
            [None, (0.0,0.0), (-0.01983, -0.4998), _rotsearch_d],
            [None, (0.48,0.0), None, _rotsearch_d],
            [aff,(0.0,0.0),None, None],
            [None, (0.48,0.0), (-0.01983, -0.4998), _rotsearch_d],
            [aff, (0.48,0.0), (-0.01983, -0.4998), _rotsearch_d],
            ]

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

        _aff, _psf_offset_r, _psf_offset_ff, fringepistons = analyze_data(df, affine2d=None,
                                                      psf_offset_find_rotation = (0.0,0.0), psf_offset_ff = None,
                                                      rotsearch_d=_rotsearch_d)

        #Analyze data with multiple sets of parameters

        for i,j in enumerate(args):

            print("Analysis parameters set:",i, "Affine2d", j[0], "psf_offset_find_rotation", j[1],"psf_offset_ff", j[2], "rotsearch_d",j[3])
            _aff, _psf_offset_r, _psf_offset_ff, fringepistons = analyze_data(df,"observables%d/"%i, affine2d=j[0],
                                                      psf_offset_find_rotation = j[1], psf_offset_ff = j[2],
                                                      rotsearch_d=j[3])
            print("  rotation search deg             ",_rotsearch_d)
            np.set_printoptions(formatter={'float': lambda x: '{:+.2e}'.format(x)}, linewidth=80)
            print("  affine rot used                 ", _aff.name, )
            np.set_printoptions(formatter={'float': lambda x: '{:+.3f}'.format(x)}, linewidth=80)
            print("  input psf offsets               ", np.array(_psf_offset_det))
            print("  psf offset used to find rotation", np.array(_psf_offset_r))
            print("  psf offset used by fringefitter ", np.array(_psf_offset_ff))
            utils.compare_pistons(2*np.pi*(_pistons_w- _pistons_w.mean()), fringepistons)
            print("implaneia output in: ", fitsimdir, "\n")

