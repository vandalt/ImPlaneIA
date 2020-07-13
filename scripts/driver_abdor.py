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


    mirexample = os.path.expanduser('~') + '/gitsrc/' +\
        "/ImPlaneIA/example_data/example_niriss/" + \
        "jw00793001001_01101_00001_nis_cal.fits"

    datasuperdir = os.path.expanduser('~') + '/data/implaneia/amisim_udem_nis019/'
    filters = ['F480M', 'F430M', 'F380M']
    datafiles_byfilter = {}
    # create file names for 'simple' CAP-019 observations...
    for filt in filters:
        datafiles_byfilter[filt] = ['t_ABDor_{:s}__myPSF_{:s}_Obs1_00'.format(filt,filt),
                                    't_ABDor_{:s}__myPSF_{:s}_Obs2_00'.format(filt,filt),
                                    't_HD37093_{:s}__myPSF_{:s}_Obs4_00'.format(filt,filt),
                                    't_HD36805_{:s}__myPSF_{:s}_Obs6_00'.format(filt,filt),
                                   ] # 12 oifits files
                                   # These file names require the addition of Obs_nn to oifn so
                                   # we don't overwrite first since there are two ABDor observations
                                   # With MAST files we might not need this mnemonic

    print(len(datafiles_byfilter[filt]), 'datafiles', len(filters), 'filters')

    # convert to _mir files, add target name & prop from amisim filename t_targetname_...
    for filt in filters:
        fitsimdir = datasuperdir + filt + '/'
        utils.amisim2mirage(fitsimdir, datafiles_byfilter[filt], mirexample, filt)
        # following target name loop only needs to occur once... but doesn't hurt
        add_targetname = True
        if add_targetname: # put in target name for oif to pick up, use in prefix
            for fitsfile in datafiles_byfilter[filt]:
                targname = fitsfile.split('_')[1].upper()
                mirfitsfn = fitsfile+'_mir.fits'
                fobj = fits.open(fitsimdir+mirfitsfn)
                fobj[0].header['TARGNAME'] = (targname, 'parsed from file name')
                fobj[0].header['TARGPROP'] = (targname, 'parsed from file name')
                fobj.writeto(fitsimdir+mirfitsfn, overwrite=True)
                print("mirage file targname", targname)

    count = 0
    for filt in filters:
        for fn in  datafiles_byfilter[filt]:
            print('__main__  analyzing', count, 'of', len( datafiles_byfilter[filt])*len(filters),  fn)
            print(len(datafiles_byfilter), len(filters))
            fn_mir = fn+'_mir.fits'
            main(fitsimdir=datasuperdir+filt+'/', 
                 ifn=fn_mir, 
                 oversample=1, 
                 mnem=fn.split('_')[-2] + '_' +fn.split('_')[-1] + '-',   # eg 'Obs1_00_' 
                 firstfew=5,
                 verbose=False)
            plot.close()
            count += 1
