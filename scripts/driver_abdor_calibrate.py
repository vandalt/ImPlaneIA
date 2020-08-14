#! /usr/bin/env python

import os
import numpy as np
from astropy.io import fits
from astropy import units as u
import sys
import glob
import string
import nrm_analysis
from nrm_analysis.misctools import oifits  
from nrm_analysis.misctools import implane2oifits

def main(alloifsdir):
    print('Searching',alloifsdir, 'for *.oifits files')

    filters = ['F480M', 'F430M', 'F380M']
    obslist = ['Obs_1_00','Obs_2_00','Obs_4_00', 'Obs_6_00']
    targlist = ['ABDOR', 'HD36805', 'HD37093']

    oifinputdict = {} # keys map to input fns without dir path or 'oifits'
    observables = {} # same keys as above
    oifs = glob.glob(alloifsdir+'*.oifits')
    sortedoifs = sorted(oifs)

    for oif in sortedoifs:
        # read into oifdicts (oif.split('.')[-2]).split('/')[-1]
        clef =  (oif.split('.')[-2]).split('/')[-1]
        oifinputdict[clef] = oifits.load(oif)
        # convert to internam obserbable in memory
        observables[clef] =  implane2oifits.DictToObservable( oifinputdict[clef] )

        print('  ', '{:55s}'.format(clef), 
              '{:10s}'.format(oifinputdict[clef]['info']['TARGET']),
              '{:10s}'.format(oifinputdict[clef]['info']['FILT']), 
              )
    """ UdeM sims
    ov1_Obs1_00-ABDOR_NIRISS_jwst_g7s6c_F380M_59112         ABDOR      F380M     
    ov1_Obs4_00-HD37093_NIRISS_jwst_g7s6c_F380M_59112       HD37093    F380M     
    ov1_Obs6_00-HD36805_NIRISS_jwst_g7s6c_F380M_59112       HD36805    F380M     

    ov1_Obs1_00-ABDOR_NIRISS_jwst_g7s6c_F430M_59112         ABDOR      F430M     
    ov1_Obs4_00-HD37093_NIRISS_jwst_g7s6c_F430M_59112       HD37093    F430M     
    ov1_Obs6_00-HD36805_NIRISS_jwst_g7s6c_F430M_59112       HD36805    F430M     

    ov1_Obs1_00-ABDOR_NIRISS_jwst_g7s6c_F480M_59112         ABDOR      F480M     
    ov1_Obs4_00-HD37093_NIRISS_jwst_g7s6c_F480M_59112       HD37093    F480M     
    ov1_Obs6_00-HD36805_NIRISS_jwst_g7s6c_F480M_59112       HD36805    F480M     

    ov1_Obs2_00-ABDOR_NIRISS_jwst_g7s6c_F380M_59112         ABDOR      F380M     
    ov1_Obs2_00-ABDOR_NIRISS_jwst_g7s6c_F430M_59112         ABDOR      F430M     
    ov1_Obs2_00-ABDOR_NIRISS_jwst_g7s6c_F480M_59112         ABDOR      F480M     

    obs_abd_37093_f380 =  implane2oifits.calibrate_observable(
                 observables["ov1_Obs1_00-ABDOR_NIRISS_jwst_g7s6c_F380M_59112"], 
                 observables["ov1_Obs4_00-HD37093_NIRISS_jwst_g7s6c_F380M_59112"])
    obs_abd_36805_f380 =  implane2oifits.calibrate_observable(
                 observables["ov1_Obs1_00-ABDOR_NIRISS_jwst_g7s6c_F380M_59112"], 
                 observables["ov1_Obs6_00-HD36805_NIRISS_jwst_g7s6c_F380M_59112"])
    """

if __name__ == "__main__":
               
    main(os.path.expanduser('~') +'/data/implaneia/amisim_udem_nis019/collected_oifits/scriptdev/')
