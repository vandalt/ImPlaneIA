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

def main(alloifsdir):
    print('Searching',alloifsdir, 'for *.oifits files')

    filters = ['F480M', 'F430M', 'F380M']
    obslist = ['Obs_1_00','Obs_2_00','Obs_4_00', 'Obs_6_00']
    targlist = ['ABDOR', 'HD36805', 'HD37093']

    oifinputdict = {}
    oifs = glob.glob(alloifsdir+'*.oifits')
    for oif in oifs:
        oifinputdict[(oif.split('.')[-2]).split('/')[-1]] = oifits.load(oif)

    print('oifits files found are read into a dict of oifits.load() dic structures')
    sortedobs = sorted(oifinputdict.keys())
    for clef in sortedobs:
        print('  ', '{:55s}'.format(clef), 
              '{:10s}'.format(oifinputdict[clef]['info']['TARGET']),
              '{:10s}'.format(oifinputdict[clef]['info']['FILT']), 
              )

if __name__ == "__main__":
               
    main(os.path.expanduser('~') +'/data/implaneia/amisim_udem_nis019/collected_oifits/scriptdev/')
