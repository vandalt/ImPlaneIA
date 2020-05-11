#! /usr/bin/env python
"""
Reads ImPlaneIA output text of [CPs,phases,amplitudes]_nn.txt in one directory
"""
import os
import numpy as np

from nrm_analysis.misctools.implane2oifits import ObservablesFromText 

oitxtpath =  os.path.expanduser('~')+"/gitsrc/ImPlaneIA/example_data/example_niriss/bin_cal_oitxt" #implaneia text oi observables in here 

# OI observables from text for 7-hole mask...
oiobs = ObservablesFromText(7, oitxtpath, verbose=True)
oiobs.showdata()





