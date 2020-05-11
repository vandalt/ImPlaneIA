#! /usr/bin/env python
"""
Reads ImPlaneIA output text of [CPs,phases,amplitudes]_nn.txt in one directory
    OI Observables are: np arrays (nslices,nobservables) in shape.
        self.fp = np.zeros((self.nslices, self.nbl))
        self.fa = np.zeros((self.nslices, self.nbl))
        self.cp = np.zeros((self.nslices, self.ncp))
        if len(self.observables) == 4:
            self.ca = np.zeros((self.nslices, self.nca))
"""
import os
import numpy as np

from nrm_analysis.misctools.implane2oifits import ObservablesFromText 


# OI observables from text for 7-hole mask...
# nholes in first argument.  
# implaneia text oi observables in the second argument, the 'oitxtpath'
oiobs = ObservablesFromText(7, 
                            os.path.expanduser('~')+"/gitsrc/ImPlaneIA/example_data/example_niriss/bin_cal_oitxt" ,
                            verbose=True)
oiobs.showdata()
