import unittest
import os
import glob
import sys
import numpy as np
from numpy.random import default_rng
from astropy.io import fits
from astropy import units as u

from nrm_analysis.misctools.utils import Affine2d

"""
    Test Affine2d class' Identity transformation
    anand@stsci.edu 2018.06.21

    run with pytest -s _moi_.py to see stdout on screen
"""

class Affine2dTestCase(unittest.TestCase):

    def setUp(self):

        # No test data on disk... make tst data here.

        mx, my = 1.0, 1.0
        sx, sy= 0.0, 0.0
        xo, yo= 0.0, 0.0
        self.aff_id = Affine2d(mx=mx,my=my, 
                               sx=sx,sy=sy, 
                               xo=xo,yo=yo, name="Ideal")
        # create nvecs random x,y locations
        nvecs = 5
        self.x = np.random.uniform(-10.0, 10.0, nvecs)
        self.y = np.random.uniform(-10.0, 10.0, nvecs)

    def test_id_xy(self):
        """ 
            check that distortFargs returns same x,y vectors as those passed to it
        """
        xprime, yprime = self.aff_id.distortFargs(self.x, self.y)
        self.assertTrue(np.abs(xprime - self.x).sum() + np.abs(yprime - self.y).sum() 
                      < 1e-15,  'test_affine2d_identity failed to preserve x,y')

    def test_id_phase(self):
        """ 
            check that distortphase returns appropriate phasor  1.0 + 0j
        """
        phasor = self.aff_id.distortphase(self.x, self.y)
        self.assertTrue(np.abs(phasor.real - 1.0).sum() + np.abs(phasor.imag.sum()) 
                      < 1e-15,  'test_affine2d_identity failed to preserve phase')

if __name__ == "__main__":
    unittest.main()

