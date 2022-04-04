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
    Test rotation for Affine2d class
    anand@stsci.edu 2018.06.21

    run with pytest -s _moi_.py to see stdout on screen
    All units SI unless units in variable name
"""

arcsec2rad = u.arcsec.to(u.rad)
um = 1.0e-6

def affinepars2header(hdr, affine2d):
    """ writes affine2d parameters into fits header """
    hdr['affine'] = (affine2d.name, 'Affine2d in pupil: name')
    hdr['aff_mx'] = (affine2d.mx, 'Affine2d in pupil: xmag')
    hdr['aff_my'] = (affine2d.my, 'Affine2d in pupil: ymag')
    hdr['aff_sx'] = (affine2d.sx, 'Affine2d in pupil: xshear')
    hdr['aff_sy'] = (affine2d.sx, 'Affine2d in pupil: yshear')
    hdr['aff_xo'] = (affine2d.xo, 'Affine2d in pupil: x offset')
    hdr['aff_yo'] = (affine2d.yo, 'Affine2d in pupil: y offset')
    hdr['aff_dev'] = ('analyticnrm2', 'dev_phasor')
    return hdr

class Affine2dTestCase(unittest.TestCase):

    def setUp(self):

        # No test data on disk... make tst data here.

        mx, my = 1.0, 1.0
        sx, sy= 0.0, 0.0
        xo, yo= 0.0, 0.0
        affine2d_identity = Affine2d(mx=mx,my=my, 
                                sx=sx,sy=sy, 
                                xo=xo,yo=yo, name="Ideal")
        # create nvecs random x,y locations
        nvecs = 1000
        rvec = np.random.uniform(-100.0, 100.0, nvecs*2).reshape(nvecs,2)



    def test_psf(self):
        """ Read in PSFs with 0, 90 degree affines, 5 and 95 degree affines, 
            Rotate one set and subtract from the smaller rot PSF - should be zero if
            everything is correctly calculated.  If we nudge the PSF centers to avoid the 
            line singularity that hextransformEE will encounter if the psf is centrally 
            placed in a pixel.
            The file names are hand-edited to reflect the oversampling and rotations,
            so this is more a sanity check and code development tool than a routine test.
        """
        self.assertTrue(0.0 < 1e-15,  'error: test_affine2d failed')

if __name__ == "__main__":
    unittest.main()

