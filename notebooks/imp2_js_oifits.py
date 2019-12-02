#! /usr/bin/env python

"""
Reads ImPlaneIA output text of [CPs,phases,apmlitudes]_nn.txt in one directory
Writes them out into oifits files in the same directory.

Chunkwide averaging might be a later development for real data.

anand@stsci.edu started 2019 09 26 

"""

import os, sys, time, glob
import numpy as np
from scipy.special import comb
from scipy.stats import sem, mstats



class ObservablesFromText():
    """
        anand@stsci.edu 2019_09_27
    """

    def __init__(self, nh, txtpath=None, 
                           oifpath=None, 
                           observables= ("phases","amplitudes","CPs","CAs"), 
                           angunit="radians",
                           verbose=True):
        """
        Methods: 
            readtxtdata(): read in implania to internal arrays
            showdata(): print out data at user-specified precision

        Input: 
            nh: number of holes in mask (int)
            textpaxt: directory where fringe observables' .txt files are stored (str)
            oifpath: directory (str).  If omitted, oifits goes to same directory as textpath
            nslices: 1 for eg unpolarized single filter observation, 
                     2 or more for polarized observations, 
                     many for IFU observations (int)
            qty_names: If ("phases", "amplitudes", "CPs", "CAs") for example - ImPlaneIA nomenclature
                       then files need to be named: "/phases_{0:02d}.txt".format(slc)
                       Three or four quantities (omitting CA is optional)
                       Order is relevant... pha, amp, cp [, ca]
            If you want to trim low and/or high ends of eg IFU spectral observables trim them
            on-the-fly before calling this routine.  Or write a trimming method for this module.

            ImPlaneIA saves fp cp in RADIANS.
        """

        print( "=== ObservablesFromText ===\nObject text observables' directorypath:\n    ", txtpath)
        self.txtpath = txtpath
        self.oifpath = oifpath
        self.verbose = verbose
        self.observables = observables
        # Assume same number of observable output files for each observable.
        # Each image analyzed has a phases, an amplitudes, ... txt output file in thie txtdir.
        # Each file might contain different numbers of individual quantities 
        print('txtpath/{0:s}*.txt'.format(self.observables[0]))
        #   - yea many fringes, more cp's, and so on.
        self.nslices = len(glob.glob(self.txtpath+'/{0:s}*.txt'.format(self.observables[0])))
        if verbose: print(self.nslices, "slices' observables text files found")
        self.nh = nh
        self.nbl = int(comb(self.nh, 2))
        self.ncp = int(comb(self.nh, 3))
        self.nca = int(comb(self.nh, 4))
        # arrays of observables, (nslices,nobservables) shape.
        self.fp = np.zeros((self.nslices, self.nbl))
        self.fa = np.zeros((self.nslices, self.nbl))
        self.cp = np.zeros((self.nslices, self.ncp))
        if len(self.observables) == 4:
            self.ca = np.zeros((self.nslices, self.nca))
        self.angunit = angunit
        if verbose: print("assumes angles in", angunit)
        if verbose: print("angle unit:", angunit)
        if angunit == 'radians': self.degree = 180.0 / np.pi
        else: self.degree = 1

        self._readtxtdata()
        if self.verbose: self._showdata()

    def _showdata(self, prec=4):
        """ set precision of your choice in calling this"""
        print('nh {0:d}  nslices {1:d}  nbl {2:d}  ncp {3:d}  '.format(
                          self.nh, self.nslices, self.nbl, self.ncp), end="")
        print("observables in np arrays with {:d} rows".format(self.nslices))

        if len(self.observables)==4: print('nca', self.nca)
        else: print()
        np.set_printoptions(precision=prec)

        print(self.fp.shape, "fp (degrees, but stored internally in radians):\n", self.fp*self.degree, "\n")
        print(self.fa.shape, "fa:\n", self.fa, "\n")

        print(self.cp.shape, "cp (degrees, but stored internally in radians):\n", self.cp*self.degree, "\n")
        if len(self.observables)==4:
            print(self.ca.shape, "ca:\n", self.ca, "\n")

    def _readtxtdata(self):
        # to only be called from init
        # loop through all the requested observables, 
        # read in the exposure slices of a data cube
        # or the wavelength slices of IFU?

        # set up files to read
        # What do we do for IFU or Pol?
        fnheads = [] # file name for each exposure (slice) in an image cube with nslices exposures
        if self.verbose: print("\tfile names that are being looked for:")
        for obsname in self.observables: 
            fnheads.append(self.txtpath+"/"+obsname+"_{0:02d}.txt") # ImPlaneIA-specific filenames
            if self.verbose: print("\t"+fnheads[-1])

        # load from text into data rrays:
        for slice in range(self.nslices):
            self.fp[slice:] = np.loadtxt(fnheads[0].format(slice))
            self.fa[slice:] = np.loadtxt(fnheads[1].format(slice))
            self.cp[slice:] = np.loadtxt(fnheads[2].format(slice))
            if len(self.observables) == 4:
                self.ca[slice:] = np.loadtxt(fnheads[3].format(slice))


def mainsmall(nh=None):
    " Assemble list of object observables' paths, target usually first, one or multiple calibrators"
    objectpaths = ("../example_data/noise/tgt_ov3/t_disk_small2_0__PSF_MASK_NRM_F430M_x11_0.82_ref__00/",
                   "../example_data/noise/cal_ov3/c_disk3_4__PSF_MASK_NRM_F430M_x11_0.82_ref__00/")
    observables_list = []
    for obj in objectpaths:
        observables_list.append(ObservablesFromText(nh, obj))
        # can mod to use glob above to count number of slices...

def main_ansou(nh=None, txtdir=None):
    "Reads in every observable available into a list of Observables"
    observables_list = []
    observables_list.append(ObservablesFromText(nh, txtdir))


if __name__ == "__main__":
    #mainsmall(nh=7)
    # one sky object's ImPlaneIA text output directory
    objecttextdir="../example_data/example_anthonysoulain/cal_ov3/c_myscene_disk_r=100mas__F430M_81_flat_x11__00_mir"
    main_ansou(nh=7, txtdir=objecttextdir)
