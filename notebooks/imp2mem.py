#! /usr/bin/env python

"""
Reads ImPlaneIA output text of [CPs,phases,apmlitudes]_nn.txt in one directory
Writes them out into oifits files in the same directory.

Chunkwide averaging might be a later development for real data.

anand@stsci.edu started 2019 09 26 

"""

import os, sys, time, glob
import pickle
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
                           oifinfofn='info4oif_dict.pkl',
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
            observables: If ("phases", "amplitudes", "CPs", "CAs") for example - ImPlaneIA nomenclature
                       then the files need to be named: "/phases_{0:02d}.txt".format(slc)
                       Three or four quantities (omitting CA is optional)
                       Order is relevant... pha, amp, cp [, ca]
            oifinfofn: default 'info4oif_dict.pkl' suits ImPlaneIA
                      Pickle file of info oifits writers might need, in a dictionary
                      Located in the same dir as text observable files.  Only one for all slices...
            If you want to trim low and/or high ends of eg IFU spectral observables trim them
            on-the-fly before calling this routine.  Or write a trimming method for this module.

            ImPlaneIA saves fp cp in RADIANS.
        """

        print( "=== ObservablesFromText ===\nObject text observables' directorypath:\n    ", txtpath)
        self.txtpath = txtpath
        self.oifpath = oifpath
        self.verbose = verbose
        self.observables = observables
        self.oifinfofn = oifinfofn
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
    

    def _makequads_all(self):
        """ returns int array of quad hole indices (0-based), 
            and float array of three uvw vectors in all quads
        """
        nholes=self.ctrs.shape[0]
        qlist = []
        for i in range(nholes):
            for j in range(nholes):
                for k in range(nholes):
                    for q in range(nholes):
                        if i<j and j<k and k<q:
                            qlist.append((i,j,k,q))
        qarray = np.array(qlist).astype(np.int)
        if self.verbose: print("qarray", qarray.shape, "\n", qarray)
        qname = []
        uvwlist = []
        # foreach row of 3 elts...
        for quad in qarray:
            qname.append("{0:d}_{1:d}_{2:d}_{3:d}".format(quad[0], quad[1], quad[2], quad[3]))
            if self.verbose: print('quad:', quad, qname[-1])
            uvwlist.append((self.ctrs[quad[0]] - self.ctrs[quad[1]],
                            self.ctrs[quad[1]] - self.ctrs[quad[2]],
                            self.ctrs[quad[2]] - self.ctrs[quad[3]]))
        if self.verbose: print( qarray.shape, np.array(uvwlist).shape)
        return qarray, np.array(uvwlist)

    def _maketriples_all(self):
        """ returns int array of triple hole indices (0-based), 
            and float array of two uv vectors in all triangles
        """
        nholes=self.ctrs.shape[0]
        tlist = []
        for i in range(nholes):
            for j in range(nholes):
                for k in range(nholes):
                    if i<j and j<k:
                         tlist.append((i,j,k))
        tarray = np.array(tlist).astype(np.int)
        if self.verbose: print("tarray", tarray.shape, "\n", tarray)
        
        tname = []
        uvlist = []
        # foreach row of 3 elts...
        for triple in tarray:
            tname.append("{0:d}_{1:d}_{2:d}".format(triple[0], triple[1], triple[2]))
            if self.verbose: print('triple:', triple, tname[-1])
            uvlist.append( (self.ctrs[triple[0]] - self.ctrs[triple[1]],
                            self.ctrs[triple[1]] - self.ctrs[triple[2]]) )
        #print(len(uvlist), "uvlist", uvlist) 
        if self.verbose: print( tarray.shape, np.array(uvlist).shape)
        return tarray, np.array(uvlist)


    def _makebaselines(self):
        """
        ctrs (nh,2) in m
        returns np arrays of eg 21 baselinenames ('0_1',...), eg (21,2) baselinevectors (2-floats)
        in the same numbering as implaneia
        """
        nholes=self.ctrs.shape[0]
        blist = []
        for i in range(nholes):
            for j in range(nholes):
                if i<j:
                    blist.append((i,j))
        barray = np.array(blist).astype(np.int)
        blname = []
        bllist = []
        for basepair in blist:
            #blname.append("{0:d}_{1:d}".format(basepair[0],basepair[1]))
            baseline = self.ctrs[basepair[0]] - self.ctrs[basepair[1]]
            bllist.append(baseline)
        return barray, np.array(bllist)


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
        #print(self.info4oif_dict)

        print("hole centers array shape:", self.ctrs.shape)

        print(len(self.bholes), "baseline hole indices\n", self.bholes)
        print(self.bls.shape, "baselines:\n", self.bls)

        print(self.tholes.shape, "triple hole indices:\n", self.tholes)
        print(self.tuv.shape, "triple uv vectors:\n", self.tuv )

        print(self.qholes.shape, "quad hole indices:\n", self.qholes)
        print(self.quvw.shape, "quad uvw vectors:\n", self.quvw)

    

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

        """ get this data in...
        pfn = self.savedir+self.sub_dir_str+"/info4oif_dict.pkl"
        pfd = open(pfn,'wb')
        pickle.dump(info4oif_fn,pfd)
        pfd.close()
        """
        # read in pickle of the info oifits might need...
        pfd = open(self.txtpath+'/'+self.oifinfofn,'rb')
        self.info4oif_dict = pickle.load(pfd)
        pfd.close()
        self.ctrs = self.info4oif_dict['ctrs']
        self.bholes, self.bls = self._makebaselines()
        self.tholes, self.tuv = self._maketriples_all()
        self.qholes, self.quvw = self._makequads_all()

def mainsmall(nh=None):
    " Assemble list of object observables' paths, target usually first, one or multiple calibrators"
    objectpaths = ("../example_data/noise/tgt_ov3/t_disk_small2_0__PSF_MASK_NRM_F430M_x11_0.82_ref__00/",
                   "../example_data/noise/cal_ov3/c_disk3_4__PSF_MASK_NRM_F430M_x11_0.82_ref__00/")
    observables_list = []
    for obj in objectpaths:
        observables_list.append(ObservablesFromText(nh, obj))
        # can mod to use glob above to count number of slices...

def main_ansou(nh=None, txtdir=None, verbose=True):
    "Reads in every observable available into a list of Observables"
    observables_list = []
    observables_list.append(ObservablesFromText(nh, txtdir, verbose=verbose))


if __name__ == "__main__":
    #mainsmall(nh=7)
    # one sky object's ImPlaneIA text output directory
    objecttextdir="../example_data/example_anthonysoulain/cal_ov3/c_myscene_disk_r=100mas__F430M_81_flat_x11__00_mir"
    main_ansou(nh=7, txtdir=objecttextdir, verbose=False)
