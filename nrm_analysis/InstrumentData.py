#! /usr/bin/env python

"""
InstrumentData Class -- defines data format, wavelength info, mask geometry

Instruments/masks supported:
GPI NRM
NIRISS AMI
VISIR SAM
** we like acronyms **


"""

# Standard Imports
import numpy as np
from astropy.io import fits
import os, sys, time

# Module imports
# mask geometries, GPI, NIRISS, VISIR supported...
from .misctools.mask_definitions import NRM_mask_definitions 
from .misctools import utils
from .misctools import lpl_ianc

um = 1.0e-6

# utility routines for InstrumentData classes

def show_cvsupport_threshold(instr):
    """ Show threshold for where 'splodge' data in CV space contains signal """
    print("InstrumentData: ", "cvsupport_threshold is: ", instr.cvsupport_threshold)
    print("InstrumentData: ", instr.cvsupport_threshold)

def set_cvsupport_threshold(instr, k, v):
    """ Set threshold for where 'splodge' data in CV space contains signal
        
    Parameters
    ----------
    instr: InstrumentData instance
    thresh: Threshold for the absolute value of the FT(interferogram).
            Normalize abs(CV = FT(a)) for unity peak, and define the support 
            of "good" CV when this is above threshold
    """
    
    instr.cvsupport_threshold[k] = v
    print("InstrumentData: ", "New cvsupport_threshold is: ", instr.cvsupport_threshold)


class GPI:

    def __init__(self, reffile, **kwargs):
        """
        Initialize GPI class

        ARGUMENTS:

        reffile - one or a list of reference fits files 
                  gpi-pipeline reduced containing useful header info
        
        optionally: 'gpifilterpath'
                - Point to a directory which contains GPI filter files
                  code will read in the relevant file and pick out a 
                  sample of wavelengths and transmissions that span the
                  list, to be used later to generate the model.
        """

        # only one NRM on GPI:
        self.arrname = "gpi_g10s40"
        self.pscale_mas = 14.1667 #14.27 looks like a better match March 2019
        self.pscale_rad = utils.mas2rad(self.pscale_mas)
        self.mask = NRM_mask_definitions(maskname=self.arrname)
        self.mask.ctrs = np.array(self.mask.ctrs)
        # Hard code -1.5 deg rotation in data (April 2016)
        # (can be moved to NRM_mask_definitions later)
        self.mask.ctrs = utils.rotate2dccw(self.mask.ctrs, (-3.7)*np.pi/180.)
        # Add in hole/baseline properties ?
        self.holeshape="circ"
        affine2d = kwargs.get( 'affine2d', None)
        if affine2d is None:
            self.affine2d = utils.Affine2d(mx=1.0,my=1.0, 
                                           sx=0.0,sy=0.0, 
                                           xo=0.0,yo=0.0, name="Ideal")
        else:
            self.affine2d = affine2d

        # Get info from reference file 
        self.hdr0 = []
        self.hdr1 = []
        self.refdata = []
        if type(reffile)==str:
            reffile = [reffile,]
        for fn in reffile:
            reffits = fits.open(fn)
            self.hdr0.append(reffits[0].header)
            self.hdr1.append(reffits[1].header)
            self.refdata.append(reffits[1].data)
            reffits.close()
        # instrument settings
        self.mode = self.hdr0[0]["DISPERSR"]
        self.obsmode = self.hdr0[0]["OBSMODE"]
        self.band = self.obsmode[-1] # K1 is two letters
        self.ref_imgs_dir = "refimgs_"+self.band+"/"

        # finding centroid from phase slope only considered cv_phase data 
        # when cv_abs data exceeds this cvsupport_threshold.  
        # Absolute value of cv data normalized to unity maximum
        # for the threshold application.
        # Data reduction gurus: tweak the threshold value with experience...
        self.cvsupport_threshold = {"Y":0.02, "J":0.02, "H":0.02, "1":0.02, "2":0.02}
        self.threshold = self.cvsupport_threshold[self.band]


        # Special mode for collapsed data
        if self.hdr1[0]["NAXIS3"]==1:
            # This is just a way handle data that is manually collapsed.
            # Not a standard data format for GPI.
            if self.verbose: print("InstrumentData: ", "No NAXIS3 keyword. This is probably collapsed data.")
            if self.verbose: print("InstrumentData: ", "Going to fake some stuff now")
            self.mode = "WOLLASTON_FAKEOUT"

        # wavelength info: spect mode or pol more
        if "PRISM" in self.mode:
            # GPI's spectral mode
            self.nwav = self.hdr1[0]["NAXIS3"]
            self.wls = np.linspace(self.hdr1[0]["CRVAL3"], \
                                   self.hdr1[0]["CRVAL3"]+\
                                   self.hdr1[0]['CD3_3']*self.nwav,\
                                   self.nwav)*um
            self.eff_band = um*np.ones(self.nwav)*(self.wls[-1] - \
                                       self.wls[0])/self.nwav
        elif "WOLLASTON" in self.mode:
            # GPI's pol mode. Will define this for the DIFFERENTIAL VISIBILITIES
            # diff vis: two channels 0/45 and 22/67

            self.nwav = 2

            # Define the bands in case we use a tophat filter
            band_ctrs = {"Y":(1.14+0.95)*um/2., "J":(1.35+1.12)*um/2., \
                         "H":(1.80+1.50)*um/2., "1":(2.19+1.9)*um/2., \
                         "2":(2.4+2.13)*um/2.0}
            band_wdth = {"Y":(1.14-0.95)*um, "J":(1.35-1.12)*um, "H":(1.80-1.50)*um, \
                         "1":(2.19-1.9)*um, "2":(2.4-2.13)*um}
            wghts = np.ones(15)
            wavls = np.linspace(band_ctrs[self.band]-band_wdth[self.band]/2.0, \
                                band_ctrs[self.band]+band_wdth[self.band]/2.0, num=15)

            if 'gpifilterpath' in kwargs:
                if self.verbose: print("InstrumentData: ", "Using GPI filter file ", end='')
                if self.band=="Y":
                    filterfile = kwargs["gpifilterpath"]+"GPI-filter-Y.fits"
                    if self.verbose: print("InstrumentData: ", kwargs["gpifilterpath"]+"GPI-filter-Y.fits")
                    cutoff=0.7
                if self.band=="J":
                    filterfile = kwargs["gpifilterpath"]+"GPI-filter-J.fits"
                    if self.verbose: print("InstrumentData: ", kwargs["gpifilterpath"]+"GPI-filter-J.fits")
                    cutoff=0.7
                if self.band=="H":
                    filterfile = kwargs["gpifilterpath"]+"GPI-filter-H.fits"
                    if self.verbose: print("InstrumentData: ", kwargs["gpifilterpath"]+"GPI-filter-H.fits")
                    cutoff=0.7
                if self.band=="1":
                    filterfile = kwargs["gpifilterpath"]+"GPI-filter-K1.fits"
                    if self.verbose: print("InstrumentData: ", kwargs["gpifilterpath"]+"GPI-filter-K1.fits")
                    cutoff=0.94
                if self.band=="2":
                    filterfile = kwargs["gpifilterpath"]+"GPI-filter-K2.fits"
                    if self.verbose: print("InstrumentData: ", kwargs["gpifilterpath"]+"GPI-filter-K2.fits")
                    cutoff=0.94 
                # Read in gpi filter file
                fitsfilter = fits.open(filterfile)[1].data
                wavls = []
                wghts = []
                # Sample the filter file so the filter is only 50 elements long
                skip = len(fitsfilter[0][0]) / 50
                for ii in range(len(fitsfilter[0][0])/skip):
                    if fitsfilter[0][1][skip*ii]>cutoff:
                        wavls.append(fitsfilter[0][0][skip*ii]*1.0e-6)
                        wghts.append(fitsfilter[0][1][skip*ii])

            lam_c = band_ctrs[self.band]
            lam_w = band_wdth[self.band]

            transmission = np.array([[wghts[f], wavls[f]] for f in range(len(wghts))])
            if "FAKEOUT" in self.mode:
                self.nwav=1
                self.wls = [transmission, ]
                self.eff_band = np.array([lam_w, ])
            else:
                self.wls = [transmission, transmission]
                self.eff_band = np.array([lam_w, lam_w])
        else:
            sys.exit("Check your reference file header. "+\
                    "Keywork DISPERSR='{0}' not understood".format(self.mode))

        # For OIFits structure
        self.wavextension = (self.wls, self.eff_band)
        if "FAKEOUT" in self.mode:
            self.wavextension = ([lam_c,], [lam_w,])  

        # Observation info
        self.telname= "GEMINI"
        self.ra, self.dec = self.hdr0[0]["RA"], self.hdr0[0]["DEC"]
        try:
            self.date = self.hdr0[0]["DATE"]
        except:
            self.date = self.hdr0[0]["DATE-OBS"]
        self.month = self.date[-5:-3]
        self.day = self.date[-2:]
        self.year = self.date[:4]
        #self.parang = self.hdr0["PAR_ANG"]
        # AVPARANG added Aug 2 2016
        self.parangs = []
        self.itime = []
        self.crpa = []
        for ii in range(len(reffile)):
            self.parangs.append(self.hdr1[ii]["AVPARANG"])
            self.itime.append(self.hdr1[ii]["ITIME"])
            if "CRPA" in self.hdr0[ii]:
                self.crpa.append(self.hdr0[ii]["CRPA"])
        self.avparang = np.mean(self.parangs)
        if len(self.crpa)>0:
            self.avcassang = np.mean(self.crpa)
        else:
            self.avcassang = 0.0
        self.parang_range = abs(self.parangs[-1] - self.parangs[0])
        self.totalinttime = np.sum(self.itime)
        try:
            self.pa = self.hdr0[0]["PA"]
        except:
            self.pa = 0.0
        try:
            self.objname = self.hdr0[0]["OBJECT"]
        except:
            self.objname = "Unknown"

        # Look for additional keyword arguments ?
    def read_data(self, fn):

        fitsfile = fits.open(fn)
        sci=fitsfile[1].data
        hdr=fitsfile[1].header
        fitsfile.close()
        #fitshdr = fitsfile[0].header
        if 'distorcorr' in fn:
            self.sub_dir_str = fn[-32:-21]
        else:
            self.sub_dir_str = fn[-21:-10]

        return sci, hdr

class VISIR:
    def __init__(self, objname="obj", band="11.3", src = "A0V",
                 affine2d=None):
        """
        Initialize VISIR class

        ARGUMENTS:
        objname - string w/name of object observed
        src - if pysynphot is installed, can provide a guess at the stellar spectrum
        """
        self.band = band

        self.objname = objname

        self.arrname = "visir_sam"
        self.pscale_mas = 45
        self.pscale_rad = utils.mas2rad(self.pscale_mas)
        self.mask = NRM_mask_definitions(maskname=self.arrname)
        self.mask.ctrs = np.array(self.mask.ctrs)
        #self.mask.ctrs[:,1]*=-1
        self.holeshape="hex"
        #self.mask.ctrs = utils.rotate2dccw(self.mask.ctrs, 7.5*np.pi/180.)

        if affine2d is None:
            self.affine2d = utils.Affine2d(mx=1.0,my=1.0, 
                                           sx=0.0,sy=0.0, 
                                           xo=0.0,yo=0.0, name="Ideal")
        else:
            self.affine2d = affine2d

        # tophat filter
        # this can be swapped with an actual filter file
        if self.band=="11.3":
            self.lam_c = 11.3*1e-6 # 11.3 microns
            self.lam_w = 0.6/11.3 # 0.6 micron bandpass
        elif self.band=="10.5":
            self.lam_c = 10.6*1e-6 # 11.3 microns
            self.lam_w = 0.1/10.5 # 0.6 micron bandpass
        else:
            raise ValueError("options for band are '11.3' or '10.5' \n{0} not supported".format(band))
        self.filt = utils.tophatfilter(self.lam_c, self.lam_w, npoints=10)
        try:
            self.wls = [utils.combine_transmission(self.filt, src), ]
        except:
            self.wls = [self.filt, ]
        #self.wavextension = (self.lam_c, self.lam_w)
        #self.wavextension = (self.lam_c*np.ones(self.nexp), \
        #                     self.lam_w*np.ones(self.nexp))
        self.wavextension = ([self.lam_c,], [self.lam_w,])
        self.nwav=1

        # finding centroid from phase slope only considered cv_phase data when cv_abs data exceeds this.  
        # absolute value of cv data normalized to unity maximum for the threshold application.
        self.cvsupport_threshold = {"10.5":0.02, "11.3":0.02} # Gurus: tweak with use...
        self.threshold = self.cvsupport_threshold[self.band]

        self.ref_imgs_dir = "refimgs/"

        #############################
        # Observation info - I don't know yet how JWST data headers will be structured
        self.telname= "VLT"
        try:
            self.ra, self.dec = self.hdr0["RA"], self.hdr0["DEC"]
        except:
            self.ra, self.dec = 00, 00
        try:
            self.date = self.hdr0["DATE"]
            self.month = self.date[-5:-3]
            self.day = self.date[-2:]
            self.year = self.date[:4]
        except:
            lt = time.localtime()
            self.date = "{0}{1:02d}{2:02d}".format(lt[0],lt[1],lt[2])
            self.month = lt[1]
            self.day = lt[2]
            self.year = lt[0]
        try:
            self.parang = self.hdr0["PAR_ANG"]
        except:
            self.parang = 00
        try:
            self.pa = self.hdr0["PA"]
        except:
            self.pa = 00
        try:
            self.itime = self.hdr1["ITIME"]
        except:
            self.itime = 00
        #############################

    def read_data(self, fn):
        # for datacube of exposures, need to read as 3D (nexp, npix, npix)
        fitsfile = fits.open(fn)
        scidata=fitsfile[0].data
        hdr=fitsfile[0].header
        #self.sub_dir_str = self.filt+"_"+objname
        self.sub_dir_str = '/' + fn.split('/')[-1].replace('.fits', '')
        #self.nexp = scidata.shape[0]
        # rewrite wavextension to be same length as nexp
        if len(scidata.shape)==3:
            self.nwav=scidata.shape[0]
            [self.wls.append(self.wls[0]) for f in range(self.nwav-1)]
            return scidata, hdr
        elif len(scidata.shape)==2: # 'cast' 2d array to 3d with shape[0]=1
            return np.array([scidata,]), hdr
        else:
            sys.exit("invalid data dimensions for VISIR. Should have dimensionality of 2 or 3.")
        return scidata, hdr

class NIRISS:
    def __init__(self, filt, 
                       objname="obj", 
                       src="A0V", 
                       out_dir='', 
                       chooseholes=None, 
                       affine2d=None, 
                       bandpass=None,
                       nbadpix=4,
                       usebp = True,
                       firstfew = None,
                       **kwargs):
        """
        Initialize NIRISS class

        ARGUMENTS:

        kwargs:
        UTR
        Or just look at the file structure
        Either user has webbpsf and filter file can be read, or this will use a tophat and give a warning
        chooseholes: None, or e.g. ['B2', 'B4', 'B5', 'B6'] for a four-hole mask

        bandpass: None or [(wt,wlen),(wt,wlen),...].  Monochromatic would be e.g. [(1.0, 4.3e-6)]
                  Explicit bandpass arg will replace *all* niriss filter-specific variables with 
                  the given bandpass, so you can simulate 21cm psfs through something called "F430M"!
        firstfew: None or the number of slices to truncate input cube to in memory,
                  the latter for fast developmpent
        nbadpix:  Number of good pixels to use when fixing bad pixels
        usebp:    True (default) use DQ array from input MAST data.  False: do not utilize DQ array.
        """
        
        if "verbose" in kwargs:
            self.verbose=kwargs["verbose"]
        else:
            self.verbose=False

        # self.bpexist set True/False if  DQ fits image extension exists/doesn't
        self.nbadpix = 4

        if chooseholes:
            print("InstrumentData.NIRISS: ", chooseholes)
        self.chooseholes = chooseholes

        self.usebp = usebp
        print("InstrumentData.NIRISS: use bad pixel DQ extension data", usebp)

        self.firstfew = firstfew
        if firstfew is not None: print("InstrumentData.NIRISS: analysing firstfew={:d} slices".format(firstfew))

        self.objname = objname

        self.filt = filt

        #############################
        self.lam_bin = {"F277W": 50, "F380M": 20, "F430M":40,  "F480M":30} # 12 waves in f430 - data analysis
                                                                           # use 150 for 3 waves ax f430m 
        self.lam_c = {"F277W": 2.77e-6,  # central wavelength (SI)
                      "F380M": 3.8e-6, 
                      "F430M": 4.28521033106325E-06,
                      "F480M": 4.8e-6}
        self.lam_w = {"F277W":0.2, "F380M": 0.1, "F430M": 0.0436, "F480M": 0.08} # fractional filter width 
        
        try:
            self.throughput = utils.trim_webbpsf_filter(self.filt, specbin=self.lam_bin[self.filt])
        except:
            self.throughput = utils.tophatfilter(self.lam_c[self.filt], self.lam_w[self.filt], npoints=11)
            print("InstrumentData.NIRISS: ", "*** WARNING *** InstrumentData: Top Hat filter being used")

        # Nominal
        """self.lam_c = {"F277W": 2.77e-6,  # central wavelength (SI)
                      "F380M": 3.8e-6, 
                      "F430M": 4.28521033106325E-06,
                      "F480M": 4.8e-6}
        self.lam_w = {"F277W":0.2, "F380M": 0.1, "F430M": 0.0436, "F480M": 0.08} # fractional filter width """

        # update nominal filter parameters with those of the filter read in and used in the analysis...
        # Weighted mean wavelength in meters, etc, etc "central wavelength" for the filter:
        from scipy.integrate import simps 
        self.lam_c[self.filt] = (self.throughput[:,1] * self.throughput[:,0]).sum() / self.throughput[:,0].sum() 
        area = simps(self.throughput[:,0], self.throughput[:,1])
        ew = area / self.throughput[:,0].max() # equivalent width
        beta = ew/self.lam_c[self.filt] # fractional bandpass
        self.lam_w[self.filt] = beta

        if bandpass is not None:
            bandpass = np.array(bandpass)  # type simplification
            wt = bandpass[:,0]
            wl = bandpass[:,1]
            print(" which is now  overwritten with user-specified bandpass:\n", bandpass)
            cw = (wl*wt).sum()/wt.sum() # Weighted mean wavelength in meters "central wavelength"
            area = simps(wt, wl)
            ew = area / wt.max() # equivalent width
            beta = ew/cw # fractional bandpass
            self.lam_c = {"F277W":cw, "F380M": cw, "F430M": cw, "F480M": cw,}
            self.lam_w = {"F277W": beta, "F380M": beta, "F430M": beta, "F480M": beta} 
            self.throughput = bandpass
        if self.verbose: print("InstrumentData.NIRISS: ", self.filt, 
              ": central wavelength {:.4e} microns, ".format(self.lam_c[self.filt]/um), end="")
        if self.verbose: print("InstrumentData.NIRISS: ", "fractional bandpass {:.3f}".format(self.lam_w[self.filt]))

        try:
            self.wls = [utils.combine_transmission(self.throughput, src), ]
        except:
            self.wls = [self.throughput, ]
        if self.verbose: print("self.throughput:\n", self.throughput)

        # Wavelength info for NIRISS bands F277W, F380M, F430M, or F480M
        self.wavextension = ([self.lam_c[self.filt],], [self.lam_w[self.filt],])
        self.nwav=1
        #############################

        # only one NRM on JWST:
        self.telname = "JWST"
        self.instrument = "NIRISS"
        self.arrname = "jwst_g7s6c"  # implaneia mask set with this - unify to short form later 
        self.holeshape="hex"
        self.mask = NRM_mask_definitions(maskname=self.arrname, chooseholes=chooseholes, 
                                         holeshape=self.holeshape )

        # save affine deformation of pupil object or create a no-deformation object. 
        # We apply this when sampling the PSF, not to the pupil geometry.
        # This will set a default Ideal or a measured rotation, for example,
        # and include pixel scale changes due to pupil distortion.
        # Separating detector tilt pixel scale effects from pupil distortion effects is 
        # yet to be determined... see comments in Affine class definition.
        # AS AZG 2018 08 15 Ann Arbor
        if affine2d is None:
            self.affine2d = utils.Affine2d(mx=1.0,my=1.0, 
                                           sx=0.0,sy=0.0, 
                                           xo=0.0,yo=0.0, name="Ideal")
        else:
            self.affine2d = affine2d

        # finding centroid from phase slope only considered cv_phase data 
        # when cv_abs data exceeds this cvsupport_threshold.  
        # Absolute value of cv data normalized to unity maximum
        # for the threshold application.
        # Data reduction gurus: tweak the threshold value with experience...
        # Gurus: tweak cvsupport with use...
        self.cvsupport_threshold = {"F277W":0.02, "F380M": 0.02, "F430M": 0.02, "F480M": 0.02}
        if self.verbose: show_cvsupport_threshold(self)
        self.threshold = self.cvsupport_threshold[filt]

        self.ref_imgs_dir = os.path.join(out_dir,"refimgs_"+self.filt+"/")


    def set_pscale(self, pscalex_deg=None, pscaley_deg=None):
        """
        Override pixel scale in header
        """
        if pscalex_deg is not None:
            self.pscalex_deg = pscalex_deg
        if pscaley_deg is not None:
            self.pscaley_deg = pscaley_deg
        self.pscale_mas = 0.5 * (pscalex_deg +  pscaley_deg) * (60*60*1000)
        self.pscale_rad = utils.mas2rad(self.pscale_mas)

    def read_data(self, fn, mode="slice"):
        # mode options are slice or UTR
        # for single slice data, need to read as 3D (1, npix, npix)
        # for utr data, need to read as 3D (ngroup, npix, npix)
        # fix bad pixels using DQ extension and LPL local averaging, 
        # but send bad pixel array down to where fringes are fit so they can be ignored.
        fitsfile = fits.open(fn)
        scidata=fitsfile[1].data
        
        # usually DQ in MAST file... make it non-fatal DQ missing
        try:
            bpdata=fitsfile['DQ'].data # bad pixel extension
            self.bpexist = True
            # If calling driver wants to skip using DQ info even if DQ array in data,
            # force the bpexist flag to False here.
            if self.usebp == False: 
                self.bpexist = False
                print('InstrumentData.NIRISS.read_data: skipping use of DQ array')
        except:
            print("InstrumentData.NIRISS: no DQ exension found in", fn)
            print("InstrumentData.NIRISS: assuming all pixels are good.", fn)
            self.bpexist = False

        if scidata.ndim == 3:  #len(scidata.shape)==3:
            print("input: 3D cube")
            # truncate all but the first few slices for rapid development
            if self.firstfew is not None:
                if scidata.shape[0] > self.firstfew:  
                    scidata = scidata[:self.firstfew, :, :]
                    bpdata = bpdata[:self.firstfew, :, :]
            self.nwav=scidata.shape[0]
            [self.wls.append(self.wls[0]) for f in range(self.nwav-1)]
        elif len(scidata.shape)==2: # 'cast' 2d array to 3d with shape[0]=1
            print("input: 2D data array convering to 3D one-slice cube")
            scidata = np.array([scidata,])
        else:
            sys.exit("invalid data dimensions for NIRISS. Should have dimensionality of 2 or 3.")

        # refpix removal by trimming
        scidata = scidata[:,4:, :] # [all slices, imaxis[0], imaxis[1]]
        print('Refpix-trimmed off scidata', scidata.shape)

        # fix pix using bad pixel map - runs now.  Need to sanity-check.
        if self.bpexist:
            # refpix removal by trimming to match image trim
            bpdata = bpdata[:,4:, :]     # refpix removal by trimming to match image trim
            bpd = np.zeros(bpdata.shape, np.uint8) # zero or 1 bad pix marker array
            bpd[bpdata!=0] = 1
            bpdata = bpd.copy()

            print('Refpix-trimmed off bpdata ', bpdata.shape)

            #
            print('InstrumentData.NIRISS.read_data: debugging output hardcoded first three slices - delete after checking')
            for slc in range(bpdata.shape[0]):  # all input slices

                if slc < 0:  # debug before fixing bps
                    scia = scidata[slc,:,:]
                    fits.PrimaryHDU(data=scidata[slc,:,:]).writeto(
                        f'/Users/anand/data/implaneia/NAP019data/bptest/prebpfix_{slc:d}.fits', overwrite=True)

                # fix the pixel values using self.nbadpix neighboring values
                scidata[slc,:,:] = lpl_ianc.bfixpix(scidata[slc,:,:], bpdata[slc,:,:], self.nbadpix)

                if slc < 0:  # debug after fixing bps
                    scib = scidata[slc,:,:]
                    fits.PrimaryHDU(data=scib).writeto(
                        f'/Users/anand/data/implaneia/NAP019data/bptest/postbpfix_{slc:d}.fits', overwrite=True)
                    fits.PrimaryHDU(data=bpd[slc,:,:]).writeto(
                        f'/Users/anand/data/implaneia/NAP019data/bptest/bpd_{slc:d}.fits', overwrite=True)
    
        prihdr=fitsfile[0].header
        scihdr=fitsfile[1].header
        # MAST header or similar kwds info for oifits writer:
        self.updatewithheaderinfo(prihdr, scihdr)

        # create subdirectory name into which to write txt & oifits observables' files
        # This is a subdirectory where the data was found, named for fits input root name
        self.sub_dir_str = '/' + fn.split('/')[-1].replace('.fits', '')

        return prihdr, scihdr, scidata, bpdata


    def cdmatrix_to_sky(self, vec, cd11, cd12, cd21, cd22):
        """ use the global header values explicitly, for clarity 
            vec is 2d, units of pixels
            cdij is 2x4 array in units degrees/pixel
        """
        return np.array((cd11*vec[0] + cd12*vec[1], cd21*vec[0] + cd22*vec[1]))


    def degrees_per_pixel(self, phdr):
        """
        input: phdr:  fits data file's header with or without CDELT1, CDELT2 (degrees per pixel)
        returns: cdelt1, cdelt2: tuple, degrees per pixel along axes 1, 2
                         EITHER: read from header CDELT[12] keywords 
                             OR: calculated using CD matrix (Jacobian of RA-TAN, DEC-TAN degrees
                                 to pixel directions 1,2.  No deformation included in this routine,
                                 but the CD matric includes non-linear field distortion.
                                 No pupil distortion or rotation here.
                        MISSING: If keywords are missing default hardcoded cdelts are returned.
                                 The exact algorithm may substitute this later.
                                 Below seems good to ~5th significant figure when compared to 
                                 cdelts header values prior to replacement by cd matrix approach.
            N.D. at stsci 11 Mar 20212

            We start in Level 1 with the PC matrix and CDELT.
            CDELTs come from the SIAF.

            The PC matrix is computed from the roll angle, V3YANG and the parity.
            The code is here
            https://github.com/spacetelescope/jwst/blob/master/jwst/assign_wcs/util.py#L153

            In the level 2 imaging pipeline, assign_wcs adds the distortion to the files.
            At the end it computes an approximation of the entire distortion transformation
            by fitting a polynomial. This approximated distortion is represented as SIP
            polynomials in the FITS headers.

            Because SIP, by definition, uses a CD matrix, the PC + CDELT are replaced by CD.

            How to get CDELTs back?

            I think once the rotation, skew and scale are in the CD matrix it's very hard to
            disentangle them. The best way IMO is to calculate the local scale using three
            point difference. There is a function in jwst that does this.

            Using a NIRISS image as an example:

            from jwst.assign_wcs import util
            from jwst import datamodels

            im=datamodels.open('niriss_image_assign_wcs.fits')

            util.compute_scale(im.meta.wcs, (im.meta.wcsinfo.ra_ref, im.meta.wcsinfo.dec_ref))

            1.823336635353374e-05

            The function returns a constant scale. Is this sufficient for what you need or
            do you need scales and sheer along each axis? The code in util.compute_scale can
            help with figuring out how to get scales along each axis.

            I hope this answers your question.
        """

        if 'CD1_1' in phdr.keys() and 'CD1_2' in phdr.keys() and  \
             'CD2_1' in phdr.keys() and 'CD2_2' in phdr.keys():
            cd11 = phdr['CD1_1']
            cd12 = phdr['CD1_2']
            cd21 = phdr['CD2_1']
            cd22 = phdr['CD2_2']
            # Create unit vectors in detector pixel X and Y directions, units: detector pixels
            dxpix  =  np.array((1.0, 0.0)) # axis 1 step
            dypix  =  np.array((0.0, 1.0)) # axis 2 step
            # transform pixel x and y steps to RA-tan, Dec-tan degrees
            dxsky = self.cdmatrix_to_sky(dxpix, cd11, cd12, cd21, cd22)
            dysky = self.cdmatrix_to_sky(dypix, cd11, cd12, cd21, cd22)
            print("Used CD matrix for pixel scales")
            return np.linalg.norm(dxsky, ord=2), np.linalg.norm(dysky, ord=2)
        elif 'CDELT1' in phdr.keys() and 'CDELT2' in phdr.keys():
            return phdr['CDELT1'], phdr['CDELT2']
            print("Used CDDELT[12] for pixel scales")
        else:
            print('InstrumentData.NIRISS: Warning: NIRISS pixel scales not in header.  Using 65.6 mas in deg/pix')
            return 65.6/(60.0*60.0*1000), 65.6/(60.0*60.0*1000)

    
    def updatewithheaderinfo(self, ph, sh):
        """ input: primary header, science header MAST"""

        # The info4oif_dict will get pickled to disk when we write txt files of results.
        # That way we don't drag in objects like InstrumentData into code that reads text results
        # and writes oifits files - a simple built-in dictionary is the only object used in this transfer.
        info4oif_dict = {}
        info4oif_dict['telname'] = self.telname

        info4oif_dict['filt'] = self.filt
        info4oif_dict['lam_c'] = self.lam_c
        info4oif_dict['lam_w'] = self.lam_w
        info4oif_dict['lam_bin'] = self.lam_bin

        # Target information
        self.objname =  ph["TARGNAME"]
        # 'ABDor' or 'AB Dor' work. Do not use '-' in aptx proposal name
        # self.objname =  "ABDor"; info4oif_dict['objname'] = self.objname 
        self.objname = self.objname.replace('-',''); info4oif_dict['objname'] = self.objname
        self.ra = ph["TARG_RA"]; info4oif_dict['ra'] = self.ra
        self.dec = ph["TARG_DEC"]; info4oif_dict['dec'] = self.dec

        # / axis 1 DS9 coordinate of the reference pixel (always POS1)
        # / axis 2 DS9 coordinate of the reference pixel (always POS1)
        self.crpix1 = sh["CRPIX1"]; info4oif_dict['crpix1'] = self.crpix1
        self.crpix2 = sh["CRPIX2"]; info4oif_dict['crpix2'] = self.crpix2
        # need Paul Goudfrooij's table for actual crval[1,2] for true pointing to detector pixel coords (DS9)

        self.instrument = ph["INSTRUME"]; info4oif_dict['instrument'] = self.instrument
        self.pupil =  ph["PUPIL"]; info4oif_dict['pupil'] = self.pupil
        # "ImPlaneIA internal mask name" - oifwriter looks for 'mask'...
        self.arrname = "jwst_g7s6c"  # implaneia internal name - historical
        info4oif_dict['arrname'] = 'g7s6' # for oif
        info4oif_dict['mask'] = info4oif_dict['arrname']  # Soulain mask goes into oif arrname

        # if data was generated on the average pixel scale of the header
        # then this is the right value that gets read in, and used in fringe fitting
        pscalex_deg, pscaley_deg = self.degrees_per_pixel(sh)
        #
        info4oif_dict['pscalex_deg'] = pscalex_deg
        info4oif_dict['pscaley_deg'] = pscaley_deg
        # Whatever we did set is averaged for isotropic pixel scale here
        self.pscale_mas = 0.5 * (pscalex_deg + pscaley_deg) * (60*60*1000); \
        info4oif_dict['pscale_mas'] = self.pscale_mas
        self.pscale_rad = utils.mas2rad(self.pscale_mas); info4oif_dict['pscale_rad'] = self.pscale_rad

        self.mask = NRM_mask_definitions(maskname=self.arrname, chooseholes=self.chooseholes,
                                         holeshape=self.holeshape) # for STAtions x y in oifs

        self.date = ph["DATE-OBS"] + "T" + ph["TIME-OBS"]; info4oif_dict['date'] = self.date
        datestr = ph["DATE-OBS"]
        self.year = datestr[:4]; info4oif_dict['year'] = self.year
        self.month = datestr[5:7]; info4oif_dict['month'] = self.month
        self.day = datestr[8:10]; info4oif_dict['day'] = self.day
        self.parangh= sh["ROLL_REF"]; info4oif_dict['parangh'] = self.parangh
        self.pa = sh["PA_V3"]; info4oif_dict['pa'] = self.pa
        self.vparity = sh["VPARITY"]; info4oif_dict['vparity'] = self.vparity

        # An INTegration is NGROUPS "frames", not relevant here but context info.
        # 2d => "cal" file combines all INTegrations (ramps)
        # 3d=> "calints" file is a cube of all INTegrations (ramps)
        if sh["NAXIS"] == 2:
            # all INTegrations or 'ramps'
            self.itime = ph["EFFINTTM"] * ph["NINTS"]; info4oif_dict['itime'] = self.itime
        elif sh["NAXIS"] == 3:
            # each slice is one INTegration or 'ramp'
            self.itime = ph["EFFINTTM"]; info4oif_dict['itime'] = self.itime


        np.set_printoptions(precision=5, suppress=True, linewidth=160, 
                            formatter={'float': lambda x: "%10.5f," % x})
        self.v3i_yang = sh['V3I_YANG']  # Angle from V3 axis to Ideal y axis (deg)
        #print("self.mask.ctrs: \n", self.mask.ctrs)
        # rotate mask hole center coords by PAV3 # RAC 2021
        ctrs_sky = self.mast2sky()
        #print("new ctrs: \n", ctrs_sky)
        # Sydney oif coords switch... slow but sure this time!  (I hope...)
        stax_oif = ctrs_sky[:,1].copy() * -1
        stay_oif = ctrs_sky[:,0].copy() * -1
        # print("stax_oif: ", stax_oif)
        # print("stay_oif: ", stay_oif)
        oifctrs = np.zeros(self.mask.ctrs.shape)
        oifctrs[:,0] = stax_oif
        oifctrs[:,1] = stay_oif
        info4oif_dict['ctrs_eqt'] = oifctrs # mask centers rotated by PAV3 (equatorial coords)
        info4oif_dict['ctrs_inst'] = self.mask.ctrs # as-built instrument mask centers
        # print("info4oif ctrs : \n", info4oif_dict['ctrs'])
        info4oif_dict['hdia'] = self.mask.hdia
        info4oif_dict['nslices'] = self.nwav # nwav: number of image slices or IFU cube slices - AMI is imager
        self.info4oif_dict = info4oif_dict # save it when writing extracted observables txt


    # rather than calling InstrumentData in the niriss example just to reset just call this routine
    def reset_nwav(self, nwav):
        print("InstrumentData.NIRISS: ", "Resetting InstrumentData instantiation's nwave to", nwav)
        self.nwav = nwav

    # Delete if this is not8 called from anywhere!
    #  def _generate_filter_files():
    #      """Either from WEBBPSF, or tophat, etc. A set of filter files will also be provided"""
    #      return None

    def jwst_dqflags(self):
        """ 
            dqdata is a 2d (32-bit U?)INT array from the DQ extension of the input file.
            We ignore all data with a non-zero DQ flag.  I copied all values from a 7.5 build jwst...
            but we ignore any non-zero flag meaning, and ignore the pixel in fringe-fitting
            The refpix are non-zero DQ, btw...
            I changed "pixel" to self.pbval and "group" to self.bpgroup. We may use these later, 
            so here they are but initially we just discriminate between good (zero value) and non-good.
        """

        """ JWST Data Quality Flags
            The definitions are documented in the JWST RTD:
            https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#data-quality-flags 
        """
        """ JWST Data Quality Flags
        The definitions are documented in the JWST RTD:
        https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#data-quality-flags
        Implementation
        -------------
        The flags are implemented as "bit flags": Each flag is assigned a bit position
        in a byte, or multi-byte word, of memory. If that bit is set, the flag assigned
        to that bit is interpreted as being set or active.
        The data structure that stores bit flags is just the standard Python `int`,
        which provides 32 bits. Bits of an integer are most easily referred to using
        the formula `2**bit_number` where `bit_number` is the 0-index bit of interest.
        """

        # Pixel-specific flags
        self.bpval = {
                 'GOOD':             0,      # No bits set, all is good
                 'DO_NOT_USE':       2**0,   # Bad pixel. Do not use.
                 'SATURATED':        2**1,   # Pixel saturated during exposure
                 'JUMP_DET':         2**2,   # Jump detected during exposure
                 'DROPOUT':          2**3,   # Data lost in transmission
                 'OUTLIER':          2**4,   # Flagged by outlier detection. Was RESERVED_1
                 'RESERVED_2':       2**5,   #
                 'RESERVED_3':       2**6,   #
                 'RESERVED_4':       2**7,   #
                 'UNRELIABLE_ERROR': 2**8,   # Uncertainty exceeds quoted error
                 'NON_SCIENCE':      2**9,   # Pixel not on science portion of detector
                 'DEAD':             2**10,  # Dead pixel
                 'HOT':              2**11,  # Hot pixel
                 'WARM':             2**12,  # Warm pixel
                 'LOW_QE':           2**13,  # Low quantum efficiency
                 'RC':               2**14,  # RC pixel
                 'TELEGRAPH':        2**15,  # Telegraph pixel
                 'NONLINEAR':        2**16,  # Pixel highly nonlinear
                 'BAD_REF_PIXEL':    2**17,  # Reference pixel cannot be used
                 'NO_FLAT_FIELD':    2**18,  # Flat field cannot be measured
                 'NO_GAIN_VALUE':    2**19,  # Gain cannot be measured
                 'NO_LIN_CORR':      2**20,  # Linearity correction not available
                 'NO_SAT_CHECK':     2**21,  # Saturation check not available
                 'UNRELIABLE_BIAS':  2**22,  # Bias variance large
                 'UNRELIABLE_DARK':  2**23,  # Dark variance large
                 'UNRELIABLE_SLOPE': 2**24,  # Slope variance large (i.e., noisy pixel)
                 'UNRELIABLE_FLAT':  2**25,  # Flat variance large
                 'OPEN':             2**26,  # Open pixel (counts move to adjacent pixels)
                 'ADJ_OPEN':         2**27,  # Adjacent to open pixel
                 'UNRELIABLE_RESET': 2**28,  # Sensitive to reset anomaly
                 'MSA_FAILED_OPEN':  2**29,  # Pixel sees light from failed-open shutter
                 'OTHER_BAD_PIXEL':  2**30,  # A catch-all flag
                 'REFERENCE_PIXEL':  2**31,  # Pixel is a reference pixel
        }

        # Group-specific flags. Once groups are combined, these flags
        # are equivalent to the pixel-specific flags.
        self.bpgroup = {
                 'GOOD':       self.bpval['GOOD'],
                 'DO_NOT_USE': self.bpval['DO_NOT_USE'],
                 'SATURATED':  self.bpval['SATURATED'],
                 'JUMP_DET':   self.bpval['JUMP_DET'],
                 'DROPOUT':    self.bpval['DROPOUT'],
        }


    def mast2sky(self):
        """
        Rotate hole center coordinates:
            Clockwise by the V3 position angle - V3I_YANG from north in degrees if VPARITY = -1
            Counterclockwise by the V3 position angle - V3I_YANG from north in degrees if VPARITY = 1
        Hole center coords are in the V2, V3 plane in meters.
        Return rotated coordinates to be put in info4oif_dict.
        implane2oifits.ObservablesFromText uses these to calculate baselines.
        """
        pa = self.pa
        mask_ctrs = self.mask.ctrs
        vpar = self.vparity # Relative sense of rotation between Ideal xy and V2V3
        v3iyang = self.v3i_yang
        rot_ang = pa - v3iyang # subject to change!
        if pa != 0.0:
            v2 = mask_ctrs[:,0]
            v3 = mask_ctrs[:,1]
            if vpar == -1:
                # clockwise rotation, usually the case for NIRISS?
                print('Rotating instrument ctrs %.3f clockwise' % rot_ang)
                v2_rot = v3*np.cos(np.deg2rad(rot_ang)) + v2*np.sin(np.deg2rad(rot_ang))
                v3_rot = -v3*np.sin(np.deg2rad(rot_ang)) + v2*np.cos(np.deg2rad(rot_ang))
            else:
                #counter-clockwise rotation
                print('Rotating instrument ctrs %.3f counterclockwise' % rot_ang)
                v2_rot = v3 * np.cos(np.deg2rad(rot_ang)) - v2 * np.sin(np.deg2rad(rot_ang))
                v3_rot = v3 * np.sin(np.deg2rad(rot_ang)) + v2 * np.cos(np.deg2rad(rot_ang))
            ctrs_rot = np.zeros(mask_ctrs.shape)
            ctrs_rot[:,0] = v2_rot
            ctrs_rot[:,1] = v3_rot
        else:
            ctrs_rot = mask_ctrs
        return ctrs_rot


class NIRC2:
    def __init__(self, reffile, **kwargs):
        """
        Initialize NIRC2 class

        ARGUMENTS:
        objname - string w/name of object observed
        src - if pysynphot is installed, can provide a guess at the stellar spectrum

        IFU simulation option set IFU = True
        """

        if "IFU" in kwargs:
            if kwargs["IFU"]==True:
                self.mode = "PRISM"
            else:
                self.mode = "BROADBAND"
        else:
            self.mode = "BROADBAND"
        if "src" in kwargs:
            src = kwargs["src"]
        else:
            pass
        affine2d = kwargs.get( 'affine2d', None )
        if affine2d is None:
            self.affine2d = utils.Affine2d(mx=1.0,my=1.0, 
                                           sx=0.0,sy=0.0, 
                                           xo=0.0,yo=0.0, name="Ideal")
        else:
            self.affine2d = affine2d

        self.arrname = "NIRC2_9NRM"
        self.pscale_mas = 9.952 # mas
        self.pscale_rad = utils.mas2rad(self.pscale_mas)
        self.mask = NRM_mask_definitions(maskname=self.arrname)
        self.mask.ctrs = np.array(self.mask.ctrs)
        # Hard code -1.5 deg rotation in data (April 2016)
        # (can be moved to NRM_mask_definitions later)
        self.mask.ctrs = utils.rotate2dccw(self.mask.ctrs, 1.0*np.pi/180.0)#, np.pi/10.)
        # Add in hole/baseline properties ?
        self.holeshape="circ"

        self.threshold = 0.02

        # Get info from reference file 
        self.hdr = []
        #self.refdata = []
        if type(reffile)==str:
            reffile = [reffile,]
        for fn in reffile:
            reffits = fits.open(fn)
            self.hdr.append(reffits[0].header)
            reffits.close()
        # instrument settings
        self.band = self.hdr[0]["FWINAME"]

        self.objname = self.hdr[0]["OBJECT"]

        # tophat filter
        # this can be swapped with an actual filter file
        #band_ctrs = {"Kp":1.633*um/2.0,"Lp":3.1*um}
        band_ctrs = {"J":1.248*um, "H":1.633*um, "CH4_short":1.5923*um,"Kp":2.2*um,"Lp":3.1*um}
        band_wdth = {"J":0.163*um, "H":0.296*um, "CH4_short":(0.1257)*um,"Kp":(0.3)*um, "Lp":(4.126 - 3.426)*um}

        lam_c = band_ctrs[self.band]
        lam_w = band_wdth[self.band]

        # wavelength info: spect mode or pol more
        if "PRISM" in self.mode:
            # GPI's spectral mode
            self.nwav = 36 #self.hdr1[0]["NAXIS3"]
            self.wls = np.linspace(lam_c - lam_w/2.0, lam_c+lam_w/2.0, num=36)*1e-6
            self.eff_band = um*np.ones(self.nwav)*(self.wls[-1] - self.wls[0])/self.nwav
            # For OIFits structure
            self.wavextension = (self.wls, self.eff_band)
        elif "BROADBAND" in self.mode:
            # Copied from GPI's pol mode.

            self.nwav=1

            # Define the bands in case we use a tophat filter
            wghts = np.ones(11)
            wavls = np.linspace(lam_c-lam_w/2.0, \
                                lam_c+lam_w/2.0, num=len(wghts))


            transmission = np.array([[wghts[f], wavls[f]] for f in range(len(wghts))])
            self.wls = [transmission, ]
            #self.wls = [np.sum(wghts*wavls) /float(len(wavls)), ]
            self.eff_band = np.array([lam_w, ])
            # For OIFits structure
            self.wavextension = ([lam_c,], [lam_w,])



        try:
            self.wls = [utils.combine_transmission(transmission, src), ]
        except:
            self.wls = [transmission, ]
        #self.wavextension = (lam_c, lam_w)
        self.nwav=1

        # finding centroid from phase slope only considered cv_phase data when cv_abs data exceeds this.  
        # absolute value of cv data normalized to unity maximum for the threshold application.
        self.cvsupport_threshold = {"threshold":0.02} # Gurus: tweak with use...

        self.ref_imgs_dir = "refimgs/"

        #############################
        # Observation info - I don't know yet how JWST data headers will be structured
        self.telname= "Keck"
        try:
            self.ra, self.dec = self.hdr[0]["RA"], self.hdr[0]["DEC"]
        except:
            self.ra, self.dec = 00, 00
        try:
            self.date = self.hdr[0]["DATE"]
            self.month = self.date[8:10]
            self.day = self.date[5:7]
            self.year = self.date[:4]
        except:
            lt = time.localtime()
            self.date = "{0}{1:02d}{2:02d}".format(lt[0],lt[1],lt[2])
            self.month = lt[1]
            self.day = lt[2]
            self.year = lt[0]


        self.parangs = []
        self.itime = []
        self.crpa = []
        self.rotposns = []
        self.instangs = []
        self.derotangs = []
        for ii in range(len(reffile)):
            self.parangs.append(self.hdr[ii]["PARANG"])
            self.rotposns.append(self.hdr[ii]["ROTPOSN"])
            self.instangs.append(self.hdr[ii]["INSTANGL"])
            # From Tom Esposito:
            # PARANG + ROTPOSN - INSTANGL - 0.262 
            self.derotangs.append(self.hdr[ii]["PARANG"]+self.hdr[ii]["ROTPOSN"] \
                                  -self.hdr[ii]["INSTANGL"]-0.262)
            self.itime.append(self.hdr[ii]["ITIME"])
            if "CRPA" in self.hdr[ii]:
                self.crpa.append(self.hdr[ii]["CRPA"])
        self.avderotang = np.mean(self.derotangs)
        self.avparang = np.mean(self.parangs)
        if len(self.crpa)>0:
            self.avcassang = np.mean(self.crpa)
        else:
            self.avcassang = 0.0
        self.parang_range = abs(self.parangs[-1] - self.parangs[0])
        self.totalinttime = np.sum(self.itime)


        try:
            self.pa = self.hdr0["PA"]
        except:
            self.pa = 00

        #############################:w
        self.parang_range = abs(self.parangs[-1] - self.parangs[0])
        self.totalinttime = np.sum(self.itime)
        self.ref_imgs_dir = "refimgs/"

    def read_data(self, fn):

        fitsfile = fits.open(fn)
        sci=fitsfile[0].data
        hdr=fitsfile[0].header
        fitsfile.close()
        #fitshdr = fitsfile[0].header
        self.sub_dir_str = fn.split("/")[-1][:-5]

        if len(sci.shape)==3:
            if self.mode=="PRISM":
                return sci, hdr
            elif self.mode=="BROADBAND":
                self.nwav=sci.shape[0]
                [self.wls.append(self.wls[0]) for f in range(self.nwav-1)]
                return sci, hdr
        elif len(sci.shape)==2:
            if self.mode=="BROADBAND":
                return np.array([sci,]), hdr
        else:
            sys.exit("invalid data dimensions for NIRC2. Should have dimensionality of 2 or 3.")

        return sci, hdr

