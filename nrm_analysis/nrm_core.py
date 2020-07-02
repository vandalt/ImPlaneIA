#! /usr/bin/env python

"""
by A. Greenbaum & A. Sivaramakrishnan 
April 2016 agreenba@pha.jhu.edu

Contains 

FringeFitter - fit fringe phases and amplitudes to data in the image plane

Calibrate - Calibrate target data with calibrator data

BinaryAnalyze - Detection, mcmc modeling, visualization tools
                Much of this section is based on tools in the pymask code
                by F. Martinache, B. Pope, and A. Cheetham
                We especially thank A. Cheetham for help and advising for
                developing the analysis tools in this package. 

    LG++ anand@stsci.edu nrm_core changes:
        Removed use of 'centering' parameter, switch to psf_offsets, meant to be uniformly ds9-compatible 
        offsets from array center (regardless of even/odd array sizes).

            nrm_core.fit_image(): refslice UNTESTED w/ new utils.centroid()
            nrm_core.fit_image(): hold_centering UNTESTED w/ new utils.centroid()

"""


from __future__ import print_function
# Standard imports
import os, sys, time
import numpy as np
from astropy.io import fits
from scipy.special import comb
from scipy.stats import sem, mstats
import pickle as pickle
import matplotlib.pyplot as plt


# Module imports
from nrm_analysis.fringefitting.LG_Model import NRM_Model
from nrm_analysis.misctools import utils  # AS LG++
from nrm_analysis.misctools import implane2oifits 
from nrm_analysis.modeling.binarymodel import model_cp_uv, model_allvis_uv, model_v2_uv, model_t3amp_uv
from nrm_analysis.modeling.multimodel import model_bispec_uv

from multiprocessing import Pool

class FringeFitter:
    def __init__(self, instrument_data, **kwargs):
        """
        Fit fringes in the image plane

        Takes an instance of the appropriate instrument class
        Various options can be set

        kwarg options:
        oversample - model oversampling (also how fine to measure the centering)
        psf_offset - If you already know the subpixel centering of your data, give it here
                     (not recommended except when debugging with perfectly know image placement))
        savedir - Where do you want to save the new files to? Default is working directory.
        #datadir - Where is your data? Default is working directory.
        npix - How many pixels of your data do you want to use? 
               Default is the shape of a data [slice or frame].  Typically odd?
        debug - will plot the FT of your data next to the FT of a reference PSF.
                Needs poppy package to run
        verbose_save - saves more than the standard files
        interactive - default True, prompts user to overwrite/create fresh directory.  
                      False will overwrite files where necessary.
        find_rotation - will find the best pupil rotation that matches the data

        main method:
        * fit_fringes


        """
        self.instrument_data = instrument_data

        #######################################################################
        # Options
        if "oversample" in kwargs:
            self.oversample = kwargs["oversample"]
        else:
            #default oversampling is 3
            self.oversample = 3
        if "find_rotation" in kwargs:
            # can be True/False or 1/0
            self.find_rotation = kwargs["find_rotation"]
        else:
            self.find_rotation = False
        if "psf_offset_ff" in kwargs: # if so do not find center of image in data
            self.psf_offset_ff = kwargs["psf_offset_ff"]         
        else:
            self.psf_offset_ff = None #find center of image in data
        if "savedir" in kwargs:
            self.savedir = kwargs["savedir"]
        else:
            self.savedir = os.getcwd()
        if "npix" in kwargs:
            self.npix = kwargs["npix"]
        else:
            self.npix = 'default'
        if "debug" in kwargs:
            self.debug=kwargs["debug"]
        else:
            self.debug=False
        if "verbose_save" in kwargs:
            self.verbose_save = kwargs["verbose_save"]
        else:
            self.verbose_save = False
        if 'interactive' in kwargs:
            self.interactive = kwargs['interactive']
        else:
            self.interactive = True
        if "save_txt_only" in kwargs:
            self.save_txt_only = kwargs["save_txt_only"]
        else:
            self.save_txt_only = False
        if "oifprefix" in kwargs:
            self.oifprefix = kwargs["oifprefix"]
        else:
            self.oifprefix = "oifprefix_"
        #######################################################################


        #######################################################################
        # Create directories if they don't already exit
        try:
            os.mkdir(self.savedir)
        except:
            if self.interactive is True:
                print(self.savedir+" Already exists, rewrite its contents? (y/n)")
                ans = input()
                if ans == "y":
                    pass
                elif ans == "n":
                    sys.exit("use alternative save directory with kwarg 'savedir' when calling FringeFitter")
                else:
                    sys.exit("Invalid answer. Stopping.")
            else:
                pass

    ###
    # May 2017 J Sahlmann updates: parallelized fringe-fitting!
    ###

    def fit_fringes(self, fns, threads = 0):
        if type(fns) == str:
            fns = [fns, ]

        # Can get fringes for images in parallel
        store_dict = [{"object":self, "file":                 fn,"id":jj} \
                        for jj,fn in enumerate(fns)]

        t2 = time.time()
        for jj, fn in enumerate(fns):
            fit_fringes_parallel({"object":self, "file":                  fn,\
                                  "id":jj},threads)
        t3 = time.time()
        print("Parallel with {0} threads took {1:.2f}s to fit all fringes".format(\
               threads, t3-t2))


    def save_output(self, slc, nrm):
        # cropped & centered PSF
        self.datapeak = self.ctrd.max()
        #TBD: Keep only n_*.fits files after testing is done and before doing ImPlaneIA delivery
        if self.save_txt_only==False:
            fits.PrimaryHDU(data=self.ctrd, \
                    header=self.scihdr).writeto(self.savedir+\
                    self.sub_dir_str+"/centered_{0}.fits".format(slc), \
                    overwrite=True)
            fits.PrimaryHDU(data=self.ctrd/self.datapeak, \
                    header=self.scihdr).writeto(self.savedir+\
                    self.sub_dir_str+"/n_centered_{0}.fits".format(slc), \
                    overwrite=True)

            model, modelhdu = nrm.plot_model(fits_true=1)
            # save to fits files
            fits.PrimaryHDU(data=nrm.residual).writeto(self.savedir+\
                        self.sub_dir_str+"/residual_{0:02d}.fits".format(slc), \
                        overwrite=True)
            fits.PrimaryHDU(data=nrm.residual/self.datapeak).writeto(self.savedir+\
                        self.sub_dir_str+"/n_residual_{0:02d}.fits".format(slc), \
                        overwrite=True)
            modelhdu.writeto(self.savedir+\
                        self.sub_dir_str+"/modelsolution_{0:02d}.fits".format(slc),\
                        overwrite=True)
            fits.PrimaryHDU(data=model/self.datapeak, \
                    header=modelhdu.header).writeto(self.savedir+\
                    self.sub_dir_str+"/n_modelsolution_{0:02d}.fits".format(slc), \
                    overwrite=True)
        else:
            print("nrm_core: NOT SAVING ANY FITS FILES. SET save_txt_only=False TO SAVE.")

        # default save to text files
        np.savetxt(self.savedir+self.sub_dir_str + \
                   "/solutions_{0:02d}.txt".format(slc), nrm.soln)
        np.savetxt(self.savedir+self.sub_dir_str + \
                   "/phases_{0:02d}.txt".format(slc), nrm.fringephase)
        np.savetxt(self.savedir+self.sub_dir_str + \
                   "/amplitudes_{0:02d}.txt".format(slc), nrm.fringeamp)
        np.savetxt(self.savedir+self.sub_dir_str + \
                   "/CPs_{0:02d}.txt".format(slc), nrm.redundant_cps)
        np.savetxt(self.savedir+self.sub_dir_str + \
                   "/CAs_{0:02d}.txt".format(slc), nrm.redundant_cas)
        np.savetxt(self.savedir+self.sub_dir_str + \
                  "/fringepistons_{0:02d}.txt".format(slc), nrm.fringepistons)

        # write info that oifits wants only when writing out first slice.
        # this will relevant to all slices... so no slice number here.
        if slc == 0:
            pfn = self.savedir+self.sub_dir_str + "/info4oif_dict.pkl"
            pfd = open(pfn, 'wb')
            pickle.dump(self.instrument_data.info4oif_dict, pfd)
            pfd.close()

        # Read in all relevant text observables and save to oifits file...
        dct = implane2oifits.oitxt2oif(nh=7, oitxtdir=self.savedir+self.sub_dir_str+'/' ,
                                             oifprefix=self.oifprefix,
                                             datadir=self.savedir+'/') # .savedir)


        # optional save outputs
        if self.verbose_save:
            np.savetxt(self.savedir+self.sub_dir_str+\
                       "/condition_{0:02d}.txt".format(slc), nrm.cond)
            np.savetxt(self.savedir+self.sub_dir_str+\
                       "/flux_{0:02d}.txt".format(slc), nrm.flux)
          
        #print(nrm.linfit_result)
        if nrm.linfit_result is not None:          
            # save linearfit results to pickle file
            myPickleFile = os.path.join(self.savedir+self.sub_dir_str,"linearfit_result_{0:02d}.pkl".format(slc))
            with open( myPickleFile , "wb" ) as f:
                pickle.dump((nrm.linfit_result), f) 
            print("Wrote pickled file  %s" % myPickleFile)
                       

    def save_auto_figs(self, slc, nrm):
        
        # rotation
        if self.find_rotation==True:
            plt.figure()
            plt.plot(nrm.rots, nrm.corrs)
            plt.vlines(nrm.rot_measured, nrm.corrs[0],
                        nrm.corrs[-1], linestyles='--', color='r')
            plt.text(nrm.rots[1], nrm.corrs[1], 
                     "best fit at {0}".format(nrm.rot_measured))
            plt.savefig(self.savedir+self.sub_dir_str+\
                        "/rotationcorrelation_{0:02d}.png".format(slc))

def fit_fringes_parallel(args, threads):
    self = args['object']
    filename = args['file']
    id_tag = args['id']
    self.prihdr, self.scihdr, self.scidata = self.instrument_data.read_data(filename)
    self.sub_dir_str = self.instrument_data.sub_dir_str
    try:
        os.mkdir(self.savedir+self.sub_dir_str)
    except:
        pass

    store_dict = [{"object":self, "slc":slc} for slc in \
                  range(self.instrument_data.nwav)]

    if threads>0:
        pool = Pool(processes=threads)
        print("Running fit_fringes in parallel with {0} threads".format(threads))
        pool.map(fit_fringes_single_integration, store_dict)
        pool.close()
        pool.join()

    else:
        for slc in range(self.instrument_data.nwav):
            fit_fringes_single_integration({"object":self, "slc":slc})

def fit_fringes_single_integration(args):
    self = args["object"]
    slc = args["slc"]
    id_tag = args["slc"]
    nrm = NRM_Model(mask=self.instrument_data.mask,
                    pixscale=self.instrument_data.pscale_rad,
                    holeshape=self.instrument_data.holeshape,
                    affine2d=self.instrument_data.affine2d,
                    over = self.oversample)

    nrm.bandpass = self.instrument_data.wls[slc]

    if self.npix == 'default':
        self.npix = self.scidata[slc,:,:].shape[0]

    DBG = False # AS testing gross psf orientatioun while getting to LG++ beta release 2018 09
    if DBG:
        nrm.simulate(fov=self.npix, bandpass=self.instrument_data.wls[slc], over=self.oversample)
        fits.PrimaryHDU(data=nrm.psf).writeto(self.savedir + "perfect.fits", overwrite=True)

    # New or modified in LG++
    # center the image on its peak pixel:
    # AS subtract 1 from "r" below  for testing >1/2 pixel offsets
    # AG 03-2019 -- is above comment still relevant?
    
    if self.instrument_data.arrname=="NIRC2_9NRM":
        self.ctrd = utils.center_imagepeak(self.scidata[slc, :,:], 
                        r = (self.npix -1)//2 - 2, cntrimg=False)  
    elif self.instrument_data.arrname=="gpi_g10s40":
        self.ctrd = utils.center_imagepeak(self.scidata[slc, :,:], 
                        r = (self.npix -1)//2 - 2, cntrimg=True)  
    else:
        self.ctrd = utils.center_imagepeak(self.scidata[slc, :,:])  
    

    nrm.reference = self.ctrd  # self. is the cropped image centered on the brightest pixel
    if self.psf_offset_ff is None:
        # returned values have offsets x-y flipped:
        # Finding centroids the Fourier way assumes no bad pixels case - Fourier domain mean slope
        centroid = utils.find_centroid(self.ctrd, self.instrument_data.threshold) # offsets from brightest pixel ctr
        # use flipped centroids to update centroid of image for JWST - check parity for GPI, Vizier,...
        # pixel coordinates: - note the flip of [0] and [1] to match DS9 view
        image_center = utils.centerpoint(self.ctrd.shape) + np.array((centroid[1], centroid[0])) # info only, unused
        nrm.xpos = centroid[1]  # flip 0 and 1 to convert
        nrm.ypos = centroid[0]  # flip 0 and 1
        nrm.psf_offset = nrm.xpos, nrm.ypos  # renamed .bestcenter to .psf_offset
        if self.debug: print("nrm.core.fit_fringes_single_integration: utils.find_centroid() -> nrm.psf_offset")
    else:
        nrm.psf_offset = self.psf_offset_ff # user-provided psf_offset python-style offsets from array center are here.


    nrm.make_model(fov = self.ctrd.shape[0], bandpass=nrm.bandpass, 
                   over=self.oversample,
                   psf_offset=nrm.psf_offset,  
                   pixscale=nrm.pixel)
    nrm.fit_image(self.ctrd, modelin=nrm.model, psf_offset=nrm.psf_offset)

    """
    Attributes now stored in nrm object:

    -----------------------------------------------------------------------------
    soln            --- resulting sin/cos coefficients from least squares fitting
    fringephase     --- baseline phases in radians
    fringeamp       --- baseline amplitudes (flux normalized)
    redundant_cps   --- closure phases in radians
    redundant_cas   --- closure amplitudes
    residual        --- fit residuals [data - model solution]
    cond            --- matrix condition for inversion
    fringepistons   --- zero-mean piston opd in radians on each hole (eigenphases)
    -----------------------------------------------------------------------------
    """

    if self.debug==True:
        import matplotlib.pyplot as plt
        import poppy.matrixDFT as mft
        dataft = mft.matrix_dft(self.ctrd, 256, 512)
        refft = mft.matrix_dft(self.refpsf, 256, 512)
        plt.figure()
        plt.title("Data")
        plt.imshow(np.sqrt(abs(dataft)), cmap = "bone")
        plt.figure()
        plt.title("Reference")
        plt.imshow(np.sqrt(abs(refft)), cmap="bone")
        plt.show()
    
    self.save_output(slc, nrm)
    self.nrm = nrm # store  extracted values here
    return None

'''
class Calibrate:

    def _from_gpi_header(fitsfiles):
        """
        Things I think are important. Average the parang measurements
        """
        parang=[]
        pa = []
        for fitsfile in fitsfiles:
            f = fits.open(fitsfile)
            hdr = f[0].header
            f.close()
            ra = hdr['RA']
            dec = hdr['DEC']
            parang.append(hdr['PAR_ANG'] - 1.00) # degree pa offset from 2014 SPIE +/- 0.03
            pa.append(hdr['PA'])
        return ra, dec, np.mean(parang), np.mean(pa)
'''
