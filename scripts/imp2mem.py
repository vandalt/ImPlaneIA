#! /usr/bin/env python

"""
Reads ImPlaneIA output text of [CPs,phases,amplitudes]_nn.txt in one directory
Writes them out into oifits files in the same directory.

Chunkwide averaging might be a later development for real data.

anand@stsci.edu started 2019 09 26 
anand@stsci.edu beta 2019 12 04

"""

import datetime
import glob
import os  # , time
import pickle
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astroquery.simbad import Simbad
#from scipy.stats import mstats, sem
from matplotlib import pyplot as plt
from scipy.special import comb
from termcolor import cprint


class ObservablesFromText():
    """
        anand@stsci.edu 2019_09_27
    """

    def __init__(self, nh, txtpath=None,
                 oifpath=None,
                 observables=("phases", "amplitudes", "CPs", "CAs"),
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
                     many for IFU or repeated integration observations (int)
            observables: If ("phases", "amplitudes", "CPs", "CAs") for example - ImPlaneIA nomenclature
                       then the files need to be named: "/phases_{0:02d}.txt".format(slc)
                       Three or four quantities (omitting CA is optional)
                       Order is relevant... pha, amp, cp [, ca]
                       Implaneia txt data will be in txtpath/*.txt
            oifinfofn: default 'info4oif_dict.pkl' suits ImPlaneIA
                      Pickle file of info oifits writers might need, in a dictionary
                      Located in the same dir as text observable files.  Only one for all slices...
                      Implaneia writes this dictionary out with the default name.
            If you want to trim low and/or high ends of eg IFU spectral observables trim them
            on-the-fly before calling this routine.

            ImPlaneIA saves fp cp in RADIANS.

            Units: as SI as possible.

        """
        print("=== ObservablesFromText ===\n One object's *.txt observables' directory path:\n    ", txtpath)
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
        self.nslices = len(
            glob.glob(self.txtpath+'/{0:s}*.txt'.format(self.observables[0])))
        if verbose:
            print(self.nslices, "slices' observables text files found")
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
        if verbose:
            print("assumes angles in", angunit)
        if verbose:
            print("angle unit:", angunit)
        if angunit == 'radians':
            self.degree = 180.0 / np.pi
        else:
            self.degree = 1

        self._readtxtdata()
        if self.verbose:
            self._showdata()

    def _makequads_all(self):
        """ returns int array of quad hole indices (0-based), 
            and float array of three uvw vectors in all quads
        """
        nholes = self.ctrs.shape[0]
        qlist = []
        for i in range(nholes):
            for j in range(nholes):
                for k in range(nholes):
                    for q in range(nholes):
                        if i < j and j < k and k < q:
                            qlist.append((i, j, k, q))
        qarray = np.array(qlist).astype(np.int)
        if self.verbose:
            print("qarray", qarray.shape, "\n", qarray)
        qname = []
        uvwlist = []
        # foreach row of 3 elts...
        for quad in qarray:
            qname.append("{0:d}_{1:d}_{2:d}_{3:d}".format(
                quad[0], quad[1], quad[2], quad[3]))
            if self.verbose:
                print('quad:', quad, qname[-1])
            uvwlist.append((self.ctrs[quad[0]] - self.ctrs[quad[1]],
                            self.ctrs[quad[1]] - self.ctrs[quad[2]],
                            self.ctrs[quad[2]] - self.ctrs[quad[3]]))
        if self.verbose:
            print(qarray.shape, np.array(uvwlist).shape)
        return qarray, np.array(uvwlist)

    def _maketriples_all(self):
        """ returns int array of triple hole indices (0-based), 
            and float array of two uv vectors in all triangles
        """
        nholes = self.ctrs.shape[0]
        tlist = []
        for i in range(nholes):
            for j in range(nholes):
                for k in range(nholes):
                    if i < j and j < k:
                        tlist.append((i, j, k))
        tarray = np.array(tlist).astype(np.int)
        if self.verbose:
            print("tarray", tarray.shape, "\n", tarray)

        tname = []
        uvlist = []
        # foreach row of 3 elts...
        for triple in tarray:
            tname.append("{0:d}_{1:d}_{2:d}".format(
                triple[0], triple[1], triple[2]))
            if self.verbose:
                print('triple:', triple, tname[-1])
            uvlist.append((self.ctrs[triple[0]] - self.ctrs[triple[1]],
                           self.ctrs[triple[1]] - self.ctrs[triple[2]]))
        #print(len(uvlist), "uvlist", uvlist)
        if self.verbose:
            print(tarray.shape, np.array(uvlist).shape)
        return tarray, np.array(uvlist)

    def _makebaselines(self):
        """
        ctrs (nh,2) in m
        returns np arrays of eg 21 baselinenames ('0_1',...), eg (21,2) baselinevectors (2-floats)
        in the same numbering as implaneia
        """
        nholes = self.ctrs.shape[0]
        blist = []
        for i in range(nholes):
            for j in range(nholes):
                if i < j:
                    blist.append((i, j))
        barray = np.array(blist).astype(np.int)
        #blname = []
        bllist = []
        for basepair in blist:
            # blname.append("{0:d}_{1:d}".format(basepair[0],basepair[1]))
            baseline = self.ctrs[basepair[0]] - self.ctrs[basepair[1]]
            bllist.append(baseline)
        return barray, np.array(bllist)

    def _showdata(self, prec=4):
        """ set precision of your choice in calling this"""
        print('nh {0:d}  nslices {1:d}  nbl {2:d}  ncp {3:d}  nca {4:d}  '.format(
            self.nh, self.nslices, self.nbl, self.ncp, self.nca), end="")
        print("observables in np arrays with {:d} rows".format(self.nslices))

        if len(self.observables) == 4:
            print('nca', self.nca)
        else:
            print()
        np.set_printoptions(precision=prec)

        print(self.fp.shape, "fp (degrees, but stored internally in radians):\n",
              self.fp*self.degree, "\n")
        print(self.fa.shape, "fa:\n", self.fa, "\n")

        print(self.cp.shape, "cp (degrees, but stored internally in radians):\n",
              self.cp*self.degree, "\n")
        if len(self.observables) == 4:
            print(self.ca.shape, "ca:\n", self.ca, "\n")
        # print(self.info4oif_dict)

        print("hole centers array shape:", self.ctrs.shape)

        print(len(self.bholes), "baseline hole indices\n", self.bholes)
        print(self.bls.shape, "baselines:\n", self.bls)

        print(self.tholes.shape, "triple hole indices:\n", self.tholes)
        print(self.tuv.shape, "triple uv vectors:\n", self.tuv)

        print(self.qholes.shape, "quad hole indices:\n", self.qholes)
        print(self.quvw.shape, "quad uvw vectors:\n", self.quvw)

    def _readtxtdata(self):
        # to only be called from init
        # loop through all the requested observables,
        # read in the exposure slices of a data cube
        # or the wavelength slices of IFU?

        # set up files to read
        # What do we do for IFU or Pol?
        # file name for each exposure (slice) in an image cube with nslices exposures
        fnheads = []
        if self.verbose:
            print("\tfile names that are being looked for:")
        for obsname in self.observables:
            # ImPlaneIA-specific filenames
            fnheads.append(self.txtpath+"/"+obsname+"_{0:02d}.txt")
            if self.verbose:
                print("\t"+fnheads[-1])

        # load from text into data rrays:
        for slice in range(self.nslices):
            self.fp[slice:] = np.loadtxt(fnheads[0].format(slice))
            self.fa[slice:] = np.loadtxt(fnheads[1].format(slice))
            self.cp[slice:] = np.loadtxt(fnheads[2].format(slice))
            if len(self.observables) == 4:
                self.ca[slice:] = np.loadtxt(fnheads[3].format(slice))

        # read in pickle of the info oifits might need...
        pfd = open(self.txtpath+'/'+self.oifinfofn, 'rb')
        self.info4oif_dict = pickle.load(pfd)
        if self.verbose:
            for key in self.info4oif_dict.keys():
                print(key)
        pfd.close()
        self.ctrs = self.info4oif_dict['ctrs']
        self.bholes, self.bls = self._makebaselines()
        self.tholes, self.tuv = self._maketriples_all()
        self.qholes, self.quvw = self._makequads_all()

    def write_oif(self, oifn):
        """
            Write out an OIFITS file using the data that was read in during initialisation
        """
        print("I don't know how to write " + oifn + " at this moment.")


def Format_STAINDEX_V2(tab):
    STA_INDEX = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        line = np.array([ap1, ap2]) + 1
        STA_INDEX.append(line)
    return STA_INDEX


def Format_STAINDEX_T3(tab):
    STA_INDEX = []
    for x in tab:
        ap1 = int(x[0])
        ap2 = int(x[1])
        ap3 = int(x[2])
        line = np.array([ap1, ap2, ap3]) + 1
        STA_INDEX.append(line)
    return STA_INDEX


def NRMtoOifits2(dic, filename=None, saveoifdir=None, verbose=False):
    """
    Save dictionary formatted data into a proper OIFITS (version 2) format file.

    Parameters:
    -----------

    dic: dict
        Dictionary containing all extracted data (keys: 'OI_VIS2', 'OI_VIS', 'OI_T3', 'OI_WAVELENGTH', 'info'),
    filename: str
        By default None, the filename is constructed using informations included in the input dictionary ('info'),
    saveofdir: saveoifdir
        default None, so the oifits is written to cwd/Saveoifits/, otherwise to saveoifdir
        Handed off to Anthony's 'datadir' variable.

    """

    if dic is not None:
        pass
    else:
        cprint('\nError NRMtoOifits2 : Wrong data format!', on_color='on_red')
        return None

    datadir = saveoifdir
    if datadir is None: datadir = 'Saveoifits/'
    else: 
        if datadir[-1] != '/':  datadir = datadir + '/'
    if not os.path.exists(datadir):
        print('### Create %s directory to save all requested Oifits ###' % datadir)
        os.system('mkdir %s' % datadir)

    if type(filename) == str:
        pass
    else:
        filename = '%s_%s_%s_%s_%2.0f.oifits' % (dic['info']['TARGET'].replace(
            ' ', ''), dic['info']['INSTRUME'], dic['info']['MASK'], dic['info']['FILT'], dic['info']['MJD'])

    # ------------------------------
    #       Creation OIFITS
    # ------------------------------
    if verbose:
        print("\n\n### Init creation of OI_FITS (%s) :" % (filename))

    hdulist = fits.HDUList()

    hdu = fits.PrimaryHDU()
    hdu.header['DATE'] = datetime.datetime.now().strftime(
        format='%F')  # , 'Creation date'
    hdu.header['ORIGIN'] = 'Sydney University'
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS']
    hdu.header['CONTENT'] = 'OIFITS2'
    hdu.header['TELESCOP'] = dic['info']['TELESCOP']
    hdu.header['INSTRUME'] = dic['info']['INSTRUME']
    hdu.header['OBSERVER'] = dic['info']['OBSERVER']
    hdu.header['OBJECT'] = dic['info']['OBJECT']
    hdu.header['INSMODE'] = dic['info']['INSMODE']

    hdulist.append(hdu)
    # ------------------------------
    #        OI Wavelength
    # ------------------------------

    if verbose:
        print('-> Including OI Wavelength table...')
    data = dic['OI_WAVELENGTH']

    # Data
    # -> Initiation new hdu table :
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
        fits.Column(name='EFF_WAVE', format='1E',
                    unit='METERS', array=[data['EFF_WAVE']]),
        fits.Column(name='EFF_BAND', format='1E',
                    unit='METERS', array=[data['EFF_BAND']])
    )))

    # Header
    hdu.header['EXTNAME'] = 'OI_WAVELENGTH'
    hdu.header['OI_REVN'] = 2  # , 'Revision number of the table definition'
    # 'Name of detector, for cross-referencing'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdulist.append(hdu)  # Add current HDU to the final fits file.

    # ------------------------------
    #          OI Target
    # ------------------------------
    if verbose:
        print('-> Including OI Target table...')

    name_star = dic['info']['TARGET']

    customSimbad = Simbad()
    customSimbad.add_votable_fields('propermotions', 'sptype', 'parallax')

    # Add information from Simbad:

    try:
        if name_star == 'UNKNOWN':
            ra = [0]
            dec = [0]
            spectyp = ['fake']
            pmra = [0]
            pmdec = [0]
            plx = [0]
        else:
            query = customSimbad.query_object(name_star)
            coord = SkyCoord(query['RA'][0]+' '+query['DEC']
                             [0], unit=(u.hourangle, u.deg))

            ra = [coord.ra.deg]
            dec = [coord.dec.deg]
            spectyp = query['SP_TYPE']
            pmra = query['PMRA']
            pmdec = query['PMDEC']
            plx = query['PLX_VALUE']
    except:
        ra = [0]
        dec = [0]
        spectyp = ['fake']
        pmra = [0]
        pmdec = [0]
        plx = [0]

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
        fits.Column(name='TARGET_ID', format='1I', array=[1]),
        fits.Column(name='TARGET', format='16A', array=[name_star]),
        fits.Column(name='RAEP0', format='1D', unit='DEGREES', array=ra),
        fits.Column(name='DECEP0', format='1D', unit='DEGREES', array=dec),
        fits.Column(name='EQUINOX', format='1E', unit='YEARS', array=[2000]),
        fits.Column(name='RA_ERR', format='1D', unit='DEGREES', array=[0]),
        fits.Column(name='DEC_ERR', format='1D', unit='DEGREES', array=[0]),
        fits.Column(name='SYSVEL', format='1D', unit='M/S', array=[0]),
        fits.Column(name='VELTYP', format='8A', array=['UNKNOWN']),
        fits.Column(name='VELDEF', format='8A', array=['OPTICAL']),
        fits.Column(name='PMRA', format='1D', unit='DEG/YR', array=pmra),
        fits.Column(name='PMDEC', format='1D', unit='DEG/YR', array=pmdec),
        fits.Column(name='PMRA_ERR', format='1D', unit='DEG/YR', array=[0]),
        fits.Column(name='PMDEC_ERR', format='1D', unit='DEG/YR', array=[0]),
        fits.Column(name='PARALLAX', format='1E', unit='DEGREES', array=plx),
        fits.Column(name='PARA_ERR', format='1E', unit='DEGREES', array=[0]),
        fits.Column(name='SPECTYP', format='16A', array=spectyp)
    )))

    hdu.header['EXTNAME'] = 'OI_TARGET'
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdulist.append(hdu)

    # ------------------------------
    #           OI Array
    # ------------------------------

    if verbose:
        print('-> Including OI Array table...')

    STAXY = dic['info']['STAXY']

    N_ap = len(STAXY)

    TEL_NAME = ['A%i' % x for x in np.arange(N_ap)+1]
    STA_NAME = TEL_NAME
    DIAMETER = [0] * N_ap

    STAXYZ = []
    for x in STAXY:
        a = list(x)
        line = [a[0], a[1], 0]
        STAXYZ.append(line)

    STA_INDEX = np.arange(N_ap) + 1

    PSCALE = dic['info']['PSCALE']/1000.  # arcsec
    ISZ = dic['info']['ISZ']  # Size of the image to extract NRM data
    FOV = [PSCALE * ISZ] * N_ap
    FOVTYPE = ['RADIUS'] * N_ap

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
        fits.Column(name='TEL_NAME', format='16A',
                    array=TEL_NAME),  # ['dummy']),
        fits.Column(name='STA_NAME', format='16A',
                    array=STA_NAME),  # ['dummy']),
        fits.Column(name='STA_INDEX', format='1I', array=STA_INDEX),
        fits.Column(name='DIAMETER', unit='METERS',
                    format='1E', array=DIAMETER),
        fits.Column(name='STAXYZ', unit='METERS', format='3D', array=STAXYZ),
        fits.Column(name='FOV', unit='ARCSEC', format='1D', array=FOV),
        fits.Column(name='FOVTYPE', format='6A', array=FOVTYPE),
    )))

    hdu.header['EXTNAME'] = 'OI_ARRAY'
    hdu.header['ARRAYX'] = float(0)
    hdu.header['ARRAYY'] = float(0)
    hdu.header['ARRAYZ'] = float(0)
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['FRAME'] = 'SKY'
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'

    hdulist.append(hdu)

    # ------------------------------
    #           OI VIS
    # ------------------------------

    if verbose:
        print('-> Including OI Vis table...')

    data = dic['OI_VIS']
    npts = len(dic['OI_VIS']['VISAMP'])

    STA_INDEX = Format_STAINDEX_V2(data['STA_INDEX'])

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name='TARGET_ID', format='1I',
                    array=[data['TARGET_ID']]*npts),
        fits.Column(name='TIME', format='1D', unit='SECONDS',
                    array=[data['TIME']]*npts),
        fits.Column(name='MJD', unit='DAY', format='1D',
                    array=[data['MJD']]*npts),
        fits.Column(name='INT_TIME', format='1D', unit='SECONDS',
                    array=[data['INT_TIME']]*npts),
        fits.Column(name='VISAMP', format='1D', array=data['VISAMP']),
        fits.Column(name='VISAMPERR', format='1D', array=data['VISAMPERR']),
        fits.Column(name='VISPHI', format='1D', unit='DEGREES',
                    array=np.rad2deg(data['VISPHI'])),
        fits.Column(name='VISPHIERR', format='1D', unit='DEGREES',
                    array=np.rad2deg(data['VISPHIERR'])),
        fits.Column(name='UCOORD', format='1D',
                    unit='METERS', array=data['UCOORD']),
        fits.Column(name='VCOORD', format='1D',
                    unit='METERS', array=data['VCOORD']),
        fits.Column(name='STA_INDEX', format='2I', array=STA_INDEX),
        fits.Column(name='FLAG', format='1L', array=data['FLAG'])
    ]))

    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['EXTNAME'] = 'OI_VIS'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS'], 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    # ------------------------------
    #           OI VIS2
    # ------------------------------

    if verbose:
        print('-> Including OI Vis2 table...')

    data = dic['OI_VIS2']
    npts = len(dic['OI_VIS2']['VIS2DATA'])

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name='TARGET_ID', format='1I',
                    array=[data['TARGET_ID']]*npts),
        fits.Column(name='TIME', format='1D', unit='SECONDS',
                    array=[data['TIME']]*npts),
        fits.Column(name='MJD', unit='DAY', format='1D',
                    array=[data['MJD']]*npts),
        fits.Column(name='INT_TIME', format='1D', unit='SECONDS',
                    array=[data['INT_TIME']]*npts),
        fits.Column(name='VIS2DATA', format='1D', array=data['VIS2DATA']),
        fits.Column(name='VIS2ERR', format='1D', array=data['VIS2ERR']),
        fits.Column(name='UCOORD', format='1D',
                    unit='METERS', array=data['UCOORD']),
        fits.Column(name='VCOORD', format='1D',
                    unit='METERS', array=data['VCOORD']),
        fits.Column(name='STA_INDEX', format='2I', array=STA_INDEX),
        fits.Column(name='FLAG', format='1L', array=data['FLAG'])
    ]))

    hdu.header['EXTNAME'] = 'OI_VIS2'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdu.header['ARRNAME'] = dic['info']['MASK']
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS'], 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    # ------------------------------
    #           OI T3
    # ------------------------------

    if verbose:
        print('-> Including OI T3 table...')

    data = dic['OI_T3']
    npts = len(dic['OI_T3']['T3PHI'])

    STA_INDEX = Format_STAINDEX_T3(data['STA_INDEX'])

    hdu = fits.BinTableHDU.from_columns(fits.ColDefs((
        fits.Column(name='TARGET_ID', format='1I', array=[1]*npts),
        fits.Column(name='TIME', format='1D', unit='SECONDS', array=[0]*npts),
        fits.Column(name='MJD', format='1D', unit='DAY',
                    array=[data['MJD']]*npts),
        fits.Column(name='INT_TIME', format='1D', unit='SECONDS',
                    array=[data['INT_TIME']]*npts),
        fits.Column(name='T3AMP', format='1D', array=data['T3AMP']),
        fits.Column(name='T3AMPERR', format='1D', array=data['T3AMPERR']),
        fits.Column(name='T3PHI', format='1D', unit='DEGREES',
                    array=np.rad2deg(data['T3PHI'])),
        fits.Column(name='T3PHIERR', format='1D', unit='DEGREES',
                    array=np.rad2deg(data['T3PHIERR'])),
        fits.Column(name='U1COORD', format='1D',
                    unit='METERS', array=data['U1COORD']),
        fits.Column(name='V1COORD', format='1D',
                    unit='METERS', array=data['V1COORD']),
        fits.Column(name='U2COORD', format='1D',
                    unit='METERS', array=data['U2COORD']),
        fits.Column(name='V2COORD', format='1D',
                    unit='METERS', array=data['V2COORD']),
        fits.Column(name='STA_INDEX', format='3I', array=STA_INDEX),
        fits.Column(name='FLAG', format='1L', array=data['FLAG'])
    )))

    hdu.header['EXTNAME'] = 'OI_T3'
    hdu.header['INSNAME'] = dic['info']['INSTRUME']
    hdu.header['OI_REVN'] = 2, 'Revision number of the table definition'
    hdu.header['DATE-OBS'] = dic['info']['DATE-OBS'], 'Zero-point for table (UTC)'
    hdulist.append(hdu)

    # ------------------------------
    #          Save file
    # ------------------------------

    hdulist.writeto(datadir + filename, overwrite=True)
    cprint('\n\n### OIFITS CREATED (%s).' % (datadir+filename), 'cyan')

    return None


class A(object):
    pass

# Class/function to transform dictionary into class like (keys accesible as dic.keys and not dic['keys'])
class Dict2Class:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if type(v) == dict:
                a = A()
                for key in v.keys():
                    a.__dict__[key] = v[key]
                setattr(self, k, a)
            else:
                setattr(self, k, v)


def Saved_filters(ins, filt, upload=False):
    dic_filt = {'JWST': {'F277W': [2.776, 0.715],
                         'F380M': [3.828, 0.205],
                         'F430M': [4.286, 0.202],
                         'F480M': [4.817, 0.298]
                         }
                }

    wl = dic_filt[ins][filt][0]*1e-6
    e_wl = dic_filt[ins][filt][1]*1e-6

    return wl, e_wl


def Plot_observables(tab, vmin=0, vmax=1.1, cpmin=-np.pi, cpmax=np.pi):
    cp = tab.cp
    cp_mean = np.mean(tab.cp, axis=0)
    cp_med = np.median(tab.cp, axis=0)

    Vis = tab.fa
    Vis_mean = np.mean(Vis, axis=0)
    Vis_med = np.median(Vis, axis=0)

    target = tab.info4oif_dict['objname']

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Uncalibrated Vis. (%s)' % target)
    plt.plot(Vis.transpose(), 'gray', alpha=.2)
    plt.plot(Vis_mean, 'k--', label='Mean')
    plt.plot(Vis_med, linestyle='--', color='crimson', label='Median')
    plt.xlabel('Index', color='dimgray', fontsize=12)
    plt.ylabel(r'Vis.', color='dimgray', fontsize=12)
    plt.ylim([vmin, vmax])
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.title('Uncalibrated CP (%s)' % target)
    plt.plot(cp.transpose(), 'gray', alpha=.2)
    plt.plot(cp_mean, 'k--', label='Mean')
    plt.plot(cp_med, linestyle='--', color='crimson', label='Median')
    plt.xlabel('Index', color='dimgray', fontsize=12)
    plt.ylabel(r'CP [rad]', color='dimgray', fontsize=12)
    plt.ylim([cpmin, cpmax])
    plt.legend(loc='best')
    plt.tight_layout()

# ef Calib_NRM(nrm, nrm_c, method='med'):  Anthony's original parameters - worked because nrm_t was global (in __main__)

def rad2mas(rad):
    """Convert angle in radians to milli arc-sec."""
    mas = rad * (3600.0 * 180 / np.pi) * 10.0 ** 3
    return mas

def oifits_show(dic, unit='arcsec', setlog=False, vmin=0, vmax=1.05, cmin=-4, cmax=4, unit_cp='rad'):
    """ Display u-v plan, squared visibilities (v2) and closure phase (cp) of oifits data.

    Parameters
    ----------
    `dic` : {dict}
        Dictionnary containing all data extracted from NRM instrument (see cal2dict function),\n
    `unit` : {str}, (optional),
        Unit of the fourier space (spatial frequencies), by default 'arcsec',\n
    `setlog` : {bool}, (optional),
        If True, v2 are plotted in log scale, by default False,\n
    `vmin` : {int}, (optional),
        Minimum v2 to set the Y axis, by default 0,\n
    `vmax` : {float}, ,
        Maximum v2 to set the Y axis, by default 1.05,\n
    `cmin` : {int}, (optional),
        Minimum cp to set the Y axis [rad], by default -4,\n
    `cmax` : {int}, (optional),
        Maximum cp to set the Y axis [rad], by default 4.\n
    """
    unit = 'arcsec'
    uv_scale = {'m': 1,
                'rad': 1/dic['OI_WAVELENGTH']['EFF_WAVE'],
                'arcsec': 1/dic['OI_WAVELENGTH']['EFF_WAVE']/rad2mas(1e-3),
                'lambda': 1/dic['OI_WAVELENGTH']['EFF_WAVE']/1e6}

    U = dic['OI_VIS2']['UCOORD']*uv_scale[unit]
    V = dic['OI_VIS2']['VCOORD']*uv_scale[unit]

    sp_freq_vis = dic['OI_VIS2']['FREQ'] * uv_scale[unit]

    V2 = dic['OI_VIS2']['VIS2DATA']
    e_V2 = dic['OI_VIS2']['VIS2ERR'] * 1

    sp_freq_cp = dic['OI_T3']['FREQ'] * uv_scale[unit]   # Anand KeyError: 'FREQ'   
    cp = dic['OI_T3']['T3PHI']
    e_cp = dic['OI_T3']['T3PHIERR']

    bmax = 1.2*np.max(np.sqrt(U**2+V**2))

    # Configuration fenetre
    fig = plt.figure(figsize=(16, 5.5))
    ax1 = plt.subplot2grid((2, 6), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=4)
    ax3 = plt.subplot2grid((2, 6), (1, 2), colspan=4)

    # Plot plan UV
    # -------
    fig.subplots_adjust(top=0.974,
                        bottom=0.091,
                        left=0.038,
                        right=0.996,
                        hspace=0.127,
                        wspace=0.35)

    #ax1.set_title('(u-v) plan', fontsize=12, color = 'grey', weight = 'bold')

    ax1.scatter(U, V, s=50, c='#00adb5',
                edgecolors='#364f6b', marker='o', alpha=1)
    ax1.scatter(-1*np.array(U), -1*np.array(V), s=50, c='#fc5185',
                edgecolors='#364f6b', marker='o', alpha=1)
    ax1.axis([bmax, -bmax, -bmax, bmax])
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.patch.set_facecolor('#f7f9fc')
    ax1.patch.set_alpha(1)
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    # ax1.yaxis.set_ticks_position('left')

    unitlabel = {'m': 'm',
                 'rad': 'rad$^{-1}$',
                 'arcsec': 'arcsec$^{-1}$',
                 'lambda': 'M$\lambda$'}

    ax1.set_xlabel(r'U [%s]' % unitlabel[unit])
    ax1.set_ylabel(r'V [%s]' % unitlabel[unit])
    ax1.grid(alpha=0.2)

    # Plot V2
    # -------
    #ax2.set_title('Visibility and closure phase', fontsize=12,color = 'grey', weight = 'bold')
    ax2.errorbar(sp_freq_vis, V2, yerr=e_V2, linestyle="None", capsize=1, ecolor='#364f6b', mfc='#00adb5', mec='#364f6b',
                 marker='.', elinewidth=0.5, alpha=1, ms=8)

    ax2.hlines(1, 0, 1.2*np.max(sp_freq_vis),
               lw=1, color='k', alpha=.2, linestyle='--')

    ax2.set_ylim([vmin, vmax])
    ax2.set_xlim([0, 1.2*np.max(sp_freq_vis)])
    ax2.set_ylabel(r'$V^2$')
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.patch.set_facecolor('#f7f9fc')  
    ax2.patch.set_alpha(1)
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    #ax2.set_xticklabels([])

    if setlog:
        ax2.set_yscale('log')
    ax2.grid(which='both', alpha=.2)

    # Plot CP
    # -------

    if unit_cp == 'rad':
        h1 = np.pi
        pass
    else:
        cp = np.rad2deg(cp).copy()
        e_cp = np.rad2deg(e_cp).copy()
        h1 = np.rad2deg(np.pi)
        
    cmin = -cmax

    ax3.errorbar(sp_freq_cp, cp, yerr=e_cp, linestyle="None", capsize=1, ecolor='#364f6b', mfc='#00adb5', mec='#364f6b',
                 marker='.', elinewidth=0.5, alpha=1, ms=8)
    ax3.hlines(h1, 0, 1.2*np.max(sp_freq_vis),
               lw=1, color='k', alpha=.2, ls='--')
    ax3.hlines(-h1, 0, 1.2*np.max(sp_freq_vis),
               lw=1, color='k', alpha=.2, ls='--')
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.patch.set_facecolor('#f7f9fc')
    ax3.patch.set_alpha(1)
    ax3.xaxis.set_ticks_position('none')
    ax3.yaxis.set_ticks_position('none')
    ax3.set_xlabel('Spatial frequency [cycle/arcsec]')
    ax3.set_ylabel('Clos. $\phi$ [%s]'%unit_cp)
    ax3.axis([0, 1.2*np.max(sp_freq_vis), cmin, cmax])
    ax3.grid(which='both', alpha=.2)
    plt.show()
    return fig

def Calib_NRM(nrm_t, nrm_c, method='med'):

    # calibration factor Vis. (supposed to be one)
    fact_calib_visamp = np.mean(nrm_c.fa, axis=0)
    # calibration factor Phase Vis. (supposed to be zero)
    fact_calib_visphi = np.mean(nrm_c.fp, axis=0)

    visamp_calibrated = nrm_t.fa/fact_calib_visamp
    visphi_calibrated = nrm_t.fp - fact_calib_visphi
    vis2_calibrated = visamp_calibrated**2

    if method == 'med':
        vis2 = np.median(vis2_calibrated, axis=0)  # V2
    else:
        vis2 = np.mean(vis2_calibrated, axis=0)  # V2

    e_vis2 = np.std(vis2_calibrated, axis=0)  # Error on V2

    if method == 'med':
        visamp = np.median(visamp_calibrated, axis=0)  # Vis. amp
    else:
        visamp = np.mean(visamp_calibrated, axis=0)  # Vis. amp

    e_visamp = np.std(visamp_calibrated, axis=0)  # Vis. amp

    if method == 'med':
        visphi = np.median(visphi_calibrated, axis=0)  # Vis. phase
    else:
        visphi = np.mean(visphi_calibrated, axis=0)  # Vis. phase

    e_visphi = np.std(visphi_calibrated, axis=0)  # Vis. phase

    # calibration factor closure amp (supposed to be one)
    fact_calib_cpamp = np.mean(nrm_c.ca, axis=0)
    # calibration factor closure phase (supposed to be zero)
    fact_calib_cpphi = np.mean(nrm_c.cp, axis=0)

    shift2pi = np.zeros(nrm_t.cp.shape)
    shift2pi[nrm_t.cp >= 6] = 2*np.pi
    shift2pi[nrm_t.cp <= -6] = -2*np.pi

    """ Anthony, is this your  _t or _c?
    nrm.cp -= shift2pi
    """
    nrm_t.cp -= shift2pi  # I'm guessing it's _t

    cp_cal = nrm_t.cp - fact_calib_cpphi
    cpamp_cal = nrm_t.ca/fact_calib_cpamp

    if method == 'med':
        cp = np.median(cp_cal, axis=0)
    else:
        cp = np.mean(cp_cal, axis=0)

    e_cp = np.std(cp_cal, axis=0)

    if method == 'med':
        cpamp = np.median(cpamp_cal, axis=0)
    else:
        cpamp = np.mean(cpamp_cal, axis=0)

    e_cpamp = np.std(cpamp_cal, axis=0)

    output = {'vis2': vis2,
              'e_vis2': e_vis2,
              'visamp': visamp,
              'e_visamp': e_visamp,
              'visphi': visphi,
              'e_visphi': e_visphi,
              'cp': cp,
              'e_cp': e_cp,
              'cpamp': cpamp,
              'e_cpamp': e_cpamp
              }

    return Dict2Class(output)


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
    observables = ObservablesFromText(nh, txtdir, verbose=verbose)
    print(observables.nslices,
          "slices of data were analysed by ImPlaneIA, and read in")
    # observables.write_oif("write_me_oifits.fits")
    return observables


plt.close('all')


# if __name__ == "__main__":
def implane2oifits2(objecttextdir_c, objecttextdir_t, saveoifdir=None):
    """
    Input:
    str objecttextdir_c = directory containing calibrator observables text file (ImPlaneIA output)
        e.g. "Simulated_data/cal_ov{:d}/c_dsk_100mas__F430M_81_flat_x11__00_mir".format(OV) 
    str objecttextdir_t =  = directory containing target observables text file (ImPlaneIA output)
        e.g. "Simulated_data/tgt_ov{:d}/t_dsk_100mas__F430M_81_flat_x11__00_mir".format(OV)
    str saveoifdir = directory to which you wish to write the calibrated oifits file
        defult results in file written to '/.Saveoifits/'
    Output: none
    """

    nrm_t = main_ansou(nh=7, txtdir=objecttextdir_t, verbose=False) # You invert here, it's good now.
    nrm_c = main_ansou(nh=7, txtdir=objecttextdir_c, verbose=False)

    info = nrm_t.info4oif_dict

    ctrs = info['ctrs']

    t = Time('%s-%s-%s' %
             (info['year'], info['month'], info['day']), format='fits')
    ins = info['telname']
    filt = info['filt']
    wl, e_wl = Saved_filters(ins, filt)

    bls = nrm_t.bls
    # Index 0 and 1 reversed to get the good u-v coverage (same fft)
    UCOORD = bls[:, 1]
    VCOORD = bls[:, 0]

    D = 6.5  # Primary mirror display

    theta = np.linspace(0, 2*np.pi, 100)

    x = D/2. * np.cos(theta)  # Primary mirror display
    y = D/2. * np.sin(theta)

    plt.figure(figsize=(14.2, 7))
    plt.subplot(1, 2, 1)
    # Index 0 and 1 reversed to get the good u-v coverage (same fft)
    plt.scatter(ctrs[:, 1], ctrs[:, 0], s=2e3, c='', edgecolors='navy')
    plt.scatter(-1000, 1000, s=5e1, c='',
                edgecolors='navy', label='Aperture mask')
    plt.plot(x, y, '--', color='gray', label='Primary mirror equivalent')

    plt.xlabel('Aperture x-coordinate/m')
    plt.ylabel('Aperture y-coordinate/m')
    plt.legend(fontsize=8)
    plt.axis([-4., 4., -4., 4.])

    plt.subplot(1, 2, 2)
    plt.scatter(UCOORD, VCOORD, s=1e2, c='', edgecolors='navy')
    plt.scatter(-UCOORD, -VCOORD, s=1e2, c='', edgecolors='crimson')

    plt.plot(0, 0, 'k+')
    plt.axis((-D, D, -D, D))
    plt.xlabel('Fourier u-coordinate/m')
    plt.ylabel('Fourier v-coordinate/m')
    plt.tight_layout()
    freq_vis = ((UCOORD**2 + VCOORD**2)**0.5)/wl/206264.806247  # arcsec-1
    flagVis = [False] * nrm_t.nbl

    tuv = nrm_t.tuv
    V1COORD = tuv[:, 0, 0]
    U1COORD = tuv[:, 0, 1]
    V2COORD = tuv[:, 1, 0]
    U2COORD = tuv[:, 1, 1]
    U3COORD = -(U1COORD+U2COORD)
    V3COORD = -(V1COORD+V2COORD)

    flagT3 = [False] * nrm_t.ncp

    Plot_observables(nrm_t)
    Plot_observables(nrm_c)  # Plot uncalibrated data

    #print(nrm_t, nrm_c)
    nrm = Calib_NRM(nrm_t, nrm_c)  # Calibrate target by calibrator

    dic = {'OI_VIS2': {'VIS2DATA': nrm.vis2,
                       'VIS2ERR': nrm.e_vis2,
                       'UCOORD': UCOORD,
                       'VCOORD': VCOORD,
                       'STA_INDEX':  nrm_t.bholes,
                       'MJD': t.mjd,
                       'INT_TIME': info['itime'],
                       'TIME': 0,
                       'TARGET_ID': 1,
                       'FLAG': flagVis,
                       'FREQ': freq_vis
                       },

           'OI_VIS': {'TARGET_ID': 1,
                      'TIME': 0,
                      'MJD': t.mjd,
                      'INT_TIME': info['itime'],
                      'VISAMP': nrm.visamp,
                      'VISAMPERR': nrm.e_visamp,
                      'VISPHI': nrm.visphi,
                      'VISPHIERR': nrm.e_visphi,
                      'UCOORD': UCOORD,
                      'VCOORD': VCOORD,
                      'STA_INDEX': nrm_t.bholes,
                      'FLAG': flagVis
                      },

           'OI_T3': {'MJD': t.mjd,
                     'INT_TIME': info['itime'],
                     'T3PHI': nrm.cp,
                     'T3PHIERR': nrm.e_cp,
                     'T3AMP': nrm.cpamp,
                     'T3AMPERR': nrm.e_cp,
                     'U1COORD': U1COORD,
                     'V1COORD': V1COORD,
                     'U2COORD': U2COORD,
                     'V2COORD': V2COORD,
                     'STA_INDEX': nrm_c.tholes,
                     'FLAG': flagT3,
                     },

           'OI_WAVELENGTH': {'EFF_WAVE': wl,
                             'EFF_BAND': e_wl
                             },

           'info': {'TARGET': info['objname'],
                    'CALIB': info['objname'],
                    'OBJECT': info['objname'],
                    'FILT': info['filt'],
                    'INSTRUME': info['instrument'],
                    'MASK': info['arrname'],
                    'MJD': t.mjd,
                    'DATE-OBS': t.fits,
                    'TELESCOP': info['telname'],
                    'OBSERVER': 'UNKNOWN',
                    'INSMODE': info['pupil'],
                    'PSCALE': info['pscale_mas'],
                    'STAXY': info['ctrs'],
                    'ISZ': 77,  # size of the image needed (or fov)
                    'NFILE': 0}
           }

    # Anand put this call inside Anthony's __main__ so it can be converted into a function.
    # Function to save oifits file (version 2)
    NRMtoOifits2(dic, saveoifdir=saveoifdir, verbose=False)
    #oifits_show(dic) # crashes on t3 dictionary 'FREQ' kwd not found # FIXME


if __name__ == "__main__":
    OV_main = 1
    textrootdir =  os.path.expanduser('~')+"/Downloads/asoulain_arch2019.12.07/"
    objecttextdir_c_main = textrootdir +\
        "Simulated_data/cal_ov{:d}/c_dsk_100mas__F430M_81_flat_x11__00_mir".format(
            OV_main)  # Calibrator result ImPlaneIA
    objecttextdir_t_main = textrootdir + \
        "Simulated_data/tgt_ov{:d}/t_dsk_100mas__F430M_81_flat_x11__00_mir".format(
            OV_main)  # Target result ImPlaneIA
    
    implane2oifits2(objecttextdir_c_main, 
                    objecttextdir_t_main,
                    saveoifdir=textrootdir+'Saveoifits/')
