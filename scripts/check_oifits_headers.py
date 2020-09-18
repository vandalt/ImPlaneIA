import os
import numpy as np
from astropy.io import fits
import glob

def check_oifits_headers(file1,file2):
    '''
    In each of the headers of the extensions in file 1, check if they are in the
    corresponding header of file 2. If not, print it.
    For use when file 1 works in the analysis code but file 2 doesn't.
    '''
    print("Finding keywords in headers of %s that are not in %s:\n" % (file1,file2))
    # read the headers into a master "header1" dict
    header1 = {}
    extlist = []
    with fits.open(file1) as hdulist1:
        # get the header and extension name of each extension
        for ext in np.arange(len(hdulist1)):
            hdr = hdulist1[ext].header
            #get the name of the extension to use as key
            if ext == 0:
                extname = 'PRIMARY'
            else:
                extname = hdr['EXTNAME']
            extlist.append(extname)
            # populate the dict
            header1[extname] = hdr
    # make a dict of headers from the second file, but using the ext names from the first
    header2 = {}
    with fits.open(file2) as hdulist2:
        ext_iter = iter(extlist)
        for extname in ext_iter:
            try:
                hdr = hdulist2[extname].header
                header2[extname] = hdr
            except KeyError:
                # what if one of the extensions in file1 is not in file2?
                # this does not work currently
                next(ext_iter, None)
                hdr = hdulist2[extname].header
                header2[extname] = hdr
                continue
    # check which header keywords exist in file1 but not file2
    for ext in header1.keys():
        print('Extension: %s' % ext)
        for hdrkey in header1[ext].keys():
            if not hdrkey in header2[ext].keys():
                print('\t key/value not present in file2: \t %s/%s' % (hdrkey, header1[ext][hdrkey]))


if __name__ == "__main__":
    indir = 'kwdcheck_tmpdir/'
    workingfile = os.path.join(indir, 'example_fakebinary_NIRISS.oifits')
    notworkingfile = os.path.join(indir, 'ov7_ABDOR_NIRISS_jwst_g7s6c_F480M_59670.oifits')

    check_oifits_headers(workingfile, notworkingfile)

    # this currently does not work
    #check_oifits_headers(notworkingfile, workingfile)