#! /usr/bin/env python

# cd .../ImplaneIA;  python setup.py develop

import sys
import os.path
import nrm_analysis
import nrm_analysis.misctools as misc
import nrm_analysis.misctools.implane2oifits as im2oi
print(nrm_analysis.__file__)
print(nrm_analysis.misctools.implane2oifits.__file__)
print(misc.__file__)
print(im2oi.__file__)


sys.exit("Arretez!")

toifn = os.path.expanduser('~') + '/data/implaneia/example_binary_newPsf/' + 'ov1_t_binary__NIRISS_g7s6_F380M_59112.oifits'
coifn = os.path.expanduser('~') + '/data/implaneia/example_binary_newPsf/' + 'ov1_c_binary__NIRISS_g7s6_F380M_59112.oifits'

tgt_cald = im2oi.calibrate_oifits(toifn, coifn, oifprefix="impc_")

