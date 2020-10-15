#! /usr/bin/env python
import sys
import os
import pickle
print(os.getcwd())
pfd = open(sys.argv[1],'rb')
dict = pickle.load(pfd)

for kk in dict.keys():
    print('{0:12s}'.format(kk), ':', dict[kk])
