#! /usr/bin/env python
import pickle
pfd = open("info4oif_dict.pkl",'rb')
dict = pickle.load(pfd)

for kk in dict.keys():
    print('{0:12s}'.format(kk), ':', dict[kk])
