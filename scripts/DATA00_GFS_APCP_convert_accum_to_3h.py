
import os
import sys
import time
import h5py
import numpy as np

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

year = 2021
LEADs = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
INIs = [0, 6, 12, 18]

name_gfs = '/glade/campaign/cisl/aiml/ksha/GFS/GFS_{}_ini{:02d}_f{:02d}.hdf'
name_save = 'GFS_APCP_{}_ini{:02d}_f{:02d}.hdf'

for ini in INIs:
    with h5py.File(name_gfs.format(year, ini, 3), 'r') as h5io:
        APCP1 = h5io['APCP'][...]
        
    tuple_save = (APCP1,)
    label_save = ['APCP',]
    du.save_hdf5(tuple_save, label_save, '/glade/campaign/cisl/aiml/ksha/GFS/', name_save.format(year, ini, 3))

for ini in INIs:
    for lead in LEADs:
        with h5py.File(name_gfs.format(year, ini, lead), 'r') as h5io:
            APCP1 = h5io['APCP'][...]
            
        with h5py.File(name_gfs.format(year, ini, lead-3), 'r') as h5io:
            APCP0 = h5io['APCP'][...]

        tuple_save = (APCP1-APCP0,)
        label_save = ['APCP',]
        du.save_hdf5(tuple_save, label_save, '/glade/campaign/cisl/aiml/ksha/GFS/', name_save.format(year, ini, lead))
