'''
GFS APCP is accumulated based on forecast lead times, e.g., f036 APCP is 36-hr accumulated total precip
This script converts GFS APCP to 3-hr accumulated values.
The script works for the pre-processed GFS files of this project
'''

import os
import sys
import time
import h5py
import numpy as np

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

# --------- #
year = 2023
LEADs = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36] # forecast lead time [hr]
INIs = [0, 6, 12, 18] # initialization times [UTC hrs]

name_gfs = '/glade/campaign/cisl/aiml/ksha/GFS/GFS_{}_ini{:02d}_f{:02d}.hdf'
name_save = 'GFS_APCP_{}_ini{:02d}_f{:02d}.hdf'
# --------- #

# keep f003 APCP as it is
for ini in INIs:
    with h5py.File(name_gfs.format(year, ini, 3), 'r') as h5io:
        APCP1 = h5io['APCP'][...]
        
    tuple_save = (APCP1,)
    label_save = ['APCP',]
    du.save_hdf5(tuple_save, label_save, '/glade/campaign/cisl/aiml/ksha/GFS/', name_save.format(year, ini, 3))

# subtract f006 from f003 to get the 04, 05, 06 hr accumulated precip
# applies for all LEADs values

for ini in INIs:
    for lead in LEADs:
        with h5py.File(name_gfs.format(year, ini, lead), 'r') as h5io:
            APCP1 = h5io['APCP'][...]
            
        with h5py.File(name_gfs.format(year, ini, lead-3), 'r') as h5io:
            APCP0 = h5io['APCP'][...]

        data = APCP1-APCP0
        if np.sum(data<0) > 0:
            print("warning: negative precip {} [mm] dectected on ini{} lead{}".format(np.min(data), ini, lead))
            data[data<0] = 0
            
        tuple_save = (data,)
        label_save = ['APCP',]
        du.save_hdf5(tuple_save, label_save, '/glade/campaign/cisl/aiml/ksha/GFS/', name_save.format(year, ini, lead))
