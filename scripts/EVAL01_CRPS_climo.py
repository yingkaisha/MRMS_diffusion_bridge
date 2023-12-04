'''
Compute the CRPS climatology reference of 2023 MRMS from ERA5 CDFs
'''

import os
import sys
import time
import numpy as np
from glob import glob
from datetime import datetime, timedelta

import h5py

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du
import verif_utils as vu

with h5py.File(save_dir+'CNN_domain.hdf', 'r') as h5io:
    lon_01 = h5io['lon_01'][...]
    lat_01 = h5io['lat_01'][...]
    lon_GFS = h5io['lon_GFS'][...]
    lat_GFS = h5io['lat_GFS'][...]

name_output = '/glade/campaign/cisl/aiml/ksha/LDM_results/LDM_2023_ini00_lead{:02d}.hdf'

with h5py.File(save_dir+'ERA5_CDFs.hdf', 'r') as h5io:
    ERA5_CDFs = h5io['ERA5_CDFs'][...]
    indx_in_GFS = h5io['indx_in_GFS'][...]
    indy_in_GFS = h5io['indy_in_GFS'][...]
    
ERA5_CDFs = 3*ERA5_CDFs # mm/hour to mm/3-hour

with h5py.File(name_output.format(3)) as h5io:
    MRMS_TRUE = h5io['MRMS_TRUE'][...]
    MRMS_PRED = h5io['MRMS_PRED'][...]

q_bins = np.arange(0, 1, 0.01)
N_time = len(MRMS_TRUE)
grid_shape = lon_01.shape # MRMS grid shape

LEADs = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
CRPS_climo = np.empty((N_time, len(LEADs))+grid_shape); CRPS_climo[...] = np.nan

for l, lead in enumerate(LEADs):

    with h5py.File(name_output.format(lead)) as h5io:
        MRMS_TRUE = h5io['MRMS_TRUE'][...]
        MRMS_PRED = h5io['MRMS_PRED'][...]
    
    for ix in range(grid_shape[0]):
        for iy in range(grid_shape[1]):
            MRMS_TRUE_ = MRMS_TRUE[:, ix, iy][..., None]
            CDFs = ERA5_CDFs[:, indx_in_GFS[ix, iy], indy_in_GFS[ix, iy]][..., None]
            CRPS_climo[:, l, ix, iy] = vu.CRPS_1d_from_quantiles(q_bins, CDFs, MRMS_TRUE_)[:, 0] # size=(100, 1)
            
    # backup every lead time  
    tuple_save = (CRPS_climo,)
    label_save = ['CRPS_climo',]
    du.save_hdf5(tuple_save, label_save, result_dir, 'CRPS_MRMS_climo_2023.hdf')

