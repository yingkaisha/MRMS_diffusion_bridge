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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('part', help='part')
args = vars(parser.parse_args())

# ---------------------------------------------------- #

part = int(args['part'])

if part == 0:
    LEADs = [3, 6, 9, 12, 15, 18, 21, 24]
elif part == 1:
    LEADs = [27, 30, 33, 36, 39, 42, 45, 48]
elif part == 2:
    LEADs = [51, 54, 57, 60, 63, 66, 69, 72]
elif part == 3:
    LEADs = [75, 78, 81, 84, 87, 90, 93, 96]
elif part == 4:
    LEADs = [99, 102, 105, 108, 111, 114, 117, 120]
elif part == 5:
    LEADs = [123, 126, 129, 132, 135, 138, 141, 144]
elif part == 6:
    LEADs = [147, 150, 153, 156, 159, 162, 165, 168]

year = 2023
N_days = 300


with h5py.File(save_dir+'CNN_domain.hdf', 'r') as h5io:
    lon_GFS = h5io['lon_GFS'][...]
    lat_GFS = h5io['lat_GFS'][...]

name_output = '/glade/campaign/cisl/aiml/ksha/LDM_results/LDM_2023_ini00_lead{:02d}.hdf'

with h5py.File(save_dir+'ERA5_CDFs.hdf', 'r') as h5io:
    ERA5_CDFs = h5io['ERA5_CDFs'][...]
    indx_in_GFS = h5io['indx_in_GFS'][...]
    indy_in_GFS = h5io['indy_in_GFS'][...]
    
ERA5_CDFs = 3*ERA5_CDFs # mm/hour to mm/6-hour

name_MRMS = '/glade/campaign/cisl/aiml/ksha/GFS/MRMS_01H_y{}_025.hdf'.format(year)

with h5py.File(name_MRMS, 'r') as h5io:
    MRMS = h5io['MRMS'][...]
N_total = len(MRMS)

q_bins = np.arange(0, 1, 0.01)
N_bins = len(q_bins)
grid_shape = lon_GFS.shape # MRMS grid shape

MRMS_lead = np.empty((N_days, grid_shape[0], grid_shape[1]))
CRPS_climo_lead = np.empty((N_days,)+grid_shape); CRPS_climo_lead[...] = np.nan

for l, lead in enumerate(LEADs):

    name_check = '/glade/work/ksha/GAN_result/CRPS_MRMS_3H_lead{:02d}_Climo_2023.hdf'.format(lead)
    
    if os.path.isfile(name_check) is False:
        # re-fill NaN for a new lead time
        CRPS_climo_lead[...] = np.nan
        
        # collect MRMS on the forecasted time
        MRMS_lead[...] = np.nan

        # hourly to 3-hourly
        for d in range(N_days):
            h1 = d*24 + lead
            h0 = h1-3
            MRMS_lead[d, ...] = np.sum(MRMS[h0:h1, ...], axis=0)

        # compute CRPS on each grid cell
        for ix in range(grid_shape[0]):
            for iy in range(grid_shape[1]):
                MRMS_TRUE_ = MRMS_lead[:, ix, iy][..., None]
                CDFs = ERA5_CDFs[:, ix, iy][..., None]
                CRPS_climo_lead[:, ix, iy] = vu.CRPS_1d_from_quantiles(q_bins, CDFs, MRMS_TRUE_)[:, 0]
                
        CRPS_climo_lead[np.isnan(MRMS_lead)] = np.nan
        
        # backup every lead time  
        tuple_save = (CRPS_climo_lead,)
        label_save = ['CRPS_climo_lead',]
        du.save_hdf5(tuple_save, label_save, result_dir, 'CRPS_MRMS_3H_lead{:02d}_Climo_2023.hdf'.format(lead))
        