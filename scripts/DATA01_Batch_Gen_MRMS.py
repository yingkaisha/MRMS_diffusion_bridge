'''
This script generate training and validation batches for VQ-VAE that compress 0.1 deg MRMS
'''

import os
import sys
import time
import h5py
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

def norm_precip(x):
    return np.log(x+1)

# ======================================================== #
year = 2023
N_days = 365 #365
BATCH_dir = '/glade/campaign/cisl/aiml/ksha/BATCH_MRMS_PRISM/'

base = datetime(year, 1, 1)
date_list = [base + timedelta(hours=h) for h in range(N_days*24)]

N_hour = 3
size = 128 # patch size: 128-by-128
gap = 64 # subset patches with gaps of 24 grids
N_rain_thres = 800 # each patch must have 1600 raining grid cells
V_rain_thres = 0.1 # 0.1 mm/3h means rain
batch_name = 'MRMS_y{}_day{:04d}_ix{}_iy{}.npy'
# ======================================================== #

with h5py.File(save_dir+'CNN_domain.hdf', 'r') as h5io:
    lon_01 = h5io['lon_01'][...]
    lat_01 = h5io['lat_01'][...]

# load 0.1 PRISM climatology
with h5py.File(save_dir+'PRISM_Climatology.hdf', 'r') as h5io:
    PRISM_01 = h5io['PRISM_01'][...]
PRISM_01[np.isnan(PRISM_01)] = 0.0

# load MRMS
with h5py.File('/glade/campaign/cisl/aiml/ksha/GFS/MRMS_y{}.hdf'.format(year), 'r') as h5io:
    MRMS = h5io['MRMS'][...]

x_mrms = 256; y_mrms = 576 # 0.1 deg MRMS size
grid_shape = (x_mrms, y_mrms) # the size of 0.1 deg MRMS
L = len(MRMS) - N_hour # number of available time dimensions

data = np.empty((1, size, size, 2))
data[...] = np.nan

for i, dt in enumerate(date_list):
    mon = dt.month
    prism_ = PRISM_01[mon-1, ...]
    
    mrms = np.zeros(grid_shape)
    for j in range(N_hour):
        mrms += MRMS[i+j]
        
    # if MRMS has no NaNs
    if np.sum(np.isnan(mrms)) == 0:
        
        for ix in range(0, grid_shape[0]+gap, gap):
            for iy in range(0, grid_shape[1]+gap, gap):
                
                # index ranges
                ix_start = ix; ix_end = ix+size
                iy_start = iy; iy_end = iy+size

                # if not at the edge
                if (ix_end < grid_shape[0]) and (iy_end < grid_shape[1]):
                    mrms_save = mrms[ix_start:ix_end, iy_start:iy_end]
                    data[0, ..., 0] = norm_precip(mrms_save)
                    data[0, ..., 1] = norm_precip(prism_[ix_start:ix_end, iy_start:iy_end])
                    
                    # if the patch contains enough raining grid cells
                    if np.sum(mrms_save > V_rain_thres) > N_rain_thres:

                        # if the patch doesn't have NaNs 
                        if np.sum(np.isnan(data)) == 0:

                            save_name = BATCH_dir+batch_name.format(year, i, ix, iy)
                            print(save_name)
                            np.save(save_name, data)


