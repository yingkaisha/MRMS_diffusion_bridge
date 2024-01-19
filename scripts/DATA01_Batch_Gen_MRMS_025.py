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

with h5py.File(save_dir+'CNN_domain.hdf', 'r') as h5io:
    lon_GFS = h5io['lon_GFS'][...]
    lat_GFS = h5io['lat_GFS'][...]

year = 2021
BATCH_dir = '/glade/campaign/cisl/aiml/ksha/BATCH_MRMS_025/'

gapx = 7
gapy = 12

N_hour = 6
size_x = 128 # patch size: 128-by-256
size_y = 256

x_mrms = 105; y_mrms = 242 # 0.1 deg MRMS size
grid_shape = (x_mrms, y_mrms) # the size of 0.1 deg MRMS

N_rain_thres = 200 # each patch must have 1600 raining grid cells
V_rain_thres = 0.1 # 0.1 mm/3h means rain

with h5py.File('/glade/campaign/cisl/aiml/ksha/GFS/MRMS_01H_y{}_025.hdf'.format(year), 'r') as h5io:
    MRMS = h5io['MRMS'][...]

L = len(MRMS) - N_hour # number of available time dimensions

batch_name = 'MRMS_y{}_day{:04d}.npy'
mrms_save = np.zeros((size_x, size_y))

for i in range(L):
    mrms = np.zeros(grid_shape)
    for j in range(N_hour):
        mrms += MRMS[i+j]
        
    # if MRMS has no NaNs
    if np.sum(np.isnan(mrms)) == 0:
        mrms_save[...] = 0.0
        mrms_save[gapy:x_mrms+gapy, gapx:y_mrms+gapx] = mrms
        
        # if the patch contains enough raining grid cells
        if np.sum(mrms_save > V_rain_thres) > N_rain_thres:

            # if the patch doesn't have NaNs 
            if np.sum(np.isnan(mrms_save)) == 0:

                save_name = BATCH_dir+batch_name.format(year, i)
                #print(save_name)
                np.save(save_name, mrms_save)







