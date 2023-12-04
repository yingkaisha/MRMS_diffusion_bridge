'''
Compute the CRPS of MRMS 2023 from LDM outputs.
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

N_time = 300
grid_shape = lon_01.shape # MRMS grid shape
LEADs = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
CRPS_PRED = np.empty((N_time, len(LEADs))+grid_shape); CRPS_PRED[...] = np.nan

for l, lead in enumerate(LEADs):

    with h5py.File(name_output.format(lead)) as h5io:
        MRMS_TRUE = h5io['MRMS_TRUE'][...]
        MRMS_PRED = h5io['MRMS_PRED'][...]

    MRMS_TRUE_flat = MRMS_TRUE.reshape(300, 256*576)
    MRMS_PRED_flat = MRMS_PRED.reshape(300, 10, 256*576)

    # handle NaNs
    for i in range(300):
        for k in range(256*576):
            if (np.sum(np.isnan(MRMS_PRED_flat[i, :, k])) > 0) and (np.isnan(MRMS_TRUE_flat[i, k]) is False):
                MRMS_PRED_flat[i, :, k] = 0

    CRPS_pred, _, _ = vu.CRPS_1d_nan(MRMS_TRUE_flat, MRMS_PRED_flat)
    CRPS_PRED[:, l, ...] = CRPS_pred.reshape((300, 256, 576))
    
    # backup every lead time  
    tuple_save = (CRPS_PRED,)
    label_save = ['CRPS_PRED',]
    du.save_hdf5(tuple_save, label_save, result_dir, 'CRPS_MRMS_LDM_2023.hdf')
    