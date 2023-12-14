import os
import sys
import time
import h5py
import numpy as np
from glob import glob
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du
import verif_utils as vu

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lead_day', help='lead_day')
args = vars(parser.parse_args())

lead_day = int(args['lead_day'])
LEADs = np.arange(24*lead_day, 24*(lead_day+1)+6, 6)

with h5py.File(save_dir+'CNN_domain.hdf', 'r') as h5io:
    lon_01 = h5io['lon_01'][...]
    lat_01 = h5io['lat_01'][...]

year = 2023
N_days = 300


q_bins_blend = np.arange(0.01, 1, 0.01)
N_bins = len(q_bins_blend)

grid_shape = lon_01.shape

name_NB = '/glade/campaign/cisl/aiml/ksha/NBlend_save/NB_2023_ini00_f{:02d}.hdf'
name_MRMS = '/glade/campaign/cisl/aiml/ksha/GFS/MRMS_y{}.hdf'.format(year)

with h5py.File(name_MRMS, 'r') as h5io:
    MRMS = h5io['MRMS'][...]
N_total = len(MRMS)

MRMS_lead = np.empty((N_days, grid_shape[0], grid_shape[1]))
CRPS_NB_lead = np.empty((N_days, grid_shape[0], grid_shape[1]))

for lead in LEADs:
    name_check = '/glade/work/ksha/GAN_result/NB_CRPS_{}_ini00_f{:02d}.hdf'.format(year, lead)

    if os.path.isfile(name_check) is False:
    
        MRMS_lead[...] = np.nan
        CRPS_NB_lead[...] = np.nan
        
        with h5py.File(name_NB.format(lead), 'r') as h5io:
            APCP_q = h5io['APCP_q'][...]
        
        APCP_CDF = np.cumsum(APCP_q*0.01, axis=1)
        APCP_CDF_flat = APCP_CDF.reshape(N_days, N_bins, grid_shape[0]*grid_shape[1])
        APCP_CDF_flat = APCP_CDF_flat/2
        
        for d in range(N_days):
            h1 = d*24 + lead
            h0 = h1-6
            MRMS_lead[d, ...] = np.sum(MRMS[h0:h1, ...], axis=0)
        
        MRMS_lead_flat = MRMS_lead.reshape(N_days, grid_shape[0]*grid_shape[1])
        MRMS_lead_flat = MRMS_lead_flat/2
        
        for d in range(N_days):
            if np.sum(np.isnan(MRMS_lead_flat[d, :])) == 0:
                crps_ = vu.CRPS_1d_from_quantiles(q_bins_blend, APCP_CDF_flat[d, ...], MRMS_lead_flat[d, :][None, ...])
                CRPS_NB_lead[d, ...] = crps_[0, :].reshape(grid_shape[0], grid_shape[1])
            else:
                CRPS_NB_lead[d, ...] = np.nan
        
        # save
        tuple_save = (CRPS_NB_lead,)
        label_save = ['CRPS_NB',]
        du.save_hdf5(tuple_save, label_save, '/glade/work/ksha/GAN_result/', 'NB_CRPS_{}_ini00_f{:02d}.hdf'.format(year, lead))


