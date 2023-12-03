import os
import sys
import time
import numpy as np
from glob import glob
from datetime import datetime, timedelta

import h5py
import netCDF4 as nc

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

with h5py.File(save_dir+'CNN_domain.hdf', 'r') as h5io:
    lon_01 = h5io['lon_01'][...]
    lat_01 = h5io['lat_01'][...]
    lon_GFS = h5io['lon_GFS'][...]
    lat_GFS = h5io['lat_GFS'][...]

ERA5_dir = '/glade/campaign/cisl/aiml/ksha/ERA5/ERA5_APCP_2018_2023.nc'
lat_lim = [158, 263]
lon_lim = [218, 459]
grid_shape = lon_GFS.shape

q_bins = np.arange(0, 1, 0.01)
N_bins = len(q_bins)
q_ERA5 = np.load('/glade/campaign/cisl/aiml/ksha/ERA5/ERA5_APCP_quantile.npy')
N_hour = 24*(365+365+366+365+365)

with nc.Dataset(ERA5_dir, 'r') as ncio:
    for i, ix in enumerate(range(lat_lim[0], lat_lim[1]+1, 1)):
        for j, iy in enumerate(range(lon_lim[0], lon_lim[1]+1, 1)):
            if np.isnan(q_ERA5[-1, i, j]):
                ERA5_APCP = ncio['tp'][:, 0, ix, iy]
                ERA5_APCP = np.array(ERA5_APCP)[:N_hour]*1000
                ERA5_APCP[ERA5_APCP<1e-7] = 0.0 # ERA5 zero = 6.93889390e-15
                q_ERA5[:, i, j] = np.quantile(ERA5_APCP, q_bins)
                print('/glade/campaign/cisl/aiml/ksha/ERA5/ERA5_APCP_quantile.npy')
                np.save('/glade/campaign/cisl/aiml/ksha/ERA5/ERA5_APCP_quantile.npy', q_ERA5)
            else:
                continue;
