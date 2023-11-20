import os
import sys
import time
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

def norm_precip(x):
    return np.log(x+1)

def norm_rh(x):
    return (x-50)/28/2

def norm_t(x):
    return (x-281)/10/2

def norm_u(x):
    return (x-3.5)/6.5/2

def norm_v(x):
    return (x)/6.5/2

def norm_pwat(x):
    return (x-20.5)/15/2

def norm_cape(x):
    return (x-200)/450/2

with h5py.File(save_dir+'CNN_domain.hdf', 'r') as h5io:
    #print(h5io.keys())
    lon_01 = h5io['lon_01'][...]
    lat_01 = h5io['lat_01'][...]
    elev_01 = h5io['elev_01'][...]
    lon_GFS = h5io['lon_GFS'][...]
    lat_GFS = h5io['lat_GFS'][...]

elev_01[elev_01<0] = 0
elev_01 = elev_01/2000 - 1

x_mrms = 256
y_mrms = 576
x_gfs = 105
y_gfs = 242

size = 128
gap = 24
grid_shape = (x_mrms, y_mrms)

data = np.empty((1, x_mrms, y_mrms, 10))
gfs = np.empty((1, x_gfs, y_gfs, 8))

# ======================================================== #
year = 2021
base = datetime(year, 1, 1)
date_list = [base + timedelta(days=d) for d in range(365)]
LEADs = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36] #
INIs = [0, 6, 12, 18]
# ======================================================== #

with h5py.File('/glade/campaign/cisl/aiml/ksha/GFS/MRMS_y{}.hdf'.format(year), 'r') as h5io:
    MRMS = h5io['MRMS'][...]

BATCH_dir = '/glade/campaign/cisl/aiml/ksha/BATCH_GFS_MRMS/'
batch_file_name = 'GFS_y{:04d}_ini{:02d}_lead{:02d}_dt{:04d}_ix{:03d}_iy{:03d}.npy'

name_gfs = '/glade/campaign/cisl/aiml/ksha/GFS/GFS_{}_ini{:02d}_f{:02d}.hdf'
name_apcp = '/glade/campaign/cisl/aiml/ksha/GFS/GFS_APCP_{}_ini{:02d}_f{:02d}.hdf'


for lead in LEADs:
    for ini in INIs:

        with h5py.File(name_gfs.format(year, ini, lead), 'r') as h5io:
            CAPE = h5io['CAPE'][...]
            PWAT = h5io['PWAT'][...]
            T800 = h5io['T800'][...]
            U800 = h5io['U800'][...]
            V800 = h5io['V800'][...]
            RH800 = h5io['RH800'][...]
        
        with h5py.File(name_apcp.format(year, ini, lead), 'r') as h5io:
            APCP = h5io['APCP'][...]

        # ======================================================== #
        for i_dt, dt in enumerate(date_list):
        
            N_hours = i_dt*24 + ini + lead
            if N_hours < 8760:
                MRMS_temp = MRMS[N_hours, ...] + MRMS[N_hours-1, ...] + MRMS[N_hours-2, ...]
                
                hr_to_lr = RegularGridInterpolator((lat_01[:, 0], lon_01[0, :]), MRMS_temp, 
                                                   bounds_error=False, fill_value=None)
                MRMS_lr = hr_to_lr((lat_GFS, lon_GFS))
                
                gfs[..., 0] = MRMS_lr
                gfs[..., 1] = APCP[i_dt, ...]
                gfs[..., 2] = CAPE[i_dt, ...]
                gfs[..., 3] = PWAT[i_dt, ...]
                gfs[..., 4] = T800[i_dt, ...]
                gfs[..., 5] = U800[i_dt, ...]
                gfs[..., 6] = V800[i_dt, ...]
                gfs[..., 7] = RH800[i_dt, ...]
                
                data[..., 0] = MRMS_temp
                
                for i in range(8):
                    lr_to_hr = RegularGridInterpolator((lat_GFS[:, 0], lon_GFS[0, :]), gfs[0, ..., i], 
                                                       bounds_error=False, fill_value=None)
                    data[..., i+1] = lr_to_hr((lat_01, lon_01))
                
                temp = data[..., 1]
                temp[temp < 0] = 0
                data[..., 1] = temp
                
                data[..., 0] = norm_precip(data[..., 0])
                data[..., 1] = norm_precip(data[..., 1])
                data[..., 2] = norm_precip(data[..., 2])
                data[..., 3] = norm_cape(data[..., 3])
                data[..., 4] = norm_pwat(data[..., 4])
                data[..., 5] = norm_t(data[..., 5])
                data[..., 6] = norm_u(data[..., 6])
                data[..., 7] = norm_v(data[..., 7])
                data[..., 8] = norm_rh(data[..., 8])
                data[..., 9] = elev_01

                for ix in range(0, grid_shape[0]+gap, gap):
                    for iy in range(0, grid_shape[1]+gap, gap):
                        ix_start = ix
                        ix_end = ix+size
            
                        iy_start = iy
                        iy_end = iy+size
            
                        if (ix_end < grid_shape[0]) and (iy_end < grid_shape[1]):
                            temp_mrms_flag = data[0, ix_start:ix_end, iy_start:iy_end, 0]
                            if np.sum(temp_mrms_flag > 0.1) > 4000:
                                if np.sum(np.isnan(data)) == 0:
                                    name_ = BATCH_dir+batch_file_name.format(year, ini, lead, i_dt, ix, iy)
                                    print(name_)
                                    np.save(name_, data[:, ix_start:ix_end, iy_start:iy_end, :])


















