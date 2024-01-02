
import os
import sys
import time
import h5py
import pygrib
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('year', help='part')
args = vars(parser.parse_args())

# ---------------------------------------------------- #

year = int(args['year'])

if year == 2020:
    N_days = 366
else:
    N_days = 365
    
with h5py.File(save_dir+'CNN_domain.hdf', 'r') as h5io:
    lon_GFS = h5io['lon_GFS'][...]
    lat_GFS = h5io['lat_GFS'][...]

with h5py.File(save_dir+'MRMS_domain.hdf', 'r') as h5io:
    lon_MRMS = h5io['lon_MRMS'][...]
    lat_MRMS = h5io['lat_MRMS'][...]

MRMS_dir = '/glade/campaign/cisl/aiml/ksha/MRMS_{}/'.format(year)

base = datetime(year, 1, 1)
date_list = [base + timedelta(hours=h) for h in range(N_days*24)]

shape_grid = lon_GFS.shape
MRMS_save = np.empty((len(date_list),)+shape_grid); MRMS_save[...] = np.nan

for i_dt, dt in enumerate(date_list):
    
    dt_str = datetime.strftime(dt, '%Y%m%d-%H%M%S')
    filename = MRMS_dir+'MRMS_MultiSensor_QPE_01H_Pass2_00.00_{}.grib2'.format(dt_str)
    try:
        with pygrib.open(filename) as grbio:
            MRMS = grbio[1].values
        MRMS = np.flipud(MRMS)

        hr_to_lr = RegularGridInterpolator((lat_MRMS[:, 0], lon_MRMS[0, :]), MRMS, 
                                            bounds_error=False, fill_value=None)
        MRMS_save[i_dt, ...] = hr_to_lr((lat_GFS, lon_GFS))
    
    except:
        print('{} not exist'.format(filename))
        
tuple_save = (MRMS_save,)
label_save = ['MRMS',]
du.save_hdf5(tuple_save, label_save, '/glade/campaign/cisl/aiml/ksha/GFS/', 'MRMS_01H_y{}_025.hdf'.format(year))

