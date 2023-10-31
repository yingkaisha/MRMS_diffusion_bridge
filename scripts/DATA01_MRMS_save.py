
import os
import sys
import time
import h5py
import pygrib
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

def spatial_agregate(data, size=10):
    Nx, Ny = data.shape
    Nxa = Nx // 10; Nya = Ny // 10
    out = np.empty((Nxa, Nya))
    
    for i in range(Nxa):
        for j in range(Nya):
            out[i, j] = np.mean(data[i*size:(i*size+size), j*size:(j*size+size)])
    return out

with h5py.File(save_dir+'CNN_domain.hdf', 'r') as h5io:
    lon_01 = h5io['lon_01'][...]
    lat_01 = h5io['lat_01'][...]

year = 2021
MRMS_dir = '/glade/campaign/cisl/aiml/ksha/MRMS_{}/'.format(year)

base = datetime(year, 1, 1)
date_list = [base + timedelta(hours=h) for h in range(365*24)]

shape_grid = lon_01.shape
MRMS_save = np.empty((len(date_list),)+shape_grid); MRMS_save[...] = np.nan

for i_dt, dt in enumerate(date_list):
    
    dt_str = datetime.strftime(dt, '%Y%m%d-%H%M%S')
    filename = MRMS_dir+'MRMS_MultiSensor_QPE_01H_Pass2_00.00_{}.grib2'.format(dt_str)
    try:
        with pygrib.open(filename) as grbio:
            MRMS = grbio[1].values
        MRMS = np.flipud(MRMS)
        MRMS_save[i_dt, ...] = spatial_agregate(MRMS, size=10)[47:303, 53:629]
    except:
        print('{} not exist'.format(filename))

tuple_save = (MRMS_save,)
label_save = ['MRMS',]
du.save_hdf5(tuple_save, label_save, '/glade/campaign/cisl/aiml/ksha/GFS/', 'MRMS_y{}.hdf'.format(year))



