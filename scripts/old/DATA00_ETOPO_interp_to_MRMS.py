
import sys
import time
import h5py
import pygrib
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du

from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import scipy.interpolate as spint
from scipy.spatial import Delaunay
import itertools

def interp_weights(xy, uv, d=2):
    tri = Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

with h5py.File(save_dir+'MRMS_ERA5_domain.hdf', 'r') as h5io:
    lon_ERA5 = h5io['lon_ERA5'][...]
    lat_ERA5 = h5io['lat_ERA5'][...]
    lon_MRMS = h5io['lon_MRMS'][...]
    lat_MRMS = h5io['lat_MRMS'][...]
    MRMS_to_ERA5_indx = h5io['MRMS_to_ERA5_indx'][...] 
    MRMS_to_ERA5_indy = h5io['MRMS_to_ERA5_indy'][...]
    land_mask_MRMS = h5io['land_mask_MRMS'][...]

with nc.Dataset('/glade/campaign/cisl/aiml/ksha/BACKUP/ETOPO1_Ice_g_gmt4.grd', 'r') as ncio:
    x = ncio['x'][2500:7400]
    y = ncio['y'][6500:9000]
    z = ncio['z'][6500:9000, 2500:7400]

long, lat = np.meshgrid(x, y)
long = np.array(long)
lat = np.array(lat)

#Computed once and for all !
vtx, wts = interp_weights(np.vstack([long.ravel(), lat.ravel()]).T, np.vstack([lon_MRMS.ravel(), lat_MRMS.ravel()]).T)

start_time = time.time()
elev_MRMS_temp = interpolate(z.ravel(), vtx, wts)
elev_MRMS_temp = elev_MRMS_temp.reshape(lon_MRMS.shape)
print("--- %s seconds ---" % (time.time() - start_time))

elev_MRMS = np.copy(elev_MRMS_temp)

tuple_save = (lon_ERA5, lat_ERA5, lon_MRMS, lat_MRMS, MRMS_to_ERA5_indx, MRMS_to_ERA5_indy, land_mask_MRMS, elev_MRMS)
label_save = ['lon_ERA5', 'lat_ERA5', 'lon_MRMS', 'lat_MRMS', 'MRMS_to_ERA5_indx', 'MRMS_to_ERA5_indy', 'land_mask_MRMS', 'elev_MRMS']

du.save_hdf5(tuple_save, label_save, save_dir, 'MRMS_ERA5_domain.hdf')

