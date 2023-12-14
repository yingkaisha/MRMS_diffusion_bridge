
import os
import sys
import time
import h5py
import pygrib
import numpy as np
from datetime import datetime, timedelta

import scipy.interpolate as spint
from scipy.spatial import Delaunay
import itertools

sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/')
sys.path.insert(0, '/glade/u/home/ksha/GAN_proj/libs/')

from namelist import *
import data_utils as du
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
parser.add_argument('lead_input', help='lead_input')
args = vars(parser.parse_args())

# ---------------------------------------------------- #

year = int(args['year'])
lead_input = int(args['lead_input'])

LEADs = [lead_input,]

N_days = 300
base = datetime(year, 1, 1)
date_list = [base + timedelta(days=day) for day in range(N_days)]

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
    
# MRMS lat/lon
with h5py.File(save_dir+'CNN_domain.hdf', 'r') as h5io:
    lon_01 = h5io['lon_01'][...]
    lat_01 = h5io['lat_01'][...]

filename = '/glade/campaign/cisl/aiml/ksha/NBlend_2023/blend_20230101_t00z.qmd.f006.co.grib2'
with pygrib.open(filename) as grbio:
    lat_blend, lon_blend = apcp_temp = grbio[2].latlons()

# compute interpolation weights from National blend grids to 0.1 deg grids
vtx, wts = interp_weights(np.vstack([lon_blend.ravel(), lat_blend.ravel()]).T, 
                          np.vstack([lon_01.ravel(), lat_01.ravel()]).T)

q_bins_blend = np.arange(0.01, 1, 0.01)
grid_shape = lat_01.shape
N_bins = len(q_bins_blend)

APCP_q = np.empty((N_days, N_bins,)+grid_shape)

for lead in LEADs:
    APCP_q[...] = np.nan
    
    name_check = '/glade/campaign/cisl/aiml/ksha/NBlend_save/NB_{}_ini00_f{:02d}.hdf'.format(year, lead)
    
    if os.path.isfile(name_check) is False:
        
        for t, dt in enumerate(date_list):
            
            dt_str = datetime.strftime(dt, '%Y%m%d')
            name_ = '/glade/campaign/cisl/aiml/ksha/NBlend_2023/blend_{}_t00z.qmd.f{:03d}.co.grib2'.format(dt_str, lead)
            
            try:
                # ------- found the starting grib index for National Blend quantile values ----- # 
                
                i_start = 1 # initialize i_start
                with pygrib.open(name_) as grbio:
                    var_list = grbio[1:]
                
                for i, var_name in enumerate(var_list):
                    if 'Probability' in str(var_name):
                        continue;
                    else:
                        i_start = i # the first entry without "Probability" keyword is quantile 0.01
                        break;
                # ------------------------------------------------------------------------------ #        
                with pygrib.open(name_) as grbio:
                    for i, ind in enumerate(range(i_start, i_start+99, 1)):
                        apcp_temp = grbio[ind].values
                        apcp_interp = interpolate(apcp_temp.ravel(), vtx, wts)
                        apcp_interp[apcp_interp<0] = 0
                        APCP_q[t, i, ...] = apcp_interp.reshape(grid_shape)
            except:
                print('missing: {}'.format(name_))
                        
        tuple_save = (APCP_q,)
        label_save = ['APCP_q',]
        du.save_hdf5(tuple_save, label_save, '/glade/campaign/cisl/aiml/ksha/NBlend_save/', 'NB_{}_ini00_f{:02d}.hdf'.format(year, lead))


