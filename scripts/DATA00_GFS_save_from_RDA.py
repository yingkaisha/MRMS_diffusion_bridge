
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

with h5py.File(save_dir+'CNN_domain.hdf', 'r') as h5io:
    lon_GFS = h5io['lon_GFS'][...]
    lat_GFS = h5io['lat_GFS'][...]


year = 2022
base = datetime(year, 1, 1)
date_list = [base + timedelta(days=h) for h in range(365)]

shape_grid = lon_GFS.shape


# T800 U800 V800, RH800, APCP, CAPE, PWAT
var_inds1 = [315, 320, 321, 316, 451, 474, 476,]
var_inds2 = [446, 452, 453, 447, 597, 624, 626,]

var_names = [':Temperature:K (instant):regular_ll:isobaricInhPa:level 80000 Pa:fcst time',
             ':U component of wind:m s**-1 (instant):regular_ll:isobaricInhPa:level 8000',
             ':V component of wind:m s**-1 (instant):regular_ll:isobaricInhPa:level 8000',
             ':Relative humidity:% (instant):regular_ll:isobaricInhPa:level 80000 Pa:fcs',
             ':Total Precipitation:kg m**-2 (accum):regular_ll:surface:level 0:fcst time',
             ':Convective available potential energy:J kg**-1 (instant):regular_ll:surfa',
             ':Precipitable water:kg m**-2 (instant):regular_ll:atmosphereSingleLayer:le']

LEADs  =[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
#INIs = ['00', '06', '12', '18']
INIs = ['18',]

for str_ini in INIs:
    for lead in LEADs:

        APCP_GFS = np.empty((len(date_list),)+shape_grid); APCP_GFS[...] = np.nan
        CAPE_GFS = np.empty((len(date_list),)+shape_grid); CAPE_GFS[...] = np.nan
        PWAT_GFS = np.empty((len(date_list),)+shape_grid); PWAT_GFS[...] = np.nan
        T800_GFS = np.empty((len(date_list),)+shape_grid); T800_GFS[...] = np.nan
        U800_GFS = np.empty((len(date_list),)+shape_grid); U800_GFS[...] = np.nan
        V800_GFS = np.empty((len(date_list),)+shape_grid); V800_GFS[...] = np.nan
        RH800_GFS = np.empty((len(date_list),)+shape_grid); RH800_GFS[...] = np.nan

        for i_dt, dt in enumerate(date_list):
        
            filename_gfs = datetime.strftime(dt, 
                '/glade/campaign/collections/rda/data/ds084.1/%Y/%Y%m%d/gfs.0p25.%Y%m%d{}.f{:03d}.grib2'.format(str_ini, lead))
    
            # ========== check grb ids ========== #
            names1 = []
            names2 = []
            
            with pygrib.open(filename_gfs) as grbio:
                for i in var_inds1:
                    names1.append(str(grbio[i])[3:77])
                for i in var_inds2:
                    try:
                        names2.append(str(grbio[i])[3:77])
                    except:
                        print('list too long')
            
            if names1 == var_names:
                var_inds = var_inds1
            elif names2 == var_names:
                var_inds = var_inds2
            else:
                print(filename_gfs)
                print(names1)
                print(names2)
                print('error')
                break;
    
            with pygrib.open(filename_gfs) as grbio:
                T = grbio[var_inds[0]].values #
                U = grbio[var_inds[1]].values
                V = grbio[var_inds[2]].values
                RH = grbio[var_inds[3]].values
                APCP = grbio[var_inds[4]].values
                CAPE = grbio[var_inds[5]].values
                PWAT = grbio[var_inds[6]].values
            
            T_NA = T[:-360, 720:]
            T_NA = np.flipud(T_NA)[98:203, 218:460]
            
            U_NA = U[:-360, 720:]
            U_NA = np.flipud(U_NA)[98:203, 218:460]
            
            V_NA = V[:-360, 720:]
            V_NA = np.flipud(V_NA)[98:203, 218:460]
            
            RH_NA = RH[:-360, 720:]
            RH_NA = np.flipud(RH_NA)[98:203, 218:460]
            
            APCP_NA = APCP[:-360, 720:]
            APCP_NA = np.flipud(APCP_NA)[98:203, 218:460]
            
            CAPE_NA = CAPE[:-360, 720:]
            CAPE_NA = np.flipud(CAPE_NA)[98:203, 218:460]
            
            PWAT_NA = PWAT[:-360, 720:]
            PWAT_NA = np.flipud(PWAT_NA)[98:203, 218:460]
            
            APCP_GFS[i_dt, ...] = APCP_NA
            CAPE_GFS[i_dt, ...] = CAPE_NA
            PWAT_GFS[i_dt, ...] = PWAT_NA
            T800_GFS[i_dt, ...] = T_NA
            U800_GFS[i_dt, ...] = U_NA
            V800_GFS[i_dt, ...] = V_NA
            RH800_GFS[i_dt, ...] = RH_NA
        
        tuple_save = (APCP_GFS, CAPE_GFS, PWAT_GFS, T800_GFS, U800_GFS, V800_GFS, RH800_GFS)
        label_save = ['APCP', 'CAPE', 'PWAT', 'T800', 'U800', 'V800', 'RH800']
    
        du.save_hdf5(tuple_save, label_save, '/glade/campaign/cisl/aiml/ksha/GFS/', 'GFS_{}_ini{}_f{:02d}.hdf'.format(year, str_ini, lead))
