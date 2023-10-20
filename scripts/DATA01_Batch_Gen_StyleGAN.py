
import os
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

# ==================== #
size_mrms = 256
size_half = 128
output_size = 8
rain_thres = 1.0
rain_cover_thres = 0.2
# ==================== #

with h5py.File(save_dir+'MRMS_ERA5_domain.hdf', 'r') as h5io:
    lon_ERA5 = h5io['lon_ERA5'][...]
    lat_ERA5 = h5io['lat_ERA5'][...]
    lon_MRMS = h5io['lon_MRMS'][...]
    lat_MRMS = h5io['lat_MRMS'][...]
    indx = h5io['MRMR_to_ERA5_indx'][...]
    indy = h5io['MRMR_to_ERA5_indy'][...]

base = datetime(2021, 1, 1)
date_list = [base + timedelta(hours=day) for day in range(365*24)]

def rain_context(data, bin_width, output_size, thres):
    output = np.zeros(output_size)

    for i in range(output_size[0]):
        for j in range(output_size[1]):

            i0 = i*bin_width; i1 = i0 + bin_width
            j0 = j*bin_width; j1 = j0 + bin_width
            patch_mean = np.mean(data[i0:i1, j0:j1])
            output[i, j] = patch_mean # >= thres
            
    return output

def norm_precip(x):
    return np.log(x+1)

def norm_wind(x):
    return x/5

def norm_rh(x):
    return x/100 - 0.5
    
def norm_gph(x):
    # 600 hPa to 1000 hPa
    return x/40000 - 0.5

def norm_t(x):
    return (x-273.15)/10


x_mrms = 3500
y_mrms = 7000

gap = size_half

x_ERA5 = 361
y_ERA5 = 721
L = 12

rain_thres = 1.0
rain_cover_thres = 0.2

profile_t = np.zeros(L)
profile_u = np.zeros(L)
profile_v = np.zeros(L)
profile_rh = np.zeros(L)
profile_gph = np.zeros(L)

nc_name_t = ERA5_dir+'ERA5_t2m_2021_bottom_layers.nc'
nc_name_u = ERA5_dir+'ERA5_u_2021_bottom_layers.nc'
nc_name_v = ERA5_dir+'ERA5_v_2021_bottom_layers.nc'
nc_name_rh = ERA5_dir+'ERA5_rh_2021_bottom_layers.nc'
nc_name_gph = ERA5_dir+'ERA5_gph_2021_bottom_layers.nc'

batch_file_name = 'DSCALE_dt{:04d}_i{:04d}_j{:04d}.npy'

with nc.Dataset(nc_name_t) as ncio_t:
    with nc.Dataset(nc_name_u) as ncio_u:
        with nc.Dataset(nc_name_v) as ncio_v:
            with nc.Dataset(nc_name_rh) as ncio_rh:
                with nc.Dataset(nc_name_gph) as ncio_gph:
                    
                    # Main section
                    for i_dt, dt in enumerate(date_list[:1]):
                        dt_str = datetime.strftime(dt, '%Y%m%d-%H%M%S')
                        name_ = 'MRMS_MultiSensor_QPE_01H_Pass2_00.00_{}.grib2'.format(dt_str)
                        filename = MRMS_dir + name_
                    
                        if os.path.isfile(filename):
                            with pygrib.open(filename) as grbio:
                                MRMS = grbio[1].values
                                
                            # flip --> lower latitude at the bottom
                            MRMS = np.flipud(MRMS)
                    
                            for i_mrms in range(0, x_mrms-size_mrms+gap, gap):
                                for j_mrms in range(0, y_mrms-size_mrms+gap, gap):
                                    i0 = i_mrms; i1 = i0 + size_mrms
                                    j0 = j_mrms; j1 = j0 + size_mrms
                                    patch_ = MRMS[i0:i1, j0:j1]
                                    patch_flag = patch_ > rain_thres
                                    patch_rain_cover_rate = np.sum(patch_flag)/(size_mrms*size_mrms)
                                    
                                    # We select this patch
                                    if (patch_.shape == (size_mrms, size_mrms)) and (patch_rain_cover_rate > rain_cover_thres):
                                        patch = patch_
                                        ix_era5 = indx[i0:i1, j0:j1]
                                        iy_era5 = indy[i0:i1, j0:j1]
                                        context = rain_context(patch, bin_width=32, output_size=(output_size, output_size), thres=rain_thres)

                                        # ========== Averaged profiles for rain grid cells ========== #
                                        ix_era5_rain_ = ix_era5[patch_flag]
                                        iy_era5_rain_ = iy_era5[patch_flag]
                                        
                                        flag_uniqe_ind = is_unique(ix_era5_rain_, iy_era5_rain_)
                                        N_points = np.sum(flag_uniqe_ind)
                                        ix_era5_rain = ix_era5_rain_[flag_uniqe_ind]
                                        iy_era5_rain = iy_era5_rain_[flag_uniqe_ind]
                                        
                                        # Get the mean ERA5 profiles
                                        profile_t[...] = 0.0
                                        profile_u[...] = 0.0
                                        profile_v[...] = 0.0
                                        profile_rh[...] = 0.0
                                        profile_gph[...] = 0.0

                                        for i_point in range(N_points):
                                            ix_era5_rain_temp = ix_era5_rain[i_point]
                                            iy_era5_rain_temp = iy_era5_rain[i_point]

                                            # ERA5 levels: top to bottom (600 --> 1000 hPa)
                                            profile_t += ncio_t['t'][i_dt, :, ix_era5_rain_temp, iy_era5_rain_temp]
                                            profile_u += ncio_u['u'][i_dt, :, ix_era5_rain_temp, iy_era5_rain_temp]
                                            profile_v += ncio_v['v'][i_dt, :, ix_era5_rain_temp, iy_era5_rain_temp]
                                            profile_rh += ncio_rh['r'][i_dt, :, ix_era5_rain_temp, iy_era5_rain_temp]
                                            profile_gph += ncio_gph['z'][i_dt, :, ix_era5_rain_temp, iy_era5_rain_temp]

                                        profile_t = norm_t(profile_t/N_points)
                                        profile_u = norm_wind(profile_u/N_points)
                                        profile_v = norm_wind(profile_v/N_points)
                                        profile_rh = norm_rh(profile_rh/N_points)
                                        profile_gph = norm_gph(profile_gph/N_points)
                                            
                                        # ========== Profiles for the max rain grid ========== #

                                        ind_max_ = np.argmax(patch)
                                        ind_max_x, ind_max_y = np.unravel_index(ind_max_, (size_mrms, size_mrms))
                                        
                                        profile_tmax = ncio_t['t'][i_dt, :, ind_max_x, ind_max_y]
                                        profile_umax = ncio_u['u'][i_dt, :, ind_max_x, ind_max_y]
                                        profile_vmax = ncio_v['v'][i_dt, :, ind_max_x, ind_max_y]
                                        profile_rhmax = ncio_rh['r'][i_dt, :, ind_max_x, ind_max_y]
                                        profile_gphmax = ncio_gph['z'][i_dt, :, ind_max_x, ind_max_y]
                                        
                                        profile_tmax = norm_t(profile_tmax)
                                        profile_umax = norm_wind(profile_umax)
                                        profile_vmax = norm_wind(profile_vmax)
                                        profile_rhmax = norm_rh(profile_rhmax)
                                        profile_gphmax = norm_gph(profile_gphmax)
                                        
                                        # ========== Precip normalization and save ======== #
                                        
                                        patch = norm_precip(patch)
                                        context = norm_precip(context)

                                        save_dict = {}
                                        save_dict['patch'] = patch
                                        save_dict['context'] = context
                                        save_dict['profile_t'] = profile_t
                                        save_dict['profile_u'] = profile_u
                                        save_dict['profile_v'] = profile_v
                                        save_dict['profile_rh'] = profile_rh
                                        save_dict['profile_gph'] = profile_gph

                                        save_dict['profile_tmax'] = profile_tmax
                                        save_dict['profile_umax'] = profile_umax
                                        save_dict['profile_vmax'] = profile_vmax
                                        save_dict['profile_rhmax'] = profile_rhmax
                                        save_dict['profile_gphmax'] = profile_gphmax
                                        
                                        name_ = BATCH_dir+batch_file_name.format(i_dt, i_mrms, j_mrms)
                                        print(name_)
                                        np.save(name_, save_dict)


