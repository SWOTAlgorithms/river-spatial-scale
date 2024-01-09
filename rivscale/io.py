
import glob
import os.path
import numpy as np
import pandas as pd
import netCDF4 as nc
import geopandas as gpd

from errtools.misc import (split_utc_time, field_time_to_swot_time,
    swot_time_to_field_time, get_dist_to_outlet_from_node_id)

import rivscale.data
import rivscale.misc

def read_reach_rivertiles(basedir):
    files = glob.glob(os.path.join(basedir, 'SWOT_L2_HR_RiverTile*.nc'))
    d = rivscale.data.init_swot_reach_rivertile()
    if len(files) == 0:
        # return empty
        return pd.DataFrame(d)
    for fle in files:
        with nc.Dataset(fle) as f:
            #breakpoint()
            for var in d.keys():
                if var == 'cycle':
                    d[var] = d[var] + list(np.zeros_like(f.groups['reaches'].variables['wse'][:]) + int(f.groups['reaches'].cycle_number))
                else:
                    d[var] = d[var] + list(f.groups['reaches'].variables[var][:])
    for var in d.keys():
        d[var] = np.array(d[var])
    df = pd.DataFrame(d)
    # drop data from Nan times
    df = df[np.abs(df['time'])>0]
    df['time_UTC'] = np.array(swot_time_to_field_time(df['time']))
    df['time_UTC'] = df['time_UTC'].astype('datetime64[ns]')
    return df

def read_node_rivertiles(basedir):
    # nodes
    files = glob.glob(os.path.join(basedir, 'SWOT_L2_HR_RiverTile*.nc'))
    d = {
        'wse':[],
        'wse_u':[],
        'area_detct':[],
        'area_total':[],
        'width':[],
        'node_id':[],
        'reach_id':[],
        'time':[],
        'cycle':[],
        'node_q':[]
        }
    if len(files) == 0:
        # try the shape files
        files = glob.glob(os.path.join(basedir,'SWOT_L2_HR_RiverSP_*/SWOT_L2_HR_RiverSP_*.shp'))
        #breakpoint()
        
        if len(files) ==0:
            # return empty dataframe
            return pd.DataFrame(d)
    for fle in files:
        if fle.endswith('.shp'):
            # read in the shape file
            f = gpd.read_file(fle)
            # get the cycle from the filename since it is easy
            jnk, base = os.path.split(fle)
            cyc = int(base.split('_')[5])
            #breakpoint()
            cycs = np.zeros_like(np.array(f['node_id'])) + cyc
            f['cycle'] = cycs
            for var in d.keys():
                tmp = np.array(f[var])
                if (var == 'reach_id') or (var == 'node_id'):
                    tmp = np.array(tmp, dtype=int)
                d[var] = d[var] + list(tmp)
            #breakpoint()
        else:
            # read the netcdf file
            with nc.Dataset(fle) as f:
                #breakpoint()
                for var in d.keys():
                    if var == 'cycle':
                        d[var] = d[var] + list(np.zeros_like(f.groups['nodes'].variables['node_id'][:]) + int(f.groups['nodes'].cycle_number))
                    else:
                        d[var] = d[var] + list(f.groups['nodes'].variables[var][:])
    for var in d.keys():
        d[var] = np.array(d[var])
    df_node = pd.DataFrame(d)
    # drop data from bad times
    #df_node = df_node[np.abs(df_node['time'])>0]
    df_node = df_node[np.array(df_node['time'])>0]
    df_node['time_UTC'] = np.array(swot_time_to_field_time(df_node['time']))
    df_node['time_UTC'] = df_node['time_UTC'].astype('datetime64[ns]')
    df_node['local_node_id'] = rivscale.misc.node_id_to_local_node_id(df_node.node_id) 
    return df_node

def read_rivertiles(basedir):
    df = read_reach_rivertiles(basedir)
    df_node = read_node_rivertiles(basedir)
    return df, df_node

def read_SWORD(fle):
    d = {
        'reach_id':[],
        'reach_length':[],
        'dist_out':[],
        }
    d_up = {}
    d_down = {}
    with nc.Dataset(fle) as f:
        for var in d.keys():
            d[var] = d[var] + list(f.groups['reaches'].variables[var][:])

        for k,rch in enumerate(f.groups['reaches'].variables['reach_id'][:]):
           #breakpoint()
           d_up[str(rch)] = f.groups['reaches'].variables['rch_id_up'][:,k]
           d_down[str(rch)] = f.groups['reaches'].variables['rch_id_dn'][:,k]
    for var in d.keys():
        d[var] = np.array(d[var])
    df = pd.DataFrame(d)
    #breakpoint()
    # do the nodes now
    d = {
        'reach_id':[],
        'node_id':[],
        'dist_out':[],
        }
    with nc.Dataset(fle) as f:
        for var in d.keys():
            d[var] = d[var] + list(f.groups['nodes'].variables[var][:])
    for var in d.keys():
        d[var] = np.array(d[var])
    node_df = pd.DataFrame(d)
    return df, node_df, d_up, d_down

def load_field_data(fles, sword_file):
    # read in PT datafames
    pt_df = None
    drift_df = None
    for fle in fles:
        if 'PT_node_wse' in fle:
            this_df = pd.read_csv(fle)
            # accumulate
            if pt_df is None:
                pt_df = this_df
            else:
                pt_df = pd.concat(
                    [pt_df, this_df], ignore_index=True)
        elif 'wses.csv' in fle:
            # read in drift dataframe
            drift_df = pd.read_csv(fle)
    # need to drop the _1.csv drifts if there are _2.csv ?
    all_drift_ids = np.unique(drift_df['drift_id'])
    for drift_id in all_drift_ids:
        if drift_id.split('_')[-1] == '2.csv':
            # drop the _1 version id _2 version exists
            drift_df = drift_df[drift_df['drift_id'] != drift_id.replace('_2.csv', '_1.csv')]
    # populate extra items
    (drift_df['year'], drift_df['month'], drift_df['daynum'],
     drift_df['hour'], drift_df['minute']) = split_utc_time(
        drift_df['time_UTC'].values)
    pt_df['pt_time_UTC_str'] = pt_df['pt_time_UTC']
    pt_df['pt_time_UTC'] = pt_df['pt_time_UTC'].astype('datetime64[ns]')
    drift_df['time_UTC_str'] = drift_df['time_UTC']
    drift_df['time_UTC'] = drift_df['time_UTC'].astype('datetime64[ns]')
    drift_df['reach_id'] = [int(str(node_id)[:-4] + str(node_id)[-1]) for
        node_id in drift_df.node_id]
    drift_df['local_node_id'] = rivscale.misc.node_id_to_local_node_id(drift_df.node_id)
    #drift_df['local_node_id'] = [int(str(node_id)[-4:-1]) for
    #    node_id in drift_df.node_id]
    dist_to_outlet = get_dist_to_outlet_from_node_id(
        drift_df['node_id'].values, sword_file)
    drift_df['p_dist_out'] = dist_to_outlet
    # pt too
    dist_to_outlet_pt = get_dist_to_outlet_from_node_id(
        pt_df['node_id'].values, sword_file)
    pt_df['p_dist_out'] = dist_to_outlet_pt
    return pt_df, drift_df




