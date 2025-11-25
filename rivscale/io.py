'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent Williams
'''
import glob
import os.path
import numpy as np
import pandas as pd
import netCDF4 as nc
import geopandas as gpd

from rivscale.misc import (split_utc_time, field_time_to_swot_time,
    swot_time_to_field_time, get_dist_to_outlet_from_node_id)

import rivscale.data
import rivscale.misc

def read_reach_rivertiles(basedir):
    files = glob.glob(os.path.join(basedir, 'SWOT_L2_HR_RiverTile*.nc'))
    #print("processing reach tile files:", files)
    d = rivscale.data.init_swot_reach_rivertile()
    if len(files) == 0:
        # return empty
        return pd.DataFrame(d)
    for i, fle in enumerate(files):
        print('reach file: {}, {} of {}'.format(fle, i, len(files)))
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
    print("node basedir", basedir)
    files = glob.glob(os.path.join(basedir, 'SWOT_L2_HR_RiverTile*.nc'))
    print("processing node tile files:", files)
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
        print("processing node tile files:", files)
        if len(files) ==0:
            # return empty dataframe
            return pd.DataFrame(d)
    for i, fle in enumerate(files):
        print('node file: {}, {} of {}'.format(fle, i, len(files)))
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
        'river_name':[],
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
        'node_length':[],
        'river_name':[],
        'y':[],
        'x':[]
        }
    with nc.Dataset(fle) as f:
        for var in d.keys():
            d[var] = d[var] + list(f.groups['nodes'].variables[var][:])
    for var in d.keys():
        d[var] = np.array(d[var])
    node_df = pd.DataFrame(d)
    node_df['p_lat'] = node_df['y']
    node_df['p_lon'] = node_df['x']
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


def load_product(infile, product_name=None, force=False):
    #try to load the product, if cant, return None
    obj = None
    # first check if the file exists
    if not os.path.exists(infile):
        return None
    if product_name is None:
        # try to derive it from the filename
        pth, base = os.path.split(infile)
        product_name = base
    if force:
        # return None
        return obj
    # now try each case
    if 'stretch_stack' in product_name:
        # it is a stretch    
        obj = rivscale.products.stretch_stack.StretchStack.from_ncfile(
            infile)
    elif 'stats' in product_name:
        # its an alongstats product
        obj = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
            infile)
    elif 'avg' in product_name:
        # it is a stretch_average (or reach_average)
        obj = rivscale.products.stretch_average.StretchAverageStats.from_ncfile(
            infile)
    elif 'height_width' in product_name:
        if 'array' in product_name:
            # it is a height_width_array_file
            obj = rivscale.products.height_width_array.HeightWidthModelArray.from_ncfile(
                infile)
        else:
            # it is a height_width file
            obj = rivscale.products.height_width.HeightWidthModel.from_ncfile(
                infile)
    elif 'flow' in product_name:
        obj = rivscale.products.flow_state.FlowStateModel.from_ncfile(
            infile)
    elif 'bayes' in product_name:
        obj = rivscale.products.bayes_data.BayesData.from_ncfile(
            infile)
    return obj



