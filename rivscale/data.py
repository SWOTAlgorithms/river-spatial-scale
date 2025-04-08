'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent WIlliams
'''


import glob
import os.path

import argparse

import numpy as np
import pandas as pd

import errtools.plots
import matplotlib.pyplot as plt

import scipy.ndimage
from scipy.linalg import pinv, svd, eigh, norm

from errtools.misc import (split_utc_time, field_time_to_swot_time,
    swot_time_to_field_time, get_dist_to_outlet_from_node_id)

from errtools.plots import plot_cdf

import netCDF4 as nc

import statsmodels.api

import geopandas as gpd

import rivscale.misc

import rivscale.products

import xarray as xr

try:
    import rivscale.ingest
except ModuleNotFoundError:
    print("Problem importing ingest tools, can't use hydrochron")
##### April 2025
def rivertiles_to_dataframe(rivertiles, group='nodes', continent='NA'):
    dataframe_list = []
    for rivertile in rivertiles:
        ds = xr.open_dataset(rivertile, group=group, decode_cf=False)
        if group=='reaches':
            ds = ds.drop(['centerline_lat', 'centerline_lon',
                'n_reach_up','n_reach_dn','rch_id_up', 'rch_id_dn'])
        df = ds.to_dataframe()
        _, tail = os.path.split(rivertile)
        parts = tail.split('_')
        #print(parts)
        df['cycle_id'] = int(parts[4])
        df['pass'] = int(parts[5])
        df['continent'] = continent
        #breakpoint()
        dataframe_list.append(df)
    swot_node_dataframe = pd.concat(dataframe_list)
    swot_node_dataframe = swot_node_dataframe.reset_index(drop=True)
    return swot_node_dataframe

##### Feb 2025
def manage_fields(df, use_wse_sm=False, qual_filter='', dark_thresh=1.0): 
    if df is None:
        return None
    if use_wse_sm:
        df['wse'] = np.array(df['wse_sm']).copy()
        df['wse_u'] = np.array(df['wse_sm_u']).copy()
        df['wse_q'] = np.array(df['wse_sm_q']).copy()
        df['wse_q_b'] = np.array(df['wse_sm_q_b']).copy()
    # drop elements with no_data times
    df = df[df['time_str']!='no_data']
    df['time_str'] = pd.to_datetime(df['time_str'])
    df['date'] = [ dt.date() for dt in df['time_str']]
    #
    # dont qual filter at this stage...only later
    # drop bad data
    #breakpoint()
    # TODO: robustify qual filter methods (OB, IM, OBIM etc)
    df = rivscale.filter.filter_qual(
        df, height=True, area=False,
        kind=qual_filter, dark_thresh=dark_thresh)
    #
    if 'cycle_id' in df.keys():
        df['cycle'] = df['cycle_id']
    if 'pass_id' in df.keys():
        df['pass'] = df['pass_id']
    if 'continent_id' in df.keys():
        df['continent'] = df['continent_id']
    if 'node_id' in df.keys():
        df['local_node_id'] = rivscale.misc.node_id_to_local_node_id(
            df['node_id'])
    df['wse_u'] = df['wse_r_u']
    df['dist_out'] = df['p_dist_out']
    # put sig0 in dB
    if 'rdr_sig0' in df.keys():
        df['sig0 (dB)'] = 10*np.log10(df['rdr_sig0'])
    #if 'continent' not in df.keys():
    #    df['continent'] = 'NA' # TODO: un-hard-code this one
    return df

def get_swot_data(
        cfg,
        stretch_reaches,#sword_node_df,
        kind='Node',
        force=False):
    """
    kind can be 'Node' or 'Reach'
    """
    # first handle optional config params
    section = 'stretch_stack'
    group = 'nodes'
    if kind == 'Reach':
        section = 'reach_avg'
        group = 'reaches'

    if 'method' not in cfg[section].keys():
        cfg[section]['method'] = 'csv'
    if 'use_wse_sm' not in cfg[section].keys():
        cfg[section]['use_wse_sm'] = 'False'
    if 'qual_filter' not in cfg[section].keys():
        cfg[section]['qual_filter'] = 'OB'
    if 'dark_thresh' not in cfg[section].keys():
        cfg[section]['dark_thresh'] = '1.0'
    if cfg[section]['method'] == 'csv':
        df = pd.read_csv(cfg['main']['data_path'])
        # TODO: filter out orbit and granules we want
    if cfg[section]['method'] == 'ingest':
        node_csv_file = cfg['main']['node_csv_file']
        if (os.path.isfile(node_csv_file)) and (not force):
            # just read the already-made input file
            df = pd.read_csv(node_csv_file)
        else:
            basin_ids = cfg['main']['stretch_subset']
            if isinstance(basin_ids, int):
                basin_ids = [basin_ids,] 
            basin_ids0 = list(set(basin_ids).union(set(stretch_reaches)))
            # take off the last number if it is a full reach_id
            # TODO: maybe should ingest more than one reach since
            #       (it is faster to grab a bunch than one at a time)
            basin_ids = []
            for bid in basin_ids0:
                id_str = '{}'.format(bid)
                if len(id_str)>10:#==11:
                   bid = int(id_str[0:10])
                basin_ids.append(bid)
            basin_ids = np.unique(basin_ids)
            print("ingesting basins:", basin_ids)
            df = rivscale.ingest.basin_loop(
                basin_ids,
                out_csv_name=node_csv_file)
        # filter out reaches we want to keep
        df = df[df['reach_id'].isin(stretch_reaches)]
    elif cfg[section]['method'] == 'offline':
        node_csv_file = cfg['main']['node_csv_file']
        if (os.path.isfile(node_csv_file)) and (not force):
            # just read the already-made input file
            df = pd.read_csv(node_csv_file)
        else:
            #read in and create the csv file on the fly
            slc_flavor = cfg['main']['slc_flavor']
            if slc_flavor is None:
                slc_flavor = ''
            pixc_flavor = cfg['main']['pixc_flavor']
            river_flavor = cfg['main']['river_flavor']
            basedir = cfg['main']['data_path']
            # get pass and continent from the config granule name
            pas, continent = cfg['main']['granule'].split('_')
            glob_str = os.path.join(
                basedir,
                '{}_*/*/SWOT_L1B_HR_SLC_*/{}'.format(pas, slc_flavor),
                'SWOT_L2_HR_PIXC_*',
                '{}'.format(pixc_flavor),
                'SWOT_L2_HR_RiverTile_*',
                'SWOT_L2_HR_RiverTile_*{}'.format(river_flavor),
                'SWOT_L2_HR_RiverTile*.nc')
            print(glob_str)
            rivertiles = glob.glob(glob_str)
            df = rivertiles_to_dataframe(rivertiles, group=group,
                    continent=continent)
            # write out the csv file
            df.to_csv(node_csv_file, index=False)
        #breakpoint()
        # filter out reaches we want to keep
        df = df[df['reach_id'].isin(stretch_reaches)]
    elif cfg[section]['method'] == 'reach':
        # go through all the RiverSP data for the desired granules/orbit
        orbit = cfg['main']['orbit']
        pass_cont = cfg['main']['granule']
        df = None
        for reach in stretch_reaches:#df_stretches.keys():
            if 'both' in orbit:
                orbits = ['cal_orbit', 'science_orbit']
            else:
                orbits = [orbit,]
            fles = []
            for this_orbit in orbits:
                fle_str = os.path.join(cfg['main']['data_path'],
                    '{}/{}/Multitemporal_{}/{}_{}_{}_{}.csv'.format(
                        this_orbit,
                        pass_cont,
                        kind,
                        reach,
                        kind,
                        pass_cont,
                        this_orbit))
                this_fles = glob.glob(fle_str)
                fles = fles + this_fles

            for fle in fles:
                print('  ',fle)
                # TODO: should catch if file doesnt exist or cant read it?
                if df is None:
                    # note that keep_default_na=False handles the 'NA'
                    # fields so they dont become 'NaN'
                    df = pd.read_csv(fle, keep_default_na=False)
                else:
                    df = pd.concat(
                        [df,pd.read_csv(fle, keep_default_na=False)],
                        ignore_index=True)
    #
    #use_wse_sm = False
    #sm = cfg['stretch_stack']['use_wse_sm']
    #if ((sm == 'True') or (sm == 'True') or (sm is True)):
    #    use_wse_sm = True
    #
    #if 'dark_thresh' in cfg['data'].keys():
    #dark_thresh = float(cfg['data']['dark_thresh'])
    # don't filter on dark frac here...only on qual
    #df = manage_fields(df, use_wse_sm=use_wse_sm)#, dark_thresh=dark_thresh)
    #breakpoint()
    df = manage_fields(
        df,
        use_wse_sm=cfg[section]['use_wse_sm'],
        qual_filter=cfg[section]['qual_filter'],
        dark_thresh=cfg[section]['dark_thresh'])
    return df

########## Sept 2024, modified to use product class for river stretch processing
def init_dict_from_keys(keys):
    """
    generic way to initialize a dictionary with
    empty lists from a list of keys
    """
    data = dict.fromkeys(keys)
    for key in data.keys():
        data[key] = []
    return data

def check_connected_stretch(stretch_reaches, sword_df, d_up, d_down):
    this_sword_df = sword_df[sword_df['reach_id'].isin(stretch_reaches)]
    reaches_str = [str(reach) for reach in stretch_reaches]
    up_reaches = np.roll(stretch_reaches, -1)
    down_reaches = np.roll(stretch_reaches, 1)
    down_reaches[0] = -1
    up_reaches[-1] = -1
    good = False
    for reach_str, up_reach, down_reach in zip(reaches_str, up_reaches, down_reaches):
        up_good = True
        if up_reach > 0:
            up_good = up_reach in d_up[reach_str]
        down_good = True
        if down_reach > 0:
            down_good = down_reach in d_down[reach_str]
        good = up_good and down_good
    return good

def make_stretch_stack(stretch_name, stretch_reaches, swot_node_df, sword_node_df, d_up, d_down):
    """
    create and populate a StretchStack object from node dataframes and SWORD
    stretch_reaches = list of connected reaches along a river
    swot_node_df    = the swot node data (multiple cycles)
    sword_node_df, d_up, d_down   = SWORD info as output by rivscale.io.read_SWORD()
    """
    # Init the potentially multireach stretch stack 
    #data = rivscale.products.RiverStretchData()
    #data['stretch_reaches'] = stretch_reaches
    data = rivscale.products.StretchStack()
    data['reaches'] = stretch_reaches
    data['stretch_name'] = stretch_name
    # check that stretch_reaches list is a connected stretch
    #if not check_connected_stretch(stretch_reaches, sword_node_df, d_up, d_down):
    #    print('WARNING: THE INPUT REACHES NOT A CONNECTED STRETCH:', stretch_reaches)
    #    return data
    #if 'cycle_id' not in swot_node_df.keys():
    #    if 'cycle' in swot_node_df.keys():
    #        swot_node_df['cycle_id'] = swot_node_df['cycle']
    swot_keys = [*swot_node_df.keys()]
    data_keys = [*data.VARIABLES.keys()]
    keys_2D = []
    keys_1D = []
    for key in data_keys:
        siz = len(data.VARIABLES[key]['dimensions'])
        if siz==2:
            keys_2D.append(key)
        elif siz==1:
            keys_1D.append(key)
    keys = list(set(swot_keys) & set(keys_2D))
    #breakpoint()
    # get separate list of the 1D keys
    sword_keys = ['dist_out', 'node_id','local_node_id']
    extra_keys = sword_keys + ['time_id','granule_id']
    time_ids = np.sort(np.unique(np.floor(swot_node_df['time']/60/60)))
    stretch_data = init_dict_from_keys(keys + extra_keys)
    # go through each reach and stack the various items
    for reach in stretch_reaches:
        this_df = swot_node_df[
            swot_node_df['reach_id']==reach].sort_values(['local_node_id', 'time'])
        local_node_ids = np.unique(this_df['local_node_id'])
        this_sword_df = sword_node_df[
            sword_node_df['reach_id']==reach].sort_values(['dist_out','node_id'])
        if 'local_node_id' not in this_sword_df.keys():
            this_sword_df['local_node_id'] = rivscale.misc.node_id_to_local_node_id(
                this_sword_df['node_id'])
        nodes = this_sword_df['local_node_id']
        node_ids = this_sword_df['node_id']
        dist_out = this_sword_df['dist_out']
        signal = init_dict_from_keys(keys + extra_keys)
        # go through the time/cycles
        for t_id in time_ids:
            that_df = this_df[np.floor(this_df['time']/60/60)==t_id]
            this_nodes = np.array(that_df['local_node_id'])
            for key in keys:
                this_key = np.array(that_df[key])
                full_key = np.ones(np.shape(nodes)) + np.nan
                for n,kk in zip(this_nodes, this_key):
                    full_key[nodes==n] = kk
                signal[key].append(full_key)
            signal['time_id'].append(t_id)
            g_ids = []
            for cyc, pas, cont in zip(
                    that_df['cycle'], that_df['pass'], that_df['continent']):
                g_id = '{:03d}_{:03d}_{}'.format(cyc, pas, cont)
                g_ids.append(g_id)
            #breakpoint()
            ugids = np.unique(g_ids)
            ugid='000_000_00'
            if len(ugids)>0:
                ugid = ugids[0]
            #else:
            #    breakpoint()
            #for g in ugids:# concat strings
            #    if len(ugid)==0:
            #        ugid = g
            #    else:
            #        ugid = ugid + "_" + g
            signal['granule_id'].append(ugid)
        if len(np.array(signal[keys[0]]))==0:
            continue
        for key in keys:
            stretch_data[key].append(np.array(signal[key]).T)
        for key in sword_keys:
            stretch_data[key].append(this_sword_df[key])
        stretch_data['time_id'].append(np.array(signal['time_id']))
        stretch_data['granule_id'].append(np.array(signal['granule_id']))
    # now smash each consecutive reach together
    for key in keys + sword_keys:
        data[key] = np.concatenate(stretch_data[key])
    data['time_id'] = stretch_data['time_id'][0]
    #breakpoint()
    # squash out the '000_000_00' if possible
    out_g = stretch_data['granule_id'][0].copy()
    for k in range(len(stretch_reaches)):
        this_g = stretch_data['granule_id'][k].copy()
        out_g[out_g=='000_000_00'] = this_g[out_g=='000_000_00']
    data['granule_id'] = out_g
    return data

########## Older code ... TODO: clean up/delete/revise
def init_swot_reach_rivertile():
    d = {
        'wse':[],
        'wse_u':[],
        'slope':[],
        'slope2':[],
        'area_detct':[],
        'area_total':[],
        'width':[],
        'reach_id':[],
        'p_length':[],
        'time':[],
        'cycle':[],
        'xovr_cal_c':[],
        'reach_q':[]
        }
    return d

def bayes_to_swot_df(reach_average_data, swot_df):
    d = init_swot_reach_rivertile()
    for reach in swot_df['reach_id']:
       this_swot_df = swot_df[swot_df['reach_id']==reach].sort_values('cycle')
       ind = np.where(np.array(reach_average_data['reach'])==reach)
       #breakpoint()
       if len(ind[0])==0:
            continue
       for key in d.keys():
           dat = np.array(this_swot_df[key])
           cycles = np.array(this_swot_df['cycle'])
           if key == 'wse':
               # null out the wse
               dat = dat + np.nan
           if key in reach_average_data.keys():
               if key == 'cycle':
                   dat = cycles
               else:
                   # replace the specific cycles
                   dat0 = np.array(reach_average_data[key][ind[0][0]])
                   cycles0 = np.array(reach_average_data['cycle'][ind[0][0]])
                   for k,cycl in enumerate(cycles0):
                       ind2 = np.where(cycles == cycl)
                       #breakpoint()
                       dat[ind2] = dat0[k]
           d[key] = d[key] + list(dat)
    df = pd.DataFrame(d)
    df['time_UTC'] = np.array(swot_time_to_field_time(df['time']))
    df['time_UTC'] = df['time_UTC'].astype('datetime64[ns]')
    return df 







def get_reach_average_df(pt_df, reaches):
    d = {
        'reach_wse':[],
        'reach_slope':[],
        'reach_num_pt':[],
        'reach_length':[],
        'reach_times':[],
        'reach_id':[],
        }
    for reach in reaches:
        # compute the reach wse and slope from PTs
        this_pt_df = pt_df[pt_df['reach_id']==reach].sort_values('pt_time_UTC')
        if len(this_pt_df) == 0:
            continue
        #breakpoint()
        # loop through each time
        this_pt_df['local_node_id'] = rivscale.misc.node_id_to_local_node_id(this_pt_df['node_id'])
        pt_times = np.unique(this_pt_df['pt_time_UTC'])
        
        reach_wse = []
        reach_slope = []
        reach_num_pt = []
        reach_length = []
        reach_times = []
        reach_ids = []
        for pt_time in pt_times:
            this_df = this_pt_df[this_pt_df['pt_time_UTC']==pt_time].sort_values('local_node_id')
            pt_ids = this_df['pt_serial']
            wse = np.array(this_df['mean_node_pt_wse_m'])
            dist_out = np.array(this_df['p_dist_out'])
            length = np.nan
            slope = np.nan
            num_pt = len(wse)
            if num_pt > 1:
                length = dist_out[-1] - dist_out[0]
                slope = (wse[-1] - wse[0]) / length
            reach_wse.append(np.mean(wse))
            reach_slope.append(slope)
            reach_num_pt.append(num_pt)
            reach_length.append(length)
            reach_times.append(pt_time)
            reach_ids.append(reach)
        d['reach_wse'] = d['reach_wse'] + reach_wse
        d['reach_slope'] = d['reach_slope'] + reach_slope
        d['reach_num_pt'] = d['reach_num_pt'] + reach_num_pt
        d['reach_length'] = d['reach_length'] + reach_length
        d['reach_times'] = d['reach_times'] + reach_times
        d['reach_id'] = d['reach_id'] + reach_ids
        reach_average_df = pd.DataFrame(d)
    return reach_average_df

def init_full_profile_data(keys=[
        'wse_stack',
        'time_stack',
        'wse_u_stack', 
        'node_q_stack',
        'area_stack',
        'area_total_stack',
        'node_id',
        'dist_out',
        'nodes',
        'reach',
        'cycle',
        'network',]):
    full_profile_data = dict.fromkeys(keys)
    for key in full_profile_data.keys():
        full_profile_data[key] = []
    """
    full_profile_data = {
        'wse_stack':[],
        'time_stack':[],
        'wse_u_stack':[],
        'node_q_stack':[],
        'area_stack':[],
        'area_total_stack':[],
        'node_id':[],
        'dist_out':[],
        'nodes':[],
        'reach':[],
        'cycle':[],
        'network':[],
        }
    """
    return full_profile_data

def make_swot_data_stack(swot_node_df, sword_node_df):
    reaches = np.unique(swot_node_df['reach_id'])
    cycles = np.sort(np.unique(swot_node_df['cycle']))
    full_profile_data = init_full_profile_data()
    for k,reach in enumerate(reaches):
        # get "full" profile by linearly interpolating over missing nodes...
        this_df = swot_node_df[
            swot_node_df['reach_id']==reach].sort_values('local_node_id')
        local_node_ids = np.unique(this_df['local_node_id'])
        this_sword_df = sword_node_df[sword_node_df['reach_id']==reach].sort_values('node_id')
        nodes = rivscale.misc.node_id_to_local_node_id(this_sword_df['node_id'])
        node_ids = this_sword_df['node_id']
        dist_out = this_sword_df['dist_out']
        #cycles = np.sort(np.unique(this_df['cycle']))
        signal = []
        signal_u = []
        signal_q = []
        signal_t = []
        signal_a = []
        signal_a_t = []
        signal_cycle = []
        for cycl in cycles:
            df = this_df[this_df['cycle']==cycl]
            # drop bad node data
            this_node_q = np.array(df['local_node_id'])
            df = df[df['node_q']<=1]# 1
            this_nodes = np.array(df['local_node_id'])
            this_wse = np.array(df['wse'])
            this_wse_u = np.array(df['wse_u'])
            this_area_total = np.array(df['area_total'])
            this_area = np.array(df['area_detct'])
            this_node_q = np.array(df['node_q'])
            this_time = np.array(df['time'])
            #if len(this_wse) < 10:#len(nodes)*3/4:
            #    continue
            #breakpoint()
            full_wse = np.ones(np.shape(nodes)) + np.nan
            full_wse_u = np.ones(np.shape(nodes)) + np.nan
            full_area = np.ones(np.shape(nodes)) + np.nan
            full_area_t = np.ones(np.shape(nodes)) + np.nan
            full_wse_q = np.ones(np.shape(nodes)) + np.nan
            full_time = np.ones(np.shape(nodes)) + np.nan
            for n,w,u,q,t,a,a_t in zip(this_nodes, this_wse, this_wse_u, this_node_q, this_time, this_area, this_area_total):
                full_wse[nodes==n] = w
                full_wse_u[nodes==n] = u
                full_wse_q[nodes==n] = q
                full_time[nodes==n] = t
                full_area[nodes==n] = a
                full_area_t[nodes==n] = a_t
            # interpolate to full extent
            #full_wse = np.interp(nodes, this_nodes, this_wse)
            #signal.append(full_wse.T)
            signal.append(full_wse)
            signal_u.append(full_wse_u)
            signal_q.append(full_wse_q)
            signal_t.append(full_time)
            signal_a.append(full_area)
            signal_a_t.append(full_area_t)
            signal_cycle.append(cycl)
        signal = np.array(signal)
        signal_cycle = np.array(signal_cycle)
        if len(signal)==0:
            continue
        #full_profile_data['signal'].append(signal)
        full_profile_data['wse_stack'].append(signal.T)
        full_profile_data['time_stack'].append(np.array(signal_t).T)
        full_profile_data['wse_u_stack'].append(np.array(signal_u).T)
        full_profile_data['node_q_stack'].append(np.array(signal_q).T)
        full_profile_data['area_stack'].append(np.array(signal_a).T)
        full_profile_data['area_total_stack'].append(np.array(signal_a_t).T)
        full_profile_data['node_id'].append(node_ids)
        full_profile_data['dist_out'].append(dist_out)
        full_profile_data['nodes'].append(nodes)
        full_profile_data['reach'].append([reach,])
        full_profile_data['network'].append(k)
        full_profile_data['cycle'].append(np.array(signal_cycle))
        #
    #full_profile_data['cycle'] = cycles
    return full_profile_data

def make_generic_stack(swot_node_df, sword_node_df,
        keys=['wse','node_q_b','width','area_total',
            'wse_r_u','area_tot_u','time', 'lat', 'lon']):
    # make stack for each reach for list of key values
    # do the filtering for quality etc before calling this function
    reaches = np.unique(swot_node_df['reach_id'])
    full_profile_data = init_full_profile_data(keys)#dict.fromkeys(keys, [])
    full_profile_data['node_id'] = []
    full_profile_data['nodes'] = []
    full_profile_data['dist_out'] = []
    full_profile_data['reach'] = []
    full_profile_data['time_id'] = []
    full_profile_data['network'] = []
    time_ids = np.sort(np.unique(np.floor(swot_node_df['time']/60/60)))
    for k,reach in enumerate(reaches):
        #print(reach)
        this_df = swot_node_df[
            swot_node_df['reach_id']==reach].sort_values(['local_node_id', 'time'])
        local_node_ids = np.unique(this_df['local_node_id'])
        this_sword_df = sword_node_df[
            sword_node_df['reach_id']==reach].sort_values('node_id')
        nodes = rivscale.misc.node_id_to_local_node_id(this_sword_df['node_id'])
        node_ids = this_sword_df['node_id']
        dist_out = this_sword_df['dist_out']
        signal = init_full_profile_data(keys)#dict.fromkeys(keys, [])
        signal['time_id'] = []
        for t_id in time_ids:
            #print(t_id)
            that_df = this_df[np.floor(this_df['time']/60/60)==t_id]
            this_nodes = np.array(that_df['local_node_id'])
            for key in keys:
                this_key = np.array(that_df[key])
                full_key = np.ones(np.shape(nodes)) + np.nan
                for n,kk in zip(this_nodes, this_key):
                    full_key[nodes==n] = kk
                signal[key].append(full_key)
            signal['time_id'].append(t_id)
        #print(keys)
        if len(np.array(signal[keys[0]]))==0:
            continue
        for key in keys:
            full_profile_data[key].append(np.array(signal[key]).T)
        full_profile_data['node_id'].append(node_ids)
        full_profile_data['nodes'].append(nodes)
        full_profile_data['dist_out'].append(dist_out)
        full_profile_data['reach'].append([reach,])
        full_profile_data['time_id'].append(np.array(signal['time_id']))
        full_profile_data['network'].append(k)
    return full_profile_data

def stack_mean_and_covariance(full_profile_data,
        signal_key='wse_stack_interp', uncert_key='wse_u_stack'):
    stats = {'mean':[], 'median':[], 'Ry':[], 'reach':[], 'network':[]}
    for k, network in enumerate(full_profile_data['network']):
        signal = full_profile_data[signal_key][k]
        signal_u = full_profile_data[uncert_key][k]
        mn = np.nanmean(signal, axis=1)
        med = np.nanmedian(signal, axis=1)
        # define the zero-mean signal
        #breakpoint()
        M,N = np.shape(signal)
        Y = signal-np.tile(mn,(N, 1)).T
        Y[np.isnan(signal)] = 0
        count = np.ones((M,N))
        count[np.isnan(signal)] = 0
        # estimate the covariance of the zero-mean signal
        #Ry = np.inner(Y, Y) / N
        C = np.inner(count, count)
        Ry = np.inner(Y, Y) / C
        Ry[C==0] = 0
        
        #breakpoint()
        ## subtract off the known noise covariance
        ## handling resulting negaitve diagonal elements
        #Rv = np.diag(np.nanmean(signal_u, axis=1))
        #Ry_diag = np.diag(np.diag(Ry))
        #R1 = Ry_diag - Rv
        #R1_full = Ry - Rv
        #Rmax = np.diag(np.max(np.abs(R1_full), axis=0))
        ## replace negative diag values with the next biggest magnitude element
        ## in each row
        ##R1[R1<0] = Rmax[R1<0]
        #Ry = Ry - Rv
        #Ry[R1<0] = Rmax[R1<0] + 1e-3
        
        
        #breakpoint()
        stats['mean'].append(mn)
        stats['median'].append(med)
        stats['Ry'].append(Ry)
        stats['reach'].append(full_profile_data['reach'][k])
        stats['network'].append(network)
    return stats


def compute_reach(bayes_data):
    reach_data = {
        'wse':[],
        'slope':[],
        'width':[],
        'reach':[],
        'cycle':[]
        }
    for k, reach in enumerate(bayes_data['reach']):
        wse_stack = bayes_data['wse_stack'][k]
        dist_out = np.array(bayes_data['dist_out'][k])
        reach_length = dist_out[-1]-dist_out[0]
        area_total_stack = bayes_data['area_total_stack'][k]
        #breakpoint()
        reach_data['wse'].append(np.mean(wse_stack, axis=0))
        reach_data['slope'].append((wse_stack[-1,:] - wse_stack[0,:]) / reach_length)
        reach_data['width'].append(np.nansum(area_total_stack, axis=0) / reach_length)
        reach_data['reach'].append(reach)
        reach_data['cycle'].append(bayes_data['cycle'][k])
    return reach_data



def get_connected_networks(sword_df, d_up, d_down):
    this_sword_df = sword_df.sort_values('dist_out')
    reach_ids = np.array(this_sword_df['reach_id'])
    network_list = []
    while len(reach_ids)>0:
        lst = [reach_ids[0],] # start with the closest to outlet
        while True:
            #breakpoint()
            rch = str(lst[-1])
            if rch.endswith('4'):
                # break connectivity at dams and dont include dam reaches
                break
            if rch not in d_up:
                # break when next reach not in list of remaining reaches
                break
            this_d_up = d_up[rch][d_up[rch]>0]
            if len(this_d_up)==0:
                # break if there are no more upstream reaches
                break
            # TODO handle multiple up-stream reaches
            lst.append(this_d_up[0])
            #breakpoint()
        reach_ids = list(set(reach_ids) - set(lst))
        network_list.append(lst)
    return network_list

def network_stack(full_profile_data, network_list):
    out_profile_data = init_full_profile_data(full_profile_data.keys())
    #out_profile_data = dict.fromkeys(full_profile_data.keys(), [])#init_full_profile_data()
    for j, reaches in enumerate(network_list):
        this_profile_data = init_full_profile_data(full_profile_data.keys())
        #this_profile_data = dict.fromkeys(full_profile_data.keys(), [])#init_full_profile_data()
        #breakpoint()
        for k,reach in enumerate(reaches):
            rs = np.squeeze(full_profile_data['reach'])
            ind = np.where(rs==reach)
            if len(ind)==0:
                continue
            ind = ind[0]
            if len(ind)==0:
                continue
            if len(ind)>0:
                ind = ind[0]
            #if len(ind)==0:
            #    continue
            for key in full_profile_data.keys():
                #print(key, ind)
                if key == 'reach':
                    this_profile_data[key].append(rs[ind])
                elif key not in ['time_id','cycle','network']:
                    # for some reason the print statement below doesnt pass
                    # the Cryptography.InsecureAlgorithm checks
                    #if 'wse' in key:
                    #    print(key, ind, np.shape(full_profile_data[key][ind]))
                    this_profile_data[key].append(full_profile_data[key][ind])
        for key in full_profile_data.keys():
            print(key)
            #breakpoint()
            if key == 'network':
                out_profile_data[key].append(j)
            elif key in ['time_id','cycle']:#key=='cycle':
                out_profile_data[key].append(full_profile_data[key][0])
            elif key not in ['time_id','cycle','reach']:
                #breakpoint()
                # for some reason the print statement below doesnt pass
                # the Cryptography.InsecureAlgorithm checks
                #if 'wse' in key:
                #    for ii in range(len(this_profile_data[key])):
                #        print(key, ii, np.shape(this_profile_data[key][ii]))
                #    #print(np.shape(this_profile_data[key][7]))
                if len(this_profile_data[key])>0:
                    out_profile_data[key].append(np.concatenate(this_profile_data[key]))
                else:
                    out_profile_data[key].append(this_profile_data[key])
        out_profile_data['reach'].append(this_profile_data['reach'])
        #breakpoint()
        #out_profile_data['cycle'].append(full_profile_data['cycle'][j])
    return out_profile_data


def stuff_pt_stack(pt_swot_match, full_profile_data):
    """
    stuff the PT data into the cycle stack
    """
    full_pt_wse_stack = []
    for k, network in enumerate(full_profile_data['network']):
        pt_wse_stack = np.ones_like(full_profile_data['wse_stack'][k]) + np.nan
        this_df = pt_swot_match[pt_swot_match['network']==network]
        if len(this_df) == 0:
            full_pt_wse_stack.append(pt_wse_stack)
            continue
        cycle = full_profile_data['cycle'][k]
        nodes = full_profile_data['node_id'][k]
        for node_id, cycl, wse in zip(this_df['node_id'], this_df['cycle'], this_df['pt_wse_m']):
            #breakpoint()
            n = np.argmin(np.abs(nodes-node_id))
            c = np.argmin(np.abs(cycle-cycl))
            if (nodes[n]==node_id) and (cycle[c]==cycl):
                pt_wse_stack[n,c] = wse
        full_pt_wse_stack.append(pt_wse_stack)
    full_profile_data['pt_wse_stack'] = full_pt_wse_stack
    #return full_profile_data
