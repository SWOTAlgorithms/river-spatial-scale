'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent WIlliams
'''
import glob
import os.path
import numpy as np
import pandas as pd
import rivscale.filter
import rivscale.misc
import rivscale.products.stretch_stack
import xarray as xr
import datetime

import sqlite3

from rivscale.misc import TIME_ID_QUANTIZATION

def read_sql_data(fle, reach_ids, group='Node'):
    ####
    # read the nodes
    ####
    table = 'swot_node_df'
    if 'reach' in group.lower():
        table = 'swot_reach_df'
    db = sqlite3.connect(fle)
    rid_str = ""
    if len(reach_ids)>1:
        for rid in reach_ids[0:-1]:
            rid_str = rid_str + f"{rid}, "
        rid_str = rid_str + f"{reach_ids[-1]}"
    else:
        rid_str = f"{reach_ids[0]}"
    query = f"SELECT * FROM swot_reach_df WHERE reach_id IN ({rid_str})"
    df = pd.read_sql(query, db, coerce_float=False)
    # drop rows with nan times (assumes all data is invalid)
    #breakpoint()
    df = df.dropna(subset=['time'])
    if len(df)==0:
        # handle empty df
        return None
    ####
    # read in the filename table
    ####
    uindex = np.unique(df['file_index'])
    uindex_str = ""
    for ind in uindex[0:-1]:
        uindex_str = uindex_str + f"{ind}, "
    uindex_str = uindex_str + f"{uindex[-1]}"
    query = f"SELECT * FROM swot_filenames WHERE `index` IN ({uindex_str})"
    df_filenames = pd.read_sql(query, db)
    # just grab the fileds that we care about
    dic = {
            'file_index': np.array(df_filenames['index']),
            'pass_id': np.array(df_filenames['pass']),
            'cycle_id': np.array(df_filenames['cycle']),
            'crid': np.array(df_filenames['crid']),
            }
    df_ind = pd.DataFrame(dic)
    ####
    # merge the databases
    ####
    df_merged = df.merge(df_ind, on='file_index', how='inner')

    # sort it by time
    #df_merged = df_merged.sort_values(by='time')
    db.close()
    return df_merged

def dawg_file_to_df(fle, group='node'):
    this_group = 'node'
    if 'reach' in group:
        this_group = 'reach'
    #
    ds = xr.open_dataset(fle, group=this_group)#, decode_cf=False)
    # get the cyc/pas
    ds0 = xr.open_dataset(fle)                
    cyc_pas = np.array(ds0['observations'])
    cyc = np.array([
        int(i.decode("utf-8").split('_')[0]) for i in cyc_pas])
    pas = np.array([
        int(i.decode("utf-8").split('_')[1]) for i in cyc_pas])
    ds['pass'] = (('nt'), pas)
    ds['cycle'] = (('nt'), cyc)
    df = ds.to_dataframe()
    df['time_str'] = np.array([i.decode("utf-8") for i in df['time_str']])
    time_str = np.array(df['time_str'])
    time_str[time_str=='nan'] = '2000-01-01T00:00:00Z'
    # drop the trailing 'Z'
    time_str = np.array([t[0:-1] for t in time_str])
    # replace the T with space
    time_str = np.array([t.replace('T',' ') for t in time_str])
    df['time_str'] = time_str
    t_swot = rivscale.misc.field_time_to_swot_time(df['time_str'])
    # replace time with swot time
    df['time'] = t_swot
    # handle nans in node_q_b
    n_q_b = np.array(df['node_q_b'])
    n_q_b[np.isnan(n_q_b)] = 2**29-1
    df['node_q_b'] = n_q_b.astype(int)
    #breakpoint()
    return df

try:
    import rivscale.ingest
except ModuleNotFoundError:
    # TODO make this a warning instead of a print?
    print("Problem importing ingest tools, can't use hydrochron")

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

def manage_fields(df, use_wse_sm=False, qual_filter='', dark_thresh=1.0): 
    if df is None:
        return None
    if use_wse_sm:
        df['wse'] = np.array(df['wse_sm']).copy()
        df['wse_u'] = np.array(df['wse_sm_u']).copy()
        df['wse_q'] = np.array(df['wse_sm_q']).copy()
        df['wse_q_b'] = np.array(df['wse_sm_q_b']).copy()
    # drop elements with no_data times
    #breakpoint()
    if 'time_str' not in df.keys():
        # map the time variable to datetime
        df['time_str'] = rivscale.misc.swot_time_to_field_time(df['time'])
        # drop dates that are too early
        df['date'] = [ dt.date() for dt in df['time_str']]
        min_date = datetime.date(2023,1,1)
        df = df[df['date']>min_date]
    else:
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
        kind=qual_filter, dark_thresh=dark_thresh,
        use_wse_sm=use_wse_sm)
    #
    if 'cycle_id' in df.keys():
        df['cycle'] = df['cycle_id']
    if 'pass_id' in df.keys():
        df['pass'] = df['pass_id']
    if 'continent_id' not in df.keys():
        # derive the continent from the reach_ids
        cont_mapping = {
            '9':'GR',
            '8':'AR',
            '7':'NA',
            '6':'SA',
            '5':'AU',
            '4':'AS',
            '3':'SI',
            '2':'EU',
            '1':'AF'
            }
        cont_code = [f'{i}'[0] for i in df['reach_id']]
        cont_id = []
        for code in cont_code:
           cont_id.append(cont_mapping[code])
        df['continent_id'] = np.array(cont_id)
    df['continent'] = df['continent_id']
    if 'node_id' in df.keys():
        df['local_node_id'] = rivscale.misc.node_id_to_local_node_id(
            df['node_id'])
    df['wse_u'] = df['wse_r_u']
    df['dist_out'] = df['p_dist_out']
    df['node_length'] = df['p_length']
    #if 'p_dist_out' in df.keys():
    #    df['dist_out'] = df['p_dist_out']
    #if 'p_length' in df.keys():
    #    df['node_length'] = df['p_length']
    # put sig0 in dB
    if 'rdr_sig0' in df.keys():
        df['sig0 (dB)'] = 10*np.log10(df['rdr_sig0'])
    if 'xtrk_dist' in df.keys():
        tmp = df['xtrk_dist']
        not_scalar = np.array([np.logical_not(np.isscalar(t)) for t in tmp])
        tmp[not_scalar] = np.nan # handle null ct...not sure why we get it though
        #breakpoint()
        df['cross_track'] = tmp / 1000.0 #(km)
    #if 'continent' not in df.keys():
    #    df['continent'] = 'NA' # TODO: un-hard-code this one
    return df

def get_swot_data(
        cfg,
        stretch_reaches,#sword_node_df,
        all_reaches,
        kind='Node',
        force=False,
        sword_df=None):
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
        cfg[section]['qual_filter'] = 'OUTER_BASELINE'
    if 'dark_thresh' not in cfg[section].keys():
        cfg[section]['dark_thresh'] = '1.0'
    if cfg[section]['method'] == 'csv':
        df = pd.read_csv(cfg['main']['data_path'], keep_default_na=False)
        # TODO: filter out orbit and granules we want
    if cfg[section]['method'] == 'ingest':
        ingest_csv_file = cfg['main']['ingest_csv_file']
        if 'collection_name' not in cfg[section].keys():
            # Version C
            #cfg[section]['collection_name'] = 'SWOT_L2_HR_RiverSP_2.0'
            # Version D
            cfg[section]['collection_name'] = 'SWOT_L2_HR_RiverSP_D'
        if (os.path.isfile(ingest_csv_file)) and (not force):
            # just read the already-made input file
            df = pd.read_csv(ingest_csv_file)
        else:
            """
            basin_ids = cfg['main']['stretch_subset']
            if isinstance(basin_ids, int):
                basin_ids = [basin_ids,]
            elif isinstance(basin_ids, str):
                basin_ids = [item for item in basin_ids.split()]
            # drop non-numeric basin ids (generic stretch_names)
            basin_ids0 = []
            for item in basin_ids:
                if item.isnumeric():
                    basin_ids0.append(item)
            basin_ids0 = list(set(basin_ids0).union(set(stretch_reaches)))
            """
            # ingest all reaches used in all stretches for this run
            basin_ids0 = all_reaches
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
                out_csv_name=ingest_csv_file,
                kind=kind,
                collection_name=cfg[section]['collection_name'],
                sword_df=sword_df)
        # filter out reaches we want to keep
        df = df[df['reach_id'].isin(stretch_reaches)]
    elif cfg[section]['method'] == 'offline':
        ingest_csv_file = cfg['main']['ingest_csv_file']
        if (os.path.isfile(ingest_csv_file)) and (not force):
            # just read the already-made input file
            df = pd.read_csv(ingest_csv_file)
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
            df.to_csv(ingest_csv_file, index=False)
        #breakpoint()
        # filter out reaches we want to keep
        df = df[df['reach_id'].isin(stretch_reaches)]
    elif cfg[section]['method'] == 'reach':
        # go through all the reach-centric RiverSP dataframes (csv format)
        #
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
    elif cfg[section]['method'] == 'reach_nc':
        # go through all the reach-centric RiverSP dataframes (nc format)
        #
        orbit = cfg['main']['orbit']
        pass_cont = cfg['main']['granule']
        df = None
        for reach in stretch_reaches:#df_stretches.keys():
            if 'both' in orbit:
                orbits = ['cal', 'science']
                #orbits = ['cal', 'science_20260508'] # TODO: generalize this?
            else:
                orbits = [orbit,]
            r_str = '{}'.format(reach)
            cont_num = r_str[0]
            sub_basin_num = r_str[1:6]
            fles = []
            for this_orbit in orbits:
                fle_str = os.path.join(cfg['main']['data_path'],
                    '{}/{}/{}/{}_{}.nc'.format(
                        this_orbit,
                        cont_num,
                        sub_basin_num,
                        reach,
                        kind.lower()))
                this_fles = glob.glob(fle_str)
                fles = fles + this_fles
            #breakpoint()
            for fle in fles:
                print('  ',fle)
                # TODO: should catch if file doesnt exist or cant read it?
                if df is None:
                    ds = xr.open_dataset(fle, decode_cf=False)
                    df = ds.to_dataframe()
                    #breakpoint()
                else:
                    ds = xr.open_dataset(fle, decode_cf=False)
                    this_df = ds.to_dataframe()
                    df = pd.concat([df,this_df],ignore_index=True)
    elif cfg[section]['method'] == 'sql':
        orbit = cfg['main']['orbit']
        pass_cont = cfg['main']['granule']
        df = None
        if 'both' in orbit:
            orbits = ['cal', 'science']
        else:
            orbits = [orbit,]
        fles = []
        for this_orbit in orbits:
            orbit_str = 'Nom'
            if 'cal' in this_orbit:
                orbit_str='Cal'
            fle_str = os.path.join(cfg['main']['data_path'],
                'SWOT_L2_HR_RiverSP_{}.sqlite3'.format(
                    orbit_str))
            this_fles = glob.glob(fle_str)
            fles = fles + this_fles
        #breakpoint()
        for fle in fles:
            print('  ',fle)
            # TODO: should catch if file doesnt exist or cant read it?
            if df is None:
                df = read_sql_data(fle, stretch_reaches)
            else:
                this_df = read_sql_data(fle, stretch_reaches, group=group)
                df = pd.concat([df,this_df],ignore_index=True)
                #breakpoint()
    elif cfg[section]['method'] == 'dawg':
        # read in the dawg-confluence format files
        #orbit = cfg['main']['orbit']
        #pass_cont = cfg['main']['granule']
        #this_group = 'node'
        #if 'reach' in group:
        #    this_group = 'reach'
        df = None
        for reach in stretch_reaches:#df_stretches.keys():
            #breakpoint()
            #r_str = '{}'.format(reach)
            fle_str = os.path.join(cfg['main']['data_path'],
                    '{}_SWOT.nc'.format(reach))
            fles = glob.glob(fle_str)
            #breakpoint()
            # these are already stretch-stack-like, need tool to convert
            # to stretch_stack?
            for fle in fles:
                print('  ',fle)
                # TODO: should catch if file doesnt exist or cant read it?
                if df is None:
                    df = dawg_file_to_df(fle, group=group)
                else:
                    this_df = dawg_file_to_df(fle, group=group)
                    df = pd.concat([df,this_df],ignore_index=True)
            #breakpoint()
        # TODO: figure out how to handle cycle/pass
        #df['cycle'] = np.zeros(np.shape(df['wse']), dtype=int)
        #df['pass'] = np.zeros(np.shape(df['wse']), dtype=int)
        # TODO: figure out how to handle area_total
        #df['area_total'] = np.array(df['width']).copy()
        # TODO: figure out what to do with p_dist_out and p_length
        #breakpoint()
        if df is not None:
            df['area_total'] = np.array(df['width']) + np.nan
            df['area_tot_u'] = np.array(df['width']) + np.nan
            df['p_dist_out'] = np.array(df['width']) * 0
            df['p_length'] = np.array(df['width']) * 0
    #
    df = manage_fields(
        df,
        use_wse_sm=cfg[section]['use_wse_sm'],
        qual_filter=cfg[section]['qual_filter'],
        dark_thresh=cfg[section]['dark_thresh'])
    #breakpoint()
    return df

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
    for reach_str, up_reach, down_reach in zip(
            reaches_str, up_reaches, down_reaches):
        up_good = True
        if up_reach > 0:
            up_good = up_reach in d_up[reach_str]
        down_good = True
        if down_reach > 0:
            down_good = down_reach in d_down[reach_str]
        good = up_good and down_good
    return good


def make_stretch_stack(
        stretch_name,
        stretch_reaches,
        swot_node_df,
        sword_node_df,
        d_up,
        d_down):
    """
    create and populate a StretchStack object from node dataframes and SWORD
    stretch_reaches = list of connected reaches along a river
    swot_node_df    = the swot node data (multiple cycles)
    sword_node_df, d_up, d_down   = SWORD info as output by io.read_SWORD()
    """
    # Init the potentially multireach stretch stack 
    data = rivscale.products.stretch_stack.StretchStack()
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
    sword_keys = ['node_length', 'dist_out', 'node_id', 'local_node_id','p_lat','p_lon']
    extra_keys = sword_keys + ['time_id', 'granule_id']
    time_ids = np.sort(np.unique(np.floor(
        swot_node_df['time']/TIME_ID_QUANTIZATION)))
    # check that the times are not too close..smach time indices within 1
    close = np.abs(np.diff(time_ids,append=time_ids[-1]+5)<2)
    time_ids = np.sort(np.array(list(set(time_ids) - set(time_ids[close]))))
    #
    stretch_data = init_dict_from_keys(keys + extra_keys)
    # go through each reach and stack the various items
    for reach in stretch_reaches:
        this_df = swot_node_df[
            swot_node_df['reach_id']==reach].sort_values([
                'local_node_id', 'time'])
        local_node_ids = np.unique(this_df['local_node_id'])
        this_sword_df = sword_node_df[
            sword_node_df['reach_id']==reach].sort_values([
                'dist_out','node_id'])
        if 'local_node_id' not in this_sword_df.keys():
            this_sword_df['local_node_id'] = rivscale.misc.node_id_to_local_node_id(
                this_sword_df['node_id'])
        nodes = this_sword_df['local_node_id']
        node_ids = this_sword_df['node_id']
        dist_out = this_sword_df['dist_out']
        signal = init_dict_from_keys(keys + extra_keys)
        # go through the time/cycles
        for t_id in time_ids:
            #that_df = this_df[np.floor(
            #    this_df['time']/TIME_ID_QUANTIZATION)==t_id]
            time_mask = np.logical_or.reduce((
                    np.floor(this_df['time']/TIME_ID_QUANTIZATION)==t_id+1,
                    np.floor(this_df['time']/TIME_ID_QUANTIZATION)==t_id,
                    np.floor(this_df['time']/TIME_ID_QUANTIZATION)==t_id-1))
            that_df = this_df[time_mask]
            this_nodes = np.array(that_df['local_node_id'])
            for key in keys:
                this_key = np.array(that_df[key])
                full_key = np.ones(np.shape(nodes)) + np.nan
                for n,kk in zip(this_nodes, this_key):
                    if kk == '':
                        full_key[nodes==n] = np.nan # not sure why we need to handle this...
                    else:
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
    # make along dist from node lengths
    dst = np.cumsum(np.array(data['node_length']))
    data['along_dist'] = np.max(dst) - dst # downstream dist
    # regenreate the area_total from width if all nan
    # (e.g., like dawg format that doesnt save area_total)
    #breakpoint()
    if np.nansum(data['area_total'])==0:
        node_len = np.broadcast_to(
            data['node_length'], np.shape(data['width'].T)).T
        data['area_total'] = data['width'] * node_len
    if np.nansum(data['area_tot_u'])==0:
        node_len = np.broadcast_to(
            data['node_length'], np.shape(data['width'].T)).T
        data['area_tot_u'] = data['width_u'] * node_len
    return data

