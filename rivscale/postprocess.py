'''
Copyright 2026, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

This is a place for code that repackages, combines, and/or accumulates the
output producs into dataframes/databases.
'''
import numpy as np
import rivscale.products.along_stretch
import os.path
import pandas as pd
import sqlite3

def load_combined_estimate_products(basedir, reach_id, crop_to_reach=True):
    """
    create SWORD-like dataframes from the various per-node products
    for a single-reach/stretch
    """
    # do the along_stats data
    try:
        fle = os.path.join(basedir, f'{reach_id}_dark_stats.nc')
        dark_df = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
            fle).to_dataframe(crop_to_reach=crop_to_reach)
    except FileNotFoundError as e:
        print("NO DARK STATS FILE:",e)
        dark_df = None
    #
    try:
        fle = os.path.join(basedir, f'{reach_id}_wse_stats.nc')
        wse_df = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
            fle).to_dataframe(crop_to_reach=crop_to_reach)
    except FileNotFoundError as e:
        print("NO WSE STATS FILE: ",e)
        wse_df = None
    #
    try:
        fle = os.path.join(basedir, f'{reach_id}_width_stats.nc')
        width_df = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
            fle).to_dataframe(crop_to_reach=crop_to_reach)
    except FileNotFoundError as e:
        print("NO WIDTH STATS FILE:",e)
        width_df = None
    # TODO: flow_state, height_width_array etc
    # merge the dataframes on node_id 
    out_df = None
    for this_df in [dark_df, wse_df, width_df]:
        if this_df is not None:
            if out_df is None:
                out_df = this_df
            else:
                out_df = out_df.merge(this_df, on='node_id', how='outer')
    return out_df

def load_combined_reconstruct_products(basedir, reach_id, crop_to_reach=True):
    """
    create SWORD-like dataframes from the various per-node products
    for a single-reach/stretch
    """
    # get the width correction from stretch_stack_corr
    try:
        fle = os.path.join(basedir, f'{reach_id}_stretch_stack_corr.nc')
        stack_df = rivscale.products.stretch_stack.StretchStack.from_ncfile(
            fle).to_dataframe(crop_to_reach=crop_to_reach)
    except FileNotFoundError as e:
        print("NO STRETCH_STACK_CORR FILE:",e)
        stack_df = None
    #
    try:
        fle = os.path.join(basedir, f'{reach_id}_bayes.nc')
        bayes_df = rivscale.products.bayes_data.BayesData.from_ncfile(
            fle).to_dataframe(crop_to_reach=crop_to_reach)
    except FileNotFoundError as e:
        print("NO STRETCH_STACK_CORR FILE:",e)
        bayes_df = None
    
    # merge the dataframes on node_id
    out_df = None
    for this_df in [stack_df, bayes_df]:
        if this_df is not None:
            if out_df is None:
                out_df = this_df
            else:
                out_df = out_df.merge(this_df, on=['node_id','granule_id'],
                        how='outer')
    return out_df

def accumulate_processed_df(
        config,
        kind='estimate',
        outfile=None,
        table_name=None):
    """
    loop over all output/created files from a particular config then load and 
    accumulate the SWORD-like product dataframes
    """
    #
    cfg_run = rivscale.misc.CfgParser()
    cfg_run.read(config)
    ####
    # get the list of stretches to process
    ###
    stretch_list = rivscale.misc.get_stretch_list_from_subset_cfg(
        cfg_run, None)
    #print(f'preparing to process stretches: {stretch_list}')
    # get the stretch_definition rows for the stretch_list
    df_stretches = pd.read_csv(
        cfg_run['main']['stretch_definition_file'],
        usecols=stretch_list)
    # check if there are no stretches to process
    N = len(df_stretches.keys())
    if len(df_stretches.keys())==0:
        print('no files to process')
    # go through each stretch and process
    out_df = None
    if outfile is not None:
        conn = sqlite3.connect(outfile)
        if_exists = 'replace'
    for i,key in enumerate(df_stretches.keys()):
        #this_start = time.time()
        #stretch_reaches = np.array(
        #    df_stretches[df_stretches[key]>0][key]).astype(int)
        if np.mod(i, 10)==0:
            print("postprocessing {} of {}, stretch: {}".format(
                i, N, key))
        ###
        # create a one-stretch config
        ####
        this_cfg = rivscale.estimate.make_single_stretch_config(
            cfg_run, key, section=kind)
        this_outpath = this_cfg['main']['out_path']
        if kind == 'estimate':
            this_df = load_combined_estimate_products(this_outpath, key)
        else:
            this_df = load_combined_reconstruct_products(this_outpath, key)
        if this_df is None:
            print(f'no files for stretch {key}')
            continue
        if outfile is None:
            # accumulate in memory
            if out_df is None:
                out_df = this_df
            else:
                out_df = pd.concat([out_df, this_df], ignore_index=True)
        else:
            # overwrite the table on the first go
            # otherwise append to output file
            this_df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            if if_exists=='replace':
                if_exists = 'append' 

    if outfile is not None:
        conn.close()
    #
    return out_df

