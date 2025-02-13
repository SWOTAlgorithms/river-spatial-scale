#!/usr/bin/env python
'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

This code processes the river streaches in an input csv file.  Note you first need
to create the swot_node_df data either bby calling the hydrocron.py script (to get
data from podaac), or by creating one from the off-lone rerun river-tiles.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rivscale
import rivscale.io
import rivscale.estimate 
import rivscale.plot
import rivscale.data
import rivscale.reconstruct
import rivscale.filter
import rivscale.products

import scipy.signal

import rivscale.filter

from errtools.misc import swot_time_to_field_time
import errtools.plots
import os.path
import configparser
import argparse
import glob
import scipy.ndimage

import warnings

import time

EXAMPLE=''


def manage_fields(df, use_wse_sm=False, dark_thresh=1.0):
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

    # drop bad data
    df = rivscale.filter.filter_node_qual(
        df, height=True, area=False, dark_thresh=dark_thresh)
    #
    if 'cycle_id' in df.keys():
        df['cycle'] = df['cycle_id']
    df['local_node_id'] = rivscale.misc.node_id_to_local_node_id(df['node_id'])
    df['wse_u'] = df['wse_r_u']
    df['dist_out'] = df['p_dist_out']
    return df

def get_swot_node_data(cfg, stretch_reaches, sword_node_df, dark_thresh=1.0):
    if cfg['data']['method'] == 'csv':
        df = pd.read_csv(cfg['data']['data_path'])
        # TODO: filter out orbit and granules we want
    elif cfg['data']['method'] == 'reach':
        # go through all the RiverSP data for the desired granules/orbit
        orbit = cfg['data']['orbit']
        pass_cont = cfg['data']['granule']
        df = None
        for reach in stretch_reaches:#df_stretches.keys():
            fle_str = os.path.join(cfg['data']['data_path'],
                '{}/{}/Multitemporal_Node/{}_Node_{}_{}.csv'.format(
                    orbit, pass_cont, reach, pass_cont, orbit))
            fles = glob.glob(fle_str)
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
    use_wse_sm = False
    sm = cfg['data']['use_wse_sm']
    if ((sm == 'True') or (sm == 'True') or (sm is True)):
        use_wse_sm = True
    #
    #if 'dark_thresh' in cfg['data'].keys():
    #dark_thresh = float(cfg['data']['dark_thresh'])
    # don't filter on dark frac here...only on qual
    df = manage_fields(df, use_wse_sm=use_wse_sm)#, dark_thresh=dark_thresh)
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Process river stretch, reach, or multireach',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('config', help='config file')
    parser.add_argument('--force', default=False, action='store_true',
        help='force rerun and overwriting of output files')
    args = parser.parse_args()
    # read in the config file
    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    # read int he SWORD file
    print('reading SWORD file')
    sword_df, sword_node_df, d_up, d_down = rivscale.io.read_SWORD(
        cfg['main']['sword_file'])
    # get the list of stretches (or multireaches)
    stretch_list0 = [
        '{}'.format(t) for t in cfg['main']['stretch_subset'].split()]
    stretch_list = []
    for stretch in stretch_list0:
        # get all reaches in basins smaller than stretch
        st = '{}'.format(stretch)
        tmp = [
            '{}'.format(r).startswith(st) for r in sword_df['reach_id']]
        reaches = np.array(sword_df['reach_id'][tmp])
        for r in reaches:
            stretch_list.append('{}'.format(r))
    df_stretches = pd.read_csv(
        cfg['main']['stretch_file'],
        usecols=stretch_list)
    # make the output dir if needed
    outdir = os.path.join(cfg['data']['out_path'],cfg['data']['orbit'])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # go through each stretch and process it
    N = len(df_stretches.keys())
    #breakpoint()
    stretch_list = []
    for i,key in enumerate(df_stretches.keys()):
        this_start = time.time()
        stretch_reaches = np.array(
            df_stretches[df_stretches[key]>0][key]).astype(int)
        print("processing {} of {}, stretch: {}".format(
            i, N, key), ", Reaches:", stretch_reaches)
        # check if already run
        outfile_stretch = os.path.join(outdir, '{}_stretch_stack.nc'.format(key))
        # check if output file exists, if it does skip, unless --force set
        if (os.path.exists(outfile_stretch) and (not args.force)):
            print("  This stretch already processed")
            continue
        # get the SWOT node data
        swot_node_df = get_swot_node_data(cfg, stretch_reaches, sword_node_df) 
        #breakpoint()
        if swot_node_df is None:
            # skip cases where we have no data
            print("  No SWOT data for this stretch")
            continue
        if len(swot_node_df) == 0:
            # skip cases where we have no data
            print("  No SWOT data remains after quality filtering")
            continue
        # create the data stack
        stretch_stack = rivscale.data.make_stretch_stack(
            key, stretch_reaches, swot_node_df, sword_node_df, d_up, d_down)
        #
        if stretch_stack is not None:
            stretch_stack.to_ncfile(outfile_stretch)
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

