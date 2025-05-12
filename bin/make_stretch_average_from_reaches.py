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
import rivscale.products.stretch_average
import rivscale.products.height_width
import rivscale.misc

import scipy.signal

import rivscale.filter

import os.path
import configparser
import argparse
import glob
import scipy.ndimage

import warnings

import time

EXAMPLE=''

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
    #cfg = configparser.ConfigParser()
    #cfg.read(args.config)
    cfg = rivscale.misc.CfgParser()
    cfg.read(args.config)
    # read int he SWORD file
    print('reading SWORD file')
    sword_df, sword_node_df, d_up, d_down = rivscale.io.read_SWORD(
        cfg['main']['sword_file'])
    # get the list of stretches (or multireaches)
    stretch_list = rivscale.misc.get_stretch_list_from_subset_cfg(
        cfg, sword_df)
    #breakpoint()
    """
    stretch_list0 = [
        '{}'.format(t) for t in '{}'.format(
            cfg['main']['stretch_subset']).split()]
    stretch_list = []
    for stretch in stretch_list0:
        # get all reaches in basins smaller than stretch
        st = '{}'.format(stretch)
        tmp = [
            '{}'.format(r).startswith(st) for r in sword_df['reach_id']]
        reaches = np.array(sword_df['reach_id'][tmp])
        for r in reaches:
            stretch_list.append('{}'.format(r))
    """
    df_stretches = pd.read_csv(
        cfg['main']['stretch_file'],
        usecols=stretch_list)
    # make the output dir if needed
    outdir0 = os.path.join(cfg['main']['out_path'], cfg['main']['orbit'])
    #if not os.path.exists(outdir):
    #    os.makedirs(outdir)
    # go through each stretch and process it
    N = len(df_stretches.keys())
    #breakpoint()
    # get list of all reaches in case we need to download them all
    all_reaches = []
    for i,key in enumerate(df_stretches.keys()):
        stretch_reaches = np.array(
            df_stretches[df_stretches[key]>0][key]).astype(int)
        all_reaches = all_reaches + list(stretch_reaches)
    all_reaches = np.unique(all_reaches)
    stretch_list = []
    for i,key in enumerate(df_stretches.keys()):
        outdir = os.path.join(outdir0, key, 'reach_avg_{}'.format(
            cfg['main']['flavor']))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        this_start = time.time()
        stretch_reaches = np.array(
            df_stretches[df_stretches[key]>0][key]).astype(int)
        print("processing {} of {}, stretch: {}".format(
            i, N, key), ", Reaches:", stretch_reaches)
        # check if already run
        outfile_wse = os.path.join(
            outdir, '{}_wse_reach_average.nc'.format(key))
        outfile_width = os.path.join(
            outdir, '{}_width_reach_average.nc'.format(key))
        outfile_height_width = os.path.join(
            outdir, '{}_height_width_reach_average.nc'.format(key))

        # check if output file exists, if it does skip, unless --force set
        if (os.path.exists(outfile_width) and (not args.force)):
            print("  This stretch already processed")
            continue
        # get the SWOT node data
        #swot_reach_df = get_swot_reach_data(cfg, stretch_reaches) 
        # get the SWOT node data
        swot_reach_df = rivscale.data.get_swot_data(
            cfg,
            stretch_reaches,#sword_node_df,
            all_reaches,
            kind='Reach')
        #breakpoint()
        if swot_reach_df is None:
            # skip cases where we have no data
            print("  No SWOT data for this stretch")
            continue
        if len(swot_reach_df) == 0:
            # skip cases where we have no data
            print("  No SWOT data remains after quality filtering")
            continue
        # create the data stack
        #stretch_stack = rivscale.data.make_stretch_stack(
        #    key, stretch_reaches, swot_node_df, sword_node_df, d_up, d_down)
        #
        # create the wse_stretch_avg and width_stretch_avg objects
        wse_reach_avg = rivscale.products.stretch_average.StretchAverageStats.from_reach_df(
            swot_reach_df, key, 'wse')
        width_reach_avg = rivscale.products.stretch_average.StretchAverageStats.from_reach_df(
            swot_reach_df, key, 'width')
        #breakpoint()
        # also create the height_width object from the reaches
        height_width = None
        if (wse_reach_avg is not None) and (width_reach_avg is not None):
            if (len(wse_reach_avg.percentiles)>0) and \
                    (len(width_reach_avg.percentiles)>0):
                height_width = rivscale.products.height_width.HeightWidthModel.from_objects(
                    {},wse_reach_avg, width_reach_avg)
        #
        """
        plt.figure()
        plt.plot(width_reach_avg.mean, wse_reach_avg.mean - wse_reach_avg.mean_reference, 'o')
        x = np.linspace(0, np.max(width_reach_avg.mean), 100)
        y = height_width.sample(x)
        plt.plot(x, y)
        plt.show()
        breakpoint()
        """
        if wse_reach_avg is not None:
            wse_reach_avg.to_ncfile(outfile_wse)
        if width_reach_avg is not None:
            width_reach_avg.to_ncfile(outfile_width)
        if height_width is not None:
            height_width.to_ncfile(outfile_height_width)
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

