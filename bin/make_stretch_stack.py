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
    df_stretches = pd.read_csv(
        cfg['main']['stretch_file'],
        usecols=stretch_list)
    # make the output dir if needed
    outdir0 = os.path.join(cfg['main']['out_path'],cfg['main']['orbit'])
    #if not os.path.exists(outdir):
    #    os.makedirs(outdir)
    # go through each stretch and process it
    N = len(df_stretches.keys())
    #breakpoint()
    stretch_list = []
    for i,key in enumerate(df_stretches.keys()):
        outdir = os.path.join(
            outdir0,
            key,
            'stretch_stack_{}'.format(cfg['main']['flavor']))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
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
        swot_node_df = rivscale.data.get_swot_data(
            cfg,
            stretch_reaches,#sword_node_df,
            kind='Node',
            force=args.force)
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
        # TODO: now qual filter?
        #
        if stretch_stack is not None:
            stretch_stack.to_ncfile(outfile_stretch)
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

