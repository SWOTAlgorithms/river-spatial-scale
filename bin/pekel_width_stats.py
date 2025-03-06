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

def get_pekel_dfs(reaches, pekel_dir):
    df_list = []
    for r in reaches:
        glob_str = os.path.join(
            pekel_dir,'*','Multitemporal_Node','{}_*.csv'.format(r))
        this_fle = glob.glob(glob_str)
        if len(this_fle)>0:
            df_list.append(pd.read_csv(this_fle[0]))
    return df_list
    

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
    stretch_list0 = [
        '{}'.format(t) for t in '{}'.format(
            cfg['main']['stretch_subset']).split()]
    # make the output dir if needed
    indir0 = os.path.join(cfg['main']['stretch_stack_in_path'],cfg['main']['orbit'])
    outdir0 = os.path.join(cfg['main']['out_path'],cfg['main']['orbit'])
    pekeldir = cfg['main']['pekel_in_path']
    stretch_files = []
    for stretch in stretch_list0:
        # get all reaches in basins smaller than stretch
        glob_str = os.path.join(
            pekeldir, '*','Multitemporal_Node','{}*.csv'.format(stretch))
        this_files = glob.glob(glob_str)
        stretch_files = stretch_files + this_files
    stretch_list = []
    for fle in stretch_files:
        head, tail = os.path.split(fle)
        stretch_list.append(tail.split('_')[0])
    df_stretches = pd.read_csv(
        cfg['main']['stretch_file'],
        usecols=stretch_list)
    #if not os.path.exists(outdir):
    #    os.makedirs(outdir)
    # go through each stretch and process it
    N = len(df_stretches.keys())
    #stretch_list = []
    for i,key in enumerate(df_stretches.keys()):
        outdir = os.path.join(outdir0, key, 'pekel_{}'.format(
            cfg['main']['flavor']))
        stretch_dir = os.path.join(indir0, key, 'stretch_stack_{}'.format(
            cfg['main']['stretch_stack_flavor']))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        this_start = time.time()
        stretch_reaches = np.array(
            df_stretches[df_stretches[key]>0][key]).astype(int)
        print("processing {} of {}, stretch: {}".format(
            i, N, key), ", Reaches:", stretch_reaches)
        # check if already run
        #infile_width_stats = os.path.join(outdir, '{}_width_stats.nc'.format(key))
        infile_width_stats = os.path.join(
                stretch_dir, '{}_stretch_stack.nc'.format(key))
        outfile_width_stats = os.path.join(outdir, '{}_pekel_stats.nc'.format(key))
        if not(os.path.exists(infile_width_stats)):
            print("  The input widh_stats file has not been created")
            continue
        # check if output file exists, if it does skip, unless --force set
        if (os.path.exists(outfile_width_stats) and (not args.force)):
            print("  This stretch already processed")
            continue
        # read the stretch data
        df_list = get_pekel_dfs(stretch_reaches, pekeldir)
        if len(df_list)!=len(stretch_reaches):
            print( "cannot create this stretch {}, no Pekel data".format(key))
            continue
        input_width_stats = rivscale.products.AlongStretchStats.from_ncfile(
            infile_width_stats)
        width_stats = rivscale.products.AlongStretchStats.from_pekel_df(
            cfg['pekel'],df_list, stretch_reaches, key, input_width_stats)

        if width_stats is not None:
            width_stats.to_ncfile(outfile_width_stats)
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

