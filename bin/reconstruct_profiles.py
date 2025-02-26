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
    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    stretch_list0 = [
        '{}'.format(t) for t in cfg['main']['stretch_subset'].split()]
    # make the output dir if needed
    outdir = os.path.join(cfg['data']['out_path'],cfg['data']['orbit'])
    stretch_files = []
    for stretch in stretch_list0:
        # get all reaches in basins smaller than stretch
        this_files = glob.glob(os.path.join(
            outdir, '{}*_stretch_stack.nc'.format(stretch)))
        stretch_files = stretch_files + this_files
    stretch_list = []
    for fle in stretch_files:
        head, tail = os.path.split(fle)
        stretch_list.append(tail.split('_')[0])
    df_stretches = pd.read_csv(
        cfg['main']['stretch_file'],
        usecols=stretch_list)
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
        infile_stretch = os.path.join(outdir, '{}_stretch_stack.nc'.format(key))
        infile_wse_stats = os.path.join(outdir, '{}_wse_stats.nc'.format(key))
        # TODO: prefer to use Pekel for width?
        #infile_width_stats = os.path.join(outdir, '{}_width_stats.nc'.format(key))
        infile_width_stats = os.path.join(outdir, '{}_pekel_stats.nc'.format(key))
        infile_wse = os.path.join(outdir, '{}_wse_stretch_average.nc'.format(key))
        infile_width = os.path.join(outdir, '{}_width_stretch_average.nc'.format(key))
        infile_height_width = os.path.join(outdir, '{}_height_width.nc'.format(key))
        outfile_bayes = os.path.join(outdir, '{}_bayes.nc'.format(key))
        if not(os.path.exists(infile_wse)):
            print("  The input stretch average wse has not been created")
            continue
        if not(os.path.exists(infile_width)):
            print("  The input stretch average width has not been created")
            continue
        # check if output file exists, if it does skip, unless --force set
        if (os.path.exists(outfile_bayes) and (not args.force)):
            print("  This stretch already processed")
            continue
        # read the stretch data
        stretch_stack = rivscale.products.StretchStack.from_ncfile(infile_stretch)
        wse_stats = rivscale.products.AlongStretchStats.from_ncfile(infile_wse_stats)
        width_stats = rivscale.products.AlongStretchStats.from_ncfile(infile_width_stats)
        #wse_stretch_avg = rivscale.products.StretchAverageStats.from_ncfile(
        #    infile_wse)
        #width_stretch_avg = rivscale.products.StretchAverageStats.from_ncfile(
        #    infile_width)
        height_width = rivscale.products.HeightWidthModel.from_ncfile(
            infile_height_width)
        bayes = rivscale.reconstruct.process_bayes_reconstruction(
                stretch_stack,
                wse_stats,
                width_stats,
                height_width)
        if bayes is not None:
            bayes.to_ncfile(outfile_bayes)
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

