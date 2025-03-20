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
import rivscale.misc

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
    # handle non-strings for stretch_subset
    #cfg['main']['stretch_subset'] = '{}'.format(cfg['main']['stretch_subset'])
    stretch_list0 = [
        '{}'.format(t) for t in '{}'.format(
            cfg['main']['stretch_subset']).split()]
    # make the output dir if needed
    stretch_dir0 = os.path.join(
        cfg['main']['stretch_stack_in_path'],cfg['main']['orbit'])
    pekel_dir0 = os.path.join(
        cfg['main']['pekel_in_path'],cfg['main']['orbit'])
    outdir0 = os.path.join(cfg['main']['out_path'],cfg['main']['orbit'])
    #outdir = os.path.join(outdir0, cfg['main']['flavor'])
    stretch_files = []
    for stretch in stretch_list0:
        # get all reaches in basins smaller than stretch
        this_files = glob.glob(os.path.join(
            stretch_dir0,'{}*'.format(stretch),
            'stretch_stack_*', '{}*_stretch_stack.nc'.format(stretch)))
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
    #breakpoint()
    if len(df_stretches.keys())==0:
        print('no files to process')
    #stretch_list = []
    for i,key in enumerate(df_stretches.keys()):
        stretch_dir = os.path.join(
            stretch_dir0, key, 'stretch_stack_{}'.format(
                cfg['main']['stretch_stack_flavor']))
        pekel_dir = os.path.join(
            pekel_dir0, key, 'pekel_{}'.format(
                cfg['main']['pekel_flavor']))
        outdir = os.path.join(outdir0, key, cfg['main']['flavor'])
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        this_start = time.time()
        stretch_reaches = np.array(
            df_stretches[df_stretches[key]>0][key]).astype(int)
        print("processing {} of {}, stretch: {}".format(
            i, N, key), ", Reaches:", stretch_reaches)
        # check if already run
        ####
        # setup input/output files
        ####
        infile_stretch = os.path.join(
            stretch_dir, '{}_stretch_stack.nc'.format(key))
        infile_pekel = os.path.join(
            pekel_dir, '{}_pekel_stats.nc'.format(key))
        outfile_wse_stats = os.path.join(
            outdir, '{}_wse_stats.nc'.format(key))
        outfile_width_stats = os.path.join(
            outdir, '{}_width_stats.nc'.format(key))
        outfile_dark_stats = os.path.join(
            outdir, '{}_dark_stats.nc'.format(key))
        outfile_wse_avg = os.path.join(
            outdir, '{}_wse_stretch_average.nc'.format(key))
        outfile_width_avg = os.path.join(
            outdir, '{}_width_stretch_average.nc'.format(key))
        outfile_height_width = os.path.join(
            outdir, '{}_height_width.nc'.format(key))
        outfile_bayes = os.path.join(
            outdir, '{}_bayes.nc'.format(key))
        ####
        # read the stretch data
        ####
        stretch_stack = rivscale.products.StretchStack.from_ncfile(
            infile_stretch)
        ####
        # first process along stats
        ####
        if not(os.path.exists(infile_stretch)):
            print("  The input stretch has not been created")
            continue
        if len(stretch_stack.node_id) < 50:
            print("  This is a short stretch, dont process it?")
            continue
        # check if output file exists, if it does skip, unless --force set
        if (os.path.exists(outfile_width_stats) and (not args.force)):
            print("  This along_stretch already processed")
            # read in the ones already run
            wse_stats = rivscale.products.AlongStretchStats.from_ncfile(
                outfile_wse_stats)
            width_stats = rivscale.products.AlongStretchStats.from_ncfile(
                outfile_width_stats)
            dark_stats = rivscale.products.AlongStretchStats.from_ncfile(
                outfile_dark_stats)
        else:
            print("  Processing along_stretch")
            # process it
            wse_stats, width_stats, dark_stats = \
                rivscale.estimate.process_along_stats(
                    cfg['along_stats'], stretch_stack)
            # write output files
            if wse_stats is not None:
                wse_stats.to_ncfile(outfile_wse_stats)
            if width_stats is not None:
                width_stats.to_ncfile(outfile_width_stats)
            if dark_stats is not None:
                dark_stats.to_ncfile(outfile_dark_stats)
            if (wse_stats is None):
                print(" wse_stats not generated, skipping rest of processing")
                # TODO: should we check width too? but only of not using Pekel?
                continue
        ####
        # handle optionally using Pekel
        # stop here if commanded but pekel files dont exist
        ####
        pekel_stats = None
        sa_cfg = cfg['stretch_average']
        hw_cfg = cfg['height_width']
        bayes_cfg = cfg['reconstruct']
        if 'use_pekel' not in sa_cfg.keys():
            sa_cfg['use_pekel'] = 'False'
        if 'use_pekel' not in hw_cfg.keys():
            hw_cfg['use_pekel'] = 'False'
        if 'use_pekel' not in bayes_cfg.keys():
            bayes_cfg['use_pekel'] = 'False'
        if (sa_cfg['use_pekel']) or (hw_cfg['use_pekel']) or (
                bayes_cfg['use_pekel']):
            if not(os.path.exists(infile_pekel)):
                print("  The input pekel file has not been created")
                continue
            else:
                pekel_stats = rivscale.products.AlongStretchStats.from_ncfile(
                    infile_pekel)
        ####
        # now process stretch_average
        ####
        # check if output file exists, if it does skip, unless --force set
        if (os.path.exists(outfile_width_avg) and (not args.force)):
            print("  This stretch_average already processed")
            wse_avg = rivscale.products.StretchAverageStats.from_ncfile(
                outfile_wse_avg)
            width_avg = rivscale.products.StretchAverageStats.from_ncfile(
                outfile_width_avg)
        else:
            print("  Processing stretch_average")
            if sa_cfg['use_pekel']:
                this_width_stats = pekel_stats
            else:
                this_width_stats = width_stats.copy()
            wse_avg, width_avg = rivscale.estimate.process_stretch_average(
                sa_cfg, stretch_stack, wse_stats, this_width_stats)
            if wse_avg is not None:
                wse_avg.to_ncfile(outfile_wse_avg)
            if width_avg is not None:
                width_avg.to_ncfile(outfile_width_avg)
        ####
        # now create the height_width estimate
        ####
        # check if output file exists, if it does skip, unless --force set
        if (os.path.exists(outfile_height_width) and (not args.force)):
            print("  This height_width already processed")
            height_width = rivscale.products.HeightWidthModel.from_ncfile(
                outfile_height_width)
        else:
            print("  Processing height_width")
            if hw_cfg['use_pekel']:
                this_width_stats = pekel_stats
            else:
                this_width_stats = width_stats.copy()
            height_width = rivscale.products.HeightWidthModel.from_objects(
                cfg['height_width'],
                wse_avg,
                width_avg,
                this_width_stats)
            if height_width is not None:
                height_width.to_ncfile(outfile_height_width)
        ####
        # now do Bayes reconstruction
        ####
        if (os.path.exists(outfile_bayes) and (not args.force)):
            print("  This bayes already processed")
        else:
            print("  Processing bayes")
            if bayes_cfg['use_pekel']:
                this_width_stats = pekel_stats
            else:
                this_width_stats = width_stats.copy()
            bayes = rivscale.reconstruct.process_bayes_reconstruction(
                cfg['reconstruct'],
                stretch_stack,
                wse_stats,
                this_width_stats,
                height_width)
            if bayes is not None:
                bayes.to_ncfile(outfile_bayes)
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

