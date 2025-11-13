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
import rivscale.products.stretch_stack
import rivscale.products.along_stretch
#import rivscale.products.stretch_average
import rivscale.products.height_width_array
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
    stretch_list0 = rivscale.misc.get_stretch_list_from_subset_cfg(
        cfg, None)
    """
    stretch_list0 = [
        '{}'.format(t) for t in '{}'.format(
            cfg['main']['stretch_subset']).split()]
    """
    # make the output dir if needed
    stretch_dir0 = os.path.join(
        cfg['main']['stretch_stack_in_path'],cfg['main']['orbit'])
    pekel_dir0 = None
    if 'pekel_in_path' in cfg['main'].keys():
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
        pekel_dir = None
        if pekel_dir0 is not None:
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
        #infile_pekel = None
        #if pekel_dir  is not None:
        #    infile_pekel = os.path.join(
        #        pekel_dir, '{}_pekel_stats.nc'.format(key))
        outfile_corr_stack = os.path.join(
            outdir, '{}_stretch_stack_corr.nc'.format(key))
        outfile_wse_stats = os.path.join(
            outdir, '{}_wse_stats.nc'.format(key))
        #outfile_width_stats = os.path.join(
        #    outdir, '{}_width_stats.nc'.format(key))
        #outfile_dark_stats = os.path.join(
        #    outdir, '{}_dark_stats.nc'.format(key))
        #outfile_wse_avg = os.path.join(
        #    outdir, '{}_wse_stretch_average.nc'.format(key))
        #outfile_width_avg = os.path.join(
        #    outdir, '{}_width_stretch_average.nc'.format(key))
        outfile_height_width = os.path.join(
            outdir, '{}_height_width_array.nc'.format(key))
        #outfile_bayes = os.path.join(
        #    outdir, '{}_bayes.nc'.format(key))
        ####
        # read the stretch data
        ####
        stretch_stack = rivscale.products.stretch_stack.StretchStack.from_ncfile(
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
        if (os.path.exists(outfile_corr_stack) and (not args.force)):
            print("  This along_stretch already processed")
            # read in the ones already run
            stretch_stack = rivscale.products.stretch_stack.StretchStack.from_ncfile(
                outfile_corr_stack)
        else:
            print("  Processing width_correction")
            # process it
            corr_stack = rivscale.estimate.process_width_correction(
                    cfg['width_correction'],
                    stretch_stack)
            # write output files
            #if wse_stats is not None:
            #    wse_stats.to_ncfile(outfile_wse_stats)
            if stretch_stack is not None:
                stretch_stack.to_ncfile(outfile_corr_stack)
            else:
                print(" width_correction not generated, skipping rest of processing")
                # TODO: should we check width too? but only of not using Pekel?
                continue
        # TODO: do the dark stats separately before the others?
        # check if output file exists, if it does skip, unless --force set
        if (os.path.exists(outfile_height_width) and (not args.force)):
            print("  This along_stretch already processed")
            # read in the ones already run
            
            wse_stats = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
                outfile_wse_stats)
            
            #width_stats = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
            #    outfile_width_stats)
            
            height_width_array = \
                rivscale.products.height_width_array.HeightWidthModelArray.from_ncfile(
                outfile_height_width)
        else:
            print("  Processing along_stretch")
            # process it
            filt_stack = rivscale.estimate.process_stack_filter(
                cfg['filter_stack'], stretch_stack)
            if filt_stack is None:
                print(" no data left after filtering")
                continue
            else:
                stretch_stack = filt_stack
            #TODO: should we output the filtered stack?
            wse_stats, height_width_array = \
                rivscale.estimate.process_height_width_array(
                    cfg['height_width_array'],
                    stretch_stack)
            # write output files
            if wse_stats is not None:
                wse_stats.to_ncfile(outfile_wse_stats)
            #if corr_stack is not None:
            #    corr_stack.to_ncfile(outfile_corr_stack)
            if height_width_array is not None:
                height_width_array.to_ncfile(outfile_height_width)
            if (height_width_array is None):
                print(" height_width_array not generated, skipping rest of processing")
                # TODO: should we check width too? but only of not using Pekel?
                continue
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

