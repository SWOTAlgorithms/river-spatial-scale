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

def smooth_widths(stretch_stack, size=11):
    widths = stretch_stack['width'].copy()
    w_mask = np.zeros(np.shape(widths))
    w_mask[np.isfinite(widths)] = 1
    widths[w_mask==0] = 0
    w_filt = scipy.ndimage.uniform_filter1d(widths, size, axis=0)
    w_cnt = scipy.ndimage.uniform_filter1d(w_mask, size, axis=0)
    width_filt = w_filt / w_cnt
    width_filt[w_mask==0] = np.nan
    return width_filt

def process_stretch_average(
        stretch_stack,
        wse_stats,
        width_stats,
        char_length_tau_wse = 100000,
        prior_unc_alpha_wse = 1.5,
        char_length_tau_width = 100000,
        prior_unc_alpha_width = 50, #200,
        rho_wse_width = 0.7):
    # do some filtering and massaging of the data
    # populate witdh_u
    node_len = stretch_stack['area_total'] / stretch_stack['width']
    stretch_stack['width_u'] = stretch_stack['area_tot_u'] / node_len
    # make measurement uncert at least as much as signal uncert we assume
    stretch_stack['width_u'] = stretch_stack['width_u'] + 100#2*prior_unc_alpha_width
    # first smooth widths to mitigate wedging artifacts
    #stretch_stack['width'] = smooth_widths(stretch_stack, size=5)
    # TODO: quantify amount of flagged out data?
    # filter out bad data (call it twice to get them all)
    stretch_stack = rivscale.filter.filter_bad_stretch_stack(stretch_stack)
    stretch_stack = rivscale.filter.filter_bad_stretch_stack(stretch_stack)
    # drop times/cycles with too little good quality data
    stretch_stack = rivscale.filter.drop_stretch_nans(stretch_stack)
    if np.shape(stretch_stack.width)[1]==0:
        print('  No SWOT data left after multitemporal filtering')
        return None
    # get stretch average stats
    wse_stretch_avg = rivscale.products.StretchAverageStats.from_StretchStack(
        stretch_stack, signal_key='wse', reference=wse_stats)
    width_stretch_avg = rivscale.products.StretchAverageStats.from_StretchStack(
        stretch_stack, signal_key='width', reference=width_stats)
    # compute the slope by first doing Bayes for wse-only using the reference profile
    wse_bayes = rivscale.products.BayesData()
    wse_bayes.stretch_name = stretch_stack.stretch_name
    wse_bayes.signal_key = 'wse'
    wse_bayes.signal_mean = wse_stats.reference.copy()
    time_key = 'time_id'
    wse_cov = []
    for j,cycl in enumerate(stretch_stack[time_key]):
        Rh = rivscale.reconstruct.exponential_cov(
            stretch_stack['dist_out'], # should probably use the actual node distances?
            char_length_tau=char_length_tau_wse,
            prior_unc_alpha=prior_unc_alpha_wse)
        wse_cov.append(Rh)
    wse_bayes.signal_cov = np.moveaxis(
        np.array(wse_cov), 0, -1)
    wse_bayes = rivscale.reconstruct.reconstruct_stretch(
        stretch_stack, wse_bayes, signal_key='wse', uncert_key='wse_u')
    # now estimate slope
    reach_str = [str(n)[0:-4]+str(n)[-1] for n in stretch_stack.node_id]
    inds = np.where(np.array(reach_str) == stretch_stack.stretch_name)[0]
    first_node = inds[0]
    last_node = inds[-1]
    dist_out2 = np.broadcast_to(stretch_stack.dist_out, np.shape(wse_bayes.signal.T)).T
    slope = (wse_bayes.signal[last_node,:] - wse_bayes.signal[first_node,:]) / (
        dist_out2[last_node,:] - dist_out2[first_node,:])
    breakpoint()
    

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
        outfile = os.path.join(outdir, '{}_stretch_average.nc'.format(key))
        if not(os.path.exists(infile_stretch)):
            print("  The input stretch has not been created")
            continue
        # check if output file exists, if it does skip, unless --force set
        if (os.path.exists(outfile) and (not args.force)):
            print("  This stretch already processed")
            continue
        # read the stretch data
        stretch_stack = rivscale.products.StretchStack.from_ncfile(infile_stretch)
        wse_stats = rivscale.products.AlongStretchStats.from_ncfile(infile_wse_stats)
        width_stats = rivscale.products.AlongStretchStats.from_ncfile(infile_width_stats)
        #
        stretch_avg = process_stretch_average(stretch_stack, wse_stats, width_stats)
        if stretch_avg is not None:
            stretch_avg.to_ncfile(outfile)
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

