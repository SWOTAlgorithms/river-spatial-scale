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

def smooth_widths(widths_in, size=11):
    #widths = stretch_stack['width'].copy()
    widths = widths_in.copy()
    w_mask = np.zeros(np.shape(widths))
    w_mask[np.isfinite(widths)] = 1
    widths[w_mask==0] = 0
    w_filt = scipy.ndimage.uniform_filter1d(widths, size, axis=0)
    w_cnt = scipy.ndimage.uniform_filter1d(w_mask, size, axis=0)
    width_filt = w_filt / w_cnt
    width_filt[w_mask==0] = np.nan
    return width_filt

def scaled_spread_outlier_rejector(arr_in, mean, spread, scale=5.0):
    """
    This method rejects outliers defined as being farther than
    some multiple of the spread away from the mean.
    The spread is an estimate of the dispersion of the distribution
    (e.g., the std or IQR etc).  Scale is how far from the mean.
    """
    arr = arr_in.copy()
    upper = mean + scale * spread
    lower = mean - scale * spread
    # handle the cases of 2D arr and 1D mean/dis
    if np.shape(arr)!=np.shape(np.array(upper)):
        upper = np.broadcast_to(upper, np.shape(arr.T)).T
        lower = np.broadcast_to(lower, np.shape(arr.T)).T
    # set outlier values to nan
    arr[arr > upper] = np.nan
    arr[arr < lower] = np.nan
    return arr

def filter_width_stretch_outliers(
        width_stretch_avg_in,
        width_stats,
        IQR_scale=5.0):
    # TODO make this a method of StretchAverage
    width_stretch_avg = width_stretch_avg_in.copy()
    width = width_stretch_avg.mean
    ref = np.nanmean(width_stats.reference)
    p = width_stats.percentile_list
    p25 = np.nanmean(width_stats.percentiles[:,p==25].squeeze())
    p75 = np.nanmean(width_stats.percentiles[:,p==75].squeeze())
    IQR = p75 - p25
    width = scaled_spread_outlier_rejector(width, ref, IQR, IQR_scale)
    width_stretch_avg.mean = width
    return width_stretch_avg

def filter_width_node_outliers(
        stretch_stack_in,
        width_stats,
        IQR_scale=5.0,
        plot=False):
    # TODO: make this a method of StretchStack
    stretch_stack = stretch_stack_in.copy()
    width = stretch_stack.width
    ref = width_stats.reference
    p = width_stats.percentile_list
    p25 = width_stats.percentiles[:,p==25].squeeze()
    p75 = width_stats.percentiles[:,p==75].squeeze()
    IQR = p75 - p25
    #
    width = scaled_spread_outlier_rejector(width, ref, IQR, IQR_scale)
    #IQR2D = np.broadcast_to(IQR, np.shape(width.T)).T
    #ref2D = np.broadcast_to(ref, np.shape(width.T)).T
    ## filter out the bad ones
    #width[width > ref2D + IQR_scale*IQR2D] = np.nan
    #width[width < ref2D - IQR_scale*IQR2D] = np.nan
    stretch_stack.width = width
    if plot:
        all_widths = stretch_stack_in.width
        kept_widths = stretch_stack.width
        outlier_mask = np.logical_and(
            np.isfinite(all_widths),
            np.isnan(kept_widths))
        dist_out = np.broadcast_to(
            stretch_stack_in.dist_out,
            np.shape(all_widths.T)).T
        plt.figure()
        plt.plot(dist_out, all_widths)
        plt.plot(
            dist_out[outlier_mask],
            all_widths[outlier_mask],'x')
        plt.plot(
            stretch_stack_in.dist_out,
            ref + IQR_scale*IQR,'k', linewidth=2)
        plt.plot(
            stretch_stack_in.dist_out,
            ref - IQR_scale*IQR,'k', linewidth=2)
    return stretch_stack

def process_stretch_average(
        stretch_stack,
        wse_stats,
        width_stats,
        wse_dark_thresh = 0.8,
        width_dark_thresh=0.2,
        width_outlier_scale=1.5,
        char_length_tau_wse = 100000,
        prior_unc_alpha_wse = 1.5,
        char_length_tau_width = 100000,
        prior_unc_alpha_width = 50, #200,
        rho_wse_width = 0.7,
        ):
    """
    # do some filtering and massaging of the data
    # populate witdh_u
    node_len = stretch_stack['area_total'] / stretch_stack['width']
    stretch_stack['width_u'] = stretch_stack['area_tot_u'] / node_len
    # make measurement uncert at least as much as signal uncert we assume
    stretch_stack['width_u'] = stretch_stack['width_u'] + 100#2*prior_unc_alpha_width
     
    # TODO: quantify amount of flagged out data?
    # filter out bad data (call it twice to get them all)
    stretch_stack = rivscale.filter.filter_bad_stretch_stack(
        stretch_stack, wse_dark_thresh, width_dark_thresh)
    stretch_stack = rivscale.filter.filter_bad_stretch_stack(
        stretch_stack, wse_dark_thresh, width_dark_thresh)
    # drop times/cycles with too little good quality data
    stretch_stack = rivscale.filter.drop_stretch_nans(stretch_stack)
    if np.shape(stretch_stack.width)[1]==0:
        print('  No SWOT data left after multitemporal filtering')
        return None
    """
    #stretch_stack = rivscale.filter.filter_bad_stretch_stack(
    #    stretch_stack, wse_dark_thresh, width_dark_thresh)
    # filter width outlier
    stretch_stack = filter_width_node_outliers(
        stretch_stack, width_stats, width_outlier_scale, plot=True)
    # filter out high dark_frac nodes
    stretch_stack.width[
        stretch_stack.dark_frac>width_dark_thresh] = np.nan
    stretch_stack.wse[
        stretch_stack.dark_frac>wse_dark_thresh] = np.nan
    # drop rows with too  little data
    reach_id = None
    reach_id = 'nope'
    stretch_stack = rivscale.filter.drop_stretch_nans(stretch_stack,
        reach_id=reach_id)
    # set up the cov params
    wse_stats.char_length_tau=char_length_tau_wse
    wse_stats.prior_unc_alpha=prior_unc_alpha_wse
    width_stats.char_length_tau=char_length_tau_width
    width_stats.prior_unc_alpha=prior_unc_alpha_width

    # get stretch average stats
    wse_stretch_avg = rivscale.products.StretchAverageStats.from_StretchStack(
        stretch_stack, signal_key='wse', along_stats=wse_stats,
        average_method='bayes_weighted',
        slope_method='bayes',
        reach_id=reach_id)
    width_stretch_avg = rivscale.products.StretchAverageStats.from_StretchStack(
        stretch_stack, signal_key='width', along_stats=width_stats,
        average_method='bayes_weighted',
        slope_method='bayes',
        reach_id=reach_id)
    # also filter stretch-outliers
    #width_stretch_avg = filter_width_stretch_outliers(
    #        width_stretch_avg, width_stats, width_outlier_scale)
    # plot the slope
    """
    # do some filtering
    stretch_stack2 = filter_width_outliers(stretch_stack, width_stats, 3)
    #stretch_stack2 = stretch_stack.copy()
    stretch_stack2.width[stretch_stack2.dark_frac>0.1] = np.nan
    width_stats2 = width_stats.copy()
    # do some width smoothing
    #stretch_stack3 = stretch_stack.copy()
    width_sm = smooth_widths(stretch_stack2.width)
    ref_sm = smooth_widths(width_stats.reference)
    #stretch_stack2.width = width_sm
    #width_stats2.reference = ref_sm
    # now get the averages
    width_stretch_avg2 = rivscale.products.StretchAverageStats.from_StretchStack(
        stretch_stack2, signal_key='width', reference=width_stats)
    """
    # plot
    stretch_stack.plot(
        wse_reference=wse_stats,
        width_reference=width_stats)
    #stretch_stack2.plot(
    #    wse_reference=wse_stats,
    #    width_reference=width_stats2)
    #stretch_stack2.plot()
    wse_stretch_avg.plot()
    width_stretch_avg.plot()
    #width_stretch_avg2.plot()
    #breakpoint()
    # play with cdf of 
    #Pg = np.nanmean(width_stats2.percentiles, axis=0)
    Pg = np.nanmean(width_stats.percentiles, axis=0)
    #Pm = np.nanmean(width_stretch_avg.percentiles, axis=0)
    #Ph = np.nanmean(wse_stretch_avg.percentiles, axis=0)
    #Pm = np.nanpercentile(width_stretch_avg2.mean,[5,25,32,50,68,75,95])
    Pm = np.nanpercentile(width_stretch_avg.mean,[5,25,32,50,68,75,95])
    Ph = np.nanpercentile(wse_stretch_avg.mean,[5,25,32,50,68,75,95])
    #breakpoint()
    plt.figure()
    plt.plot(width_stretch_avg.mean)
    plt.title('Pm')
    plt.figure()
    plt.plot(Pg, label='pekel', linewidth=2)
    plt.plot(Pm, label='swot', linewidth=2)
    #plt.plot(st_Pm, label='swot_st', linewidth=2)
    mu_g = Pg[3]
    mu_m = Pm[3]
    #sigma_g = 1/2*(Pg[4]-Pg[2])
    #sigma_m = 1/2*(Pm[4]-Pm[2])
    sigma_g = 1/1.349*(Pg[5]-Pg[1])
    sigma_m = 1/1.349*(Pm[5]-Pm[1])
    sigma_ns = [10,]#[30,]#[0, 15, 30,]
    for sigma_n in sigma_ns:
        a = sigma_g / np.sqrt(sigma_m**2-sigma_n**2)
        Pl_hat = a*(Pm - mu_m) + mu_g
        b = 1/a#sigma_g / np.sqrt(sigma_m**2-sigma_n**2)
        Pl_hat2 = b*(Pg - mu_g) + mu_m
        plt.plot(Pl_hat, label='swot_est, $\sigma_n$={}'.format(sigma_n))
        plt.plot(Pl_hat2, label='pekel_est, $\sigma_n$={}'.format(sigma_n))
    plt.legend()
    plt.grid()
    # plot the height/width stuff
    #breakpoint()
    plt.figure()
    plt.plot(
        #width_stretch_avg2.mean,
        width_stretch_avg.mean,
        wse_stretch_avg.mean,
        'o', label='swot data')
    #plt.plot(st_Pm, st_Ph, 'o', label='swot \%-iles')
    plt.plot(Pm, Ph, label='swot %-iles')
    plt.plot(Pg, Ph, label='pekel/swot %-iles')
    plt.plot(Pl_hat, Ph, label='pekel1/swot %-iles')
    plt.plot(Pl_hat2, Ph, label='pekel2/swot %-iles')
    plt.legend()
    plt.grid()
    plt.show()
    breakpoint()
    """
    breakpoint()
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
    # TODO: weighting in the averaging
    # TODO: linear fit to anomaly (weighted)
    # TODO: Bayes width using the best Pekel threshold from
    #       fully observed nodes (no-dark water). (maybe line
    #       to Pekel %, or put smooth function on Pekel %).
    """
    return wse_stretch_avg, width_stretch_avg
    

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
        outfile_wse = os.path.join(outdir, '{}_wse_stretch_average.nc'.format(key))
        outfile_width = os.path.join(outdir, '{}_width_stretch_average.nc'.format(key))
        if not(os.path.exists(infile_stretch)):
            print("  The input stretch has not been created")
            continue
        # check if output file exists, if it does skip, unless --force set
        if (os.path.exists(outfile_width) and (not args.force)):
            print("  This stretch already processed")
            continue
        # read the stretch data
        stretch_stack = rivscale.products.StretchStack.from_ncfile(infile_stretch)
        wse_stats = rivscale.products.AlongStretchStats.from_ncfile(infile_wse_stats)
        width_stats = rivscale.products.AlongStretchStats.from_ncfile(infile_width_stats)
        #
        wse_stretch_avg, width_stretch_avg = process_stretch_average(stretch_stack, wse_stats, width_stats)
        if wse_stretch_avg is not None:
            width_stretch_avg.to_ncfile(outfile_wse)
        if width_stretch_avg is not None:
            width_stretch_avg.to_ncfile(outfile_width)
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

