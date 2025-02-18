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

def process_bayes_reconstruction(
        stretch_stack,
        wse_stats,
        width_stats,
        height_width,
        wse_dark_thresh=0.8,
        width_dark_thresh=0.2,
        width_outlier_scale=10,
        rho_wse_width=0.999
        ):
    #
    # populate witdh_u
    node_len = stretch_stack['area_total'] / stretch_stack['width']
    stretch_stack['width_u'] = stretch_stack['area_tot_u'] / node_len
    # make measurement uncert at least as much as signal uncert we assume
    #stretch_stack['width_u'] = stretch_stack['width_u'] + 500#2*prior_unc_alpha_width
    # filter out bad data
    stretch_stack = rivscale.filter.filter_width_node_outliers(
        stretch_stack, width_stats, width_outlier_scale, plot=True)
    # filter out high dark_frac nodes
    stretch_stack.width[
        stretch_stack.dark_frac>width_dark_thresh] = np.nan
    stretch_stack.wse[
        stretch_stack.dark_frac>wse_dark_thresh] = np.nan
    # drop rows with too  little data
    #reach_id = None
    reach_id = 'nope'
    stretch_stack = rivscale.filter.drop_stretch_nans(stretch_stack,
        reach_id=reach_id)
    # now do the reconstruction
    joint_bayes = rivscale.products.BayesData.joint(
        stretch_stack,
        wse_stats,
        width_stats,
        height_width,
        rho_wse_width=rho_wse_width)
    bayes_wse, bayes_width, bayes_wse_width_post_cov = joint_bayes.unpack_joint()
    stretch_stack.plot()
    plt.figure()
    plt.plot(stretch_stack.dist_out, bayes_wse.signal)
    plt.figure()
    plt.plot(stretch_stack.dist_out, bayes_width.signal)
    ref2 = np.broadcast_to(wse_stats.reference, np.shape(bayes_width.signal.T)).T
    ref2_w = np.broadcast_to(width_stats.reference, np.shape(bayes_width.signal.T)).T
    plt.figure()
    plt.plot(bayes_width.signal - ref2_w, bayes_wse.signal - ref2,'o')
    plt.show()
    breakpoint()

def process_height_width(
        wse_stretch_avg,
        width_stretch_avg,
        width_stats=None,
        ):
    # TODO: implement:
    #       1) piece-wise fit
    #       2) percentile function, with piecewise params result?
    #wse_stretch_avg.plot()
    #width_stretch_avg.plot()
    height_width = rivscale.products.HeightWidthModel.from_objects(
        wse_stretch_avg,
        width_stretch_avg,
        width_stats)
    x = np.linspace(0,np.max(width_stretch_avg.mean),100)
    y = height_width.sample(x, x_key='width')
    plt.figure()
    plt.plot(
        width_stretch_avg.mean,
        wse_stretch_avg.mean - wse_stretch_avg.mean_reference,
        'o')
    plt.plot(x,y)
    plt.grid()
    plt.show()
    return height_width
    #width_stretch_avg2.plot()
    #breakpoint()
    # play with cdf of 
    Pg = np.nanmean(width_stats.percentiles, axis=0)
    Pm = np.nanpercentile(width_stretch_avg.mean,[5,25,32,50,68,75,95])
    Ph = np.nanpercentile(wse_stretch_avg.mean,[5,25,32,50,68,75,95])
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
    #
    height_width = None
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
        bayes = process_bayes_reconstruction(
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

