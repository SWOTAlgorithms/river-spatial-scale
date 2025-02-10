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

def smooth_widths(stretch_data, size=11):
    widths = stretch_data['width'].copy()
    w_mask = np.zeros(np.shape(widths))
    w_mask[np.isfinite(widths)] = 1
    widths[w_mask==0] = 0
    w_filt = scipy.ndimage.uniform_filter1d(widths, size, axis=0)
    w_cnt = scipy.ndimage.uniform_filter1d(w_mask, size, axis=0)
    width_filt = w_filt / w_cnt
    width_filt[w_mask==0] = np.nan
    return width_filt

def process_stretch(
        stretch_reaches,
        swot_node_df,
        sword_node_df,
        d_up,
        d_down,
        char_length_tau_wse = 100000,
        prior_unc_alpha_wse = 1.5,
        char_length_tau_width = 100000,
        prior_unc_alpha_width = 50, #200,
        rho_wse_width = 0.7):
    """
    This function processes the original stack of multitemporal 
    SWOT data node-level measurements over a multi-reach streach.
    The following operations are performed:
        1) generating the stacked stretch_stack object from the measurment dataframe
        2) data quality filtering of WSE and width
        3) estimating of multitemproal statistics (e.g., median, percentiles etc)
        4) estimating a reference profile for both WSE and width
        5) estimating the Bayes reconstructed measurements for each pass
           measurement profile. TODO: specify options handling
    """
    # make the stretch multitemporal stack object
    #breakpoint()
    stretch_data = rivscale.data.make_stretch_stack(
        stretch_reaches, swot_node_df, sword_node_df, d_up, d_down)
    # populate witdh_u
    node_len = stretch_data['area_total'] / stretch_data['width']
    stretch_data['width_u'] = stretch_data['area_tot_u'] / node_len
    # make measurement uncert at least as much as signal uncert we assume
    stretch_data['width_u'] = stretch_data['width_u'] + 100#2*prior_unc_alpha_width
    # first smooth widths to mitigate wedging artifacts
    #stretch_data['width'] = smooth_widths(stretch_data, size=5)
    # TODO: quantify amount of flagged out data?
    # filter out bad data (call it twice to get them all)
    stretch_data = rivscale.filter.filter_bad_stretch_data(stretch_data)
    stretch_data = rivscale.filter.filter_bad_stretch_data(stretch_data)
    # drop times/cycles with too little good quality data
    stretch_data = rivscale.filter.drop_stretch_nans(stretch_data)
    if np.shape(stretch_data.width)[1]==0:
        print('  No SWOT data left after multitemporal filtering')
        return None
    # compute statistics
    stretch_data = rivscale.estimate.get_stretch_stats(
        stretch_data, signal_key='wse')
    stretch_data = rivscale.estimate.get_stretch_stats(
        stretch_data, signal_key='width')
    stretch_data = rivscale.estimate.get_stretch_stats(
        stretch_data, signal_key='dark_frac')
    # compute the reference profiles
    #breakpoint()
    stretch_data['wse_reference'] = rivscale.estimate.get_med_profile(
        stretch_data['wse'], stretch_data['dist_out'])
    stretch_data['width_reference'] = rivscale.estimate.get_med_profile(
        stretch_data['width'], stretch_data['dist_out'], kernel_size=11)#, kernel_size=1)# don't smooth width
    # TODO: enable estimation of char_length_tau and prior_unc_alpha from data
    # get the reach-level averages
    stretch_data = rivscale.reconstruct.reach_average(stretch_data)
    # fit the curve to reach-level averages
    stretch_data = rivscale.reconstruct.get_height_width_fit(stretch_data)
    # set up the bayes estimator signal covariance
    # first create signal covariance (possibly different for each line because of height/width model)
    time_key = 'time_id'
    # go through each time/cycle observation in the stack 
    wse_cov = []
    width_cov = []
    wse_width_cov = []
    for j,cycl in enumerate(stretch_data[time_key]):
        this_rho_wse_width = rho_wse_width
        # constrain the height and width std magnitudes using the h/w-model
        dw_dh = rivscale.reconstruct.get_dw_dh_from_model(stretch_data, j)
        this_prior_unc_alpha_width = dw_dh * prior_unc_alpha_wse
        # handle bad h/w-fits
        if dw_dh < 1e-8:
            # use default and do not impose correlation
            this_prior_unc_alpha_width = prior_unc_alpha_width
            this_rho_wse_width = 0
        Rh = rivscale.reconstruct.exponential_cov(
            stretch_data['dist_out'],
            char_length_tau=char_length_tau_wse,
            prior_unc_alpha=prior_unc_alpha_wse)
        Rw = rivscale.reconstruct.exponential_cov(
            stretch_data['dist_out'],
            char_length_tau=char_length_tau_width,
            prior_unc_alpha=this_prior_unc_alpha_width)
        Rhw = this_rho_wse_width * np.real(scipy.linalg.sqrtm(Rh) @ scipy.linalg.sqrtm(Rw.T))
        wse_cov.append(Rh)
        width_cov.append(Rw)
        wse_width_cov.append(Rhw)
        #wse_width_cov.append(
        #        this_rho_wse_width * prior_unc_alpha_wse * this_prior_unc_alpha_width * np.eye(
        #            len(stretch_data['wse_reference'])))
    # update the object
    stretch_data['wse_cov'] = np.moveaxis(
        np.array(wse_cov), 0, -1)
    stretch_data['width_cov'] = np.moveaxis(
        np.array(width_cov), 0, -1)
    stretch_data['wse_width_cov'] = np.moveaxis(
        np.array(wse_width_cov), 0, -1)
    # now do the bayes reconstruction
    stretch_data = rivscale.reconstruct.joint_reconstruct_stretch(stretch_data)
    #stretch_data = rivscale.reconstruct.reconstruct_stretch(
    #    stretch_data, signal_key='wse', uncert_key='wse_u')
    #stretch_data = rivscale.reconstruct.reconstruct_stretch(
    #    stretch_data, signal_key='width', uncert_key='width_u')
    
    return stretch_data

def manage_fields(df, use_wse_sm=False, dark_thresh=0.8):
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

def get_swot_node_data(cfg, stretch_reaches, sword_node_df, dark_thresh=0.8):
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
    dark_thresh = float(cfg['data']['dark_thresh'])
    df = manage_fields(df, use_wse_sm=use_wse_sm, dark_thresh=dark_thresh)
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
        outfile = os.path.join(outdir, '{}_stretch.nc'.format(key))
        # check if output file exists, if it does skip, unless --force set
        if (os.path.exists(outfile) and (not args.force)):
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
        # process the stretch
        stretch_data = process_stretch(
            stretch_reaches, swot_node_df, sword_node_df, d_up, d_down)
        # write out the data to ncfile
        #outfile = os.path.join(outdir, '{}_stretch.nc'.format(key))
        if stretch_data is not None:
            stretch_data.to_ncfile(outfile)
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

