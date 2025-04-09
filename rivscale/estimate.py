'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent Williams

'''

import numpy as np
import scipy.ndimage
import rivscale.filter

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

def process_along_stats(cfg, stretch_stack_in):
    """
    This function processes the original stack of multitemporal 
    SWOT data node-level measurements over a multi-reach streach
    to estimate the along-river statistics for WSE and width
    """
    # first handle optional config params
    if 'width_smooth_size' not in cfg.keys():
        cfg['width_smooth_size'] = 'None'
    if 'wse_ref_kernel_size' not in cfg.keys():
        cfg['wse_ref_kernel_size'] = '35'
    if 'width_ref_kernel_size' not in cfg.keys():
        cfg['width_ref_kernel_size'] = '11'
    if 'crop' not in cfg.keys():
        cfg['crop'] = 'False'
    stretch_stack = stretch_stack_in.copy()
    # create the dark stats before any filtering
    dark_stats = rivscale.products.AlongStretchStats.from_StretchStack(
        stretch_stack, signal_key='dark_frac', kernel_size=None)
    # filter the data
    stretch_stack, _ = rivscale.filter.filter_stretch_stack(
        cfg,
        stretch_stack,
        wse_stats=None,
        width_stats=None,
        plot=False)
    if np.shape(stretch_stack.width)[1]==0:
        print('  No SWOT data left after multitemporal filtering')
        return None, None, None
    ## optionally smooth the widths
    #if cfg['width_smooth_size'] is not None:
    #    stretch_stack.smooth_widths(size=cfg['width_smooth_size'])
    # compute multitemporal statistics
    wse_stats = rivscale.products.AlongStretchStats.from_StretchStack(
        stretch_stack, signal_key='wse', kernel_size=cfg['wse_ref_kernel_size'])
    width_stats = rivscale.products.AlongStretchStats.from_StretchStack(
        stretch_stack, signal_key='width', kernel_size=cfg['width_ref_kernel_size'])
    if cfg['crop']:
        # crop to reach
        wse_stats = wse_stats.crop_to_reach()
        width_stats = width_stats.crop_to_reach()
    return wse_stats, width_stats, dark_stats

def process_stretch_average(
        cfg,
        stretch_stack_in,
        wse_stats_in,
        width_stats_in):
    """
    if reach_id is a valid id in the, crop the stats to be only over that reach
    """
    # first handle optional config params
    if 'wse_dark_thresh' not in cfg.keys():
        cfg['wse_dark_thresh'] = '0.8'
    if 'wse_dark_thresh' not in cfg.keys():
        cfg['width_dark_thresh'] = '0.3'
    if 'wse_outlier_scale' not in cfg.keys():
        cfg['wse_outlier_scale'] = '5.0'
    if 'width_outlier_scale' not in cfg.keys():
        cfg['wodth_outlier_scale'] = '5.0'
    if 'width_smooth_size' not in cfg.keys():
        cfg['width_smooth_size'] = 'None'
    if 'wse_char_length_tau' not in cfg.keys():
        cfg['wse_char_length_tau'] = '100000.0'
    if 'wse_prior_unc_alpha' not in cfg.keys():
        cfg['wse_prior_unc_alpha'] = '1.5'
    if 'width_char_length_tau' not in cfg.keys():
        cfg['width_char_length_tau'] = '100000.0'
    if 'width_prior_unc_alpha' not in cfg.keys():
        cfg['width_prior_unc_alpha'] = '50.0'
    if 'crop' not in cfg.keys():
        cfg['crop'] = 'False'

    stretch_stack = stretch_stack_in.copy()
    wse_stats = wse_stats_in.copy()
    width_stats = width_stats_in.copy()
    # filter the data
    stretch_stack, reach_id = rivscale.filter.filter_stretch_stack(
        cfg,
        stretch_stack,
        wse_stats,
        width_stats,
        plot=False)
    # optionally smooth the widths
    if cfg['width_smooth_size'] is not None:
        stretch_stack.smooth_widths(size=cfg['width_smooth_size'])
    # set up the cov params
    wse_stats.char_length_tau=cfg['wse_char_length_tau']
    wse_stats.prior_unc_alpha=cfg['wse_prior_unc_alpha']
    width_stats.char_length_tau=cfg['width_char_length_tau']
    width_stats.prior_unc_alpha=cfg['width_prior_unc_alpha']

    # get stretch average stats
    wse_stretch_avg = rivscale.products.StretchAverageStats.from_StretchStack(
        stretch_stack, signal_key='wse', along_stats=wse_stats,
        average_method='bayes_simple',#'bayes_weighted',
        slope_method='bayes',
        reach_id=reach_id)
    width_stretch_avg = rivscale.products.StretchAverageStats.from_StretchStack(
        stretch_stack, signal_key='width', along_stats=width_stats,
        average_method='bayes_simple',#'bayes_weighted',
        slope_method='bayes',
        reach_id=reach_id)
    # also filter stretch-outliers
    #width_stretch_avg = rivscale.filter.filter_width_stretch_outliers(
    #        width_stretch_avg, width_stats, width_outlier_scale)
    return wse_stretch_avg, width_stretch_avg

def get_med_profile(signal, dist_out, kernel_size=35):
    """
    This function estimates a river profile from multitemporal SWOT measurements
    over a connected river stretch (potentially mutu-reach section of river
    sampled at the nodes).

    inputs:
    signal   = 2D multitemporal array of along-river data (e.g., node wse)
    dist_out = 1D along-river array of connected node distance to outlet
    
    output:
    med_filt = median profile with holes onterpolated over and spatially 
               median-filter smoothed
    """
    med = np.nanmedian(signal, axis=1)
    msk = np.isfinite(med)
    # interplate over holes (but don't extrapolate. e.g., nan-fill outside)
    #med_interp = np.interp(
    #    dist_out, dist_out[msk], med[msk], left=np.nan, right=np.nan)
    # TODO: handle remaining nans by filling with linear fit or sword prior? 
    #       just fill with max or min for now
    med_min = np.nanmin(med)
    med_max = np.nanmax(med)
    med_interp = np.interp(
        dist_out, dist_out[msk], med[msk], left=med_min, right=med_max)
    # do along-river smoothing, preserving discontinuitites
    med_filt = med_interp.copy()
    if kernel_size is not None:
        med_filt = scipy.ndimage.median_filter(
            med_interp, size=kernel_size, mode='nearest')
    return med_filt

