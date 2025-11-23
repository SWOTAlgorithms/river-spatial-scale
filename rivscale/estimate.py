'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent Williams

'''

import numpy as np
import scipy.ndimage
import rivscale.filter
import rivscale.products.along_stretch
import rivscale.products.stretch_average
import rivscale.special

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

def process_dark_stats(cfg,stretch_stack_in):
    """
    This function processes the original stack of multitemporal 
    SWOT data node-level measurements over a multi-reach streach
    to estimate the along-river statistics for dark water
    """
    # first handle optional config params
    if 'crop' not in cfg.keys():
        cfg['crop'] = 'False'
    stretch_stack = stretch_stack_in.copy()
    # create the dark stats before any filtering
    dark_stats = rivscale.products.along_stretch.AlongStretchStats.from_StretchStack(
        stretch_stack, signal_key='dark_frac', kernel_size=None)
    if cfg['crop']:
        # crop to reach
        dark_stats = dark_stats.crop_to_reach()
    return dark_stats

def process_smooth_widths(cfg, stretch_stack_in):
    """
    This function processes the original stack of multitemporal 
    SWOT data node-level measurements over a multi-reach streach
    to estimate the along-river statistics for WSE and width
    """
    # first handle optional config params
    if 'width_smooth_size' not in cfg.keys():
        cfg['width_smooth_size'] = 'None'
    if 'crop' not in cfg.keys():
        cfg['crop'] = 'False'
    stretch_stack = stretch_stack_in.copy()
    ## optionally smooth the widths
    if cfg['width_smooth_size'] is not None:
        stretch_stack.smooth_widths(size=cfg['width_smooth_size'])
    return stretch_stack

def process_filter_stack(cfg, stretch_stack_in):
    """
    This function processes the original stack of multitemporal 
    SWOT data node-level measurements over a multi-reach streach
    to filter out/exclude poor data based on quality and outlier rejection
    """
    # first handle optional config params
    if 'width_smooth_size' not in cfg.keys():
        cfg['width_smooth_size'] = 'None'
    if 'wse_ref_kernel_size' not in cfg.keys():
        cfg['wse_ref_kernel_size'] = '35'
    if 'width_ref_kernel_size' not in cfg.keys():
        cfg['width_ref_kernel_size'] = 'None'#'11'
    if 'crop' not in cfg.keys():
        cfg['crop'] = 'False'
    stretch_stack = stretch_stack_in.copy()
    # filter the data
    stretch_stack, _ = rivscale.filter.filter_stretch_stack(
            cfg,
            stretch_stack,
            wse_stats=None,
            width_stats=None,
            plot=False)
    if np.shape(stretch_stack.width)[1]==0:
        print('  No SWOT data left after multitemporal filtering')
        return None
    if cfg['crop']:
        # crop to reach
        stretch_stack = stretch_stack.crop_to_reach()
    return stretch_stack

def process_along_stats(cfg, stretch_stack_in, filterit=False):
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
        cfg['width_ref_kernel_size'] = 'None'#'11'
    if 'crop' not in cfg.keys():
        cfg['crop'] = 'False'
    stretch_stack = stretch_stack_in.copy()
    # filter the data
    if filterit:
        stretch_stack, _ = rivscale.filter.filter_stretch_stack(
            cfg,
            stretch_stack,
            wse_stats=None,
            width_stats=None,
            plot=False)
    if np.shape(stretch_stack.width)[1]==0:
        print('  No SWOT data left after multitemporal filtering')
        return None, None
    ## optionally smooth the widths
    if cfg['width_smooth_size'] is not None:
        stretch_stack.smooth_widths(size=cfg['width_smooth_size'])
    # compute multitemporal statistics
    wse_stats = rivscale.products.along_stretch.AlongStretchStats.from_StretchStack(
        stretch_stack, signal_key='wse', kernel_size=cfg['wse_ref_kernel_size'])
    width_stats = rivscale.products.along_stretch.AlongStretchStats.from_StretchStack(
        stretch_stack, signal_key='width', kernel_size=cfg['width_ref_kernel_size'])
    if cfg['crop']:
        # crop to reach
        wse_stats = wse_stats.crop_to_reach()
        width_stats = width_stats.crop_to_reach()
    return wse_stats, width_stats

def process_width_correction(cfg, stretch_stack_in):
    """
    This function processes the original stack of multitemporal 
    SWOT data node-level measurements over a multi-reach streach
    to estimate the along-river statistics for WSE and width
    as well as the height/width model fit for each node
    """
    # first handle optional config params
    if 'width_smooth_size' not in cfg.keys():
        cfg['width_smooth_size'] = 'None'
    if 'wse_ref_kernel_size' not in cfg.keys():
        cfg['wse_ref_kernel_size'] = '35'
    if 'width_ref_kernel_size' not in cfg.keys():
        cfg['width_ref_kernel_size'] = '11'
    stretch_stack = stretch_stack_in.copy()
    #breakpoint()
    if cfg['method']=='bundle_adjust':
        # TODO: enable potentially different config for bundle adjustment
        stretch_stack = bundle_adjust_per_pass_widths(
            apply_filter=False, cfg=cfg)
    return stretch_stack

# TODO: should delete this one?
def process_stack_filter_defunkt(cfg, stretch_stack_in):
    """
    This function processes the original stack of multitemporal 
    SWOT data node-level measurements over a multi-reach streach
    to estimate the along-river statistics for WSE and width
    as well as the height/width model fit for each node
    """
    # first handle optional config params
    if 'width_smooth_size' not in cfg.keys():
        cfg['width_smooth_size'] = 'None'
    if 'wse_ref_kernel_size' not in cfg.keys():
        cfg['wse_ref_kernel_size'] = '35'
    if 'width_ref_kernel_size' not in cfg.keys():
        cfg['width_ref_kernel_size'] = '11'
    stretch_stack = stretch_stack_in.copy()
    stretch_stack, _ = rivscale.filter.filter_stretch_stack(
        cfg,
        stretch_stack,
        wse_stats=None,
        width_stats=None,
        plot=False)
    if np.shape(stretch_stack.width)[1]==0:
        print('  No SWOT data left after multitemporal filtering')
        return None
    return stretch_stack

def process_flow_state(cfg, stretch_stack, wse_stats, wse_stretch_avg):
    """
    """
    # first handle optional config params
    if 'bin_width' not in cfg.keys():
        cfg['bin_width'] = '0.5'
    if 'oversamp_factor' not in cfg.keys():
        cfg['oversamp_factor'] = '2'

    flow_state = None
    if 'stretch_average' in cfg['method']:
        # TODO: check that input data are valid
        flow_state = rivscale.products.flow_state.FlowStateModel.from_objects(
            stretch_stack,
            wse_stats,
            wse_stretch_avg,
            bin_width = cfg['bin_width'], 
            oversamp_factor = cfg['oversamp_factor'])
    elif 'stack' in cfg['method']:
        #TODO: implement from_arrays
        pass
    return flow_state
       

def process_height_width_array(cfg, stretch_stack_in,
        wse_stats, flow_state, filterit=False):
    """
    This function processes the original stack of multitemporal 
    SWOT data node-level measurements over a multi-reach streach
    to estimate the along-river statistics for WSE and width
    as well as the height/width model fit for each node
    """
    # first handle optional config params 
    if 'use_flow_state' not in cfg.keys():
        cfg['use_flow_state'] = 'False'
    if 'crop' not in cfg.keys():
        cfg['crop'] = 'False'
    if 'snapit' not in cfg.keys():
        cfg['snapit'] = 'False'
    if 'neighbor_win_len' not in cfg.keys():
        cfg['neighbor_win_len'] = '0'
    stretch_stack = stretch_stack_in.copy()
    if filterit:
        # normally we will filter it before calling this function
        stretch_stack, _ = rivscale.filter.filter_stretch_stack(
            cfg,
            stretch_stack,
            wse_stats=None,
            width_stats=None,
            plot=False)
    if np.shape(stretch_stack.width)[1]==0:
        print('  No SWOT data left after multitemporal filtering')
        return None
    # compute multitemporal statistics
    wse_arr = stretch_stack.wse.copy()
    width_arr = stretch_stack.width.copy()
    wse_reference = wse_stats.reference
    if cfg['use_flow_state']:
        wse_arr = flow_state.wse_profiles
        width_arr = flow_state.width_profiles
    height_width_array = \
        rivscale.products.height_width_array.HeightWidthModelArray.from_arrays(
            wse_arr,
            width_arr,
            along_dist=stretch_stack.along_dist.copy(),
            node_id=stretch_stack.node_id.copy(),
            stretch_name=stretch_stack.stretch_name,
            wse_reference=wse_reference,
            snapit=cfg['snapit'],
            neighbor_win_len=cfg['neighbor_win_len'])
    # now optionally crop
    if cfg['crop']:
        # crop to reach
        height_width_array = height_width_array.crop_to_reach()
    return height_width_array

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
    wse_stretch_avg = rivscale.products.stretch_average.StretchAverageStats.from_StretchStack(
        stretch_stack, signal_key='wse', along_stats=wse_stats,
        average_method='bayes_simple',#'bayes_weighted',
        slope_method='bayes',
        reach_id=reach_id)
    width_stretch_avg = rivscale.products.stretch_average.StretchAverageStats.from_StretchStack(
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
    if np.sum(msk)==0:
        # TODO: print warning?
        return med # med is all nonfinite, but return anyways
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

