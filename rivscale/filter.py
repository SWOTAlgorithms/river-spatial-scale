'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent Williams
'''

import numpy as np
import matplotlib.pyplot as plt
import rivscale.data
import rivscale.products.stretch_stack

def filter_stretch_stack(
        cfg,
        stretch_stack,
        wse_stats,
        width_stats,
        plot=False,
        outdir=None,
        reach_id='nope'):
    """
    Perform outier rejection and dark frac filtering etc
    """
    #breakpoint()
    # first handle optional config params
    # TODO: define parameter defaults all in one place?
    if 'width_smooth_size' not in cfg.keys():
        cfg['width_smooth_size'] = 'None'
    if 'wse_dark_thresh' not in cfg.keys():
        cfg['wse_dark_thresh'] = '0.8'
    if 'width_dark_thresh' not in cfg.keys():
        cfg['width_dark_thresh'] = '0.3'
    if 'wse_outlier_scale' not in cfg.keys():
        cfg['wse_outlier_scale'] = '5.0'
    if 'width_outlier_scale' not in cfg.keys():
        cfg['width_outlier_scale'] = '5.0'
    if 'crop' not in cfg.keys():
        cfg['crop'] = 'False'
    #
    # populate witdh_u
    # TODO: fix the uncertainty itself instead of fudging it here  
    node_len = stretch_stack['area_total'] / stretch_stack['width']
    stretch_stack['width_u'] = stretch_stack['area_tot_u'] / node_len
    # make measurement uncert at least as much as signal uncert we assume
    stretch_stack['width_u'] = stretch_stack['width_u'] + 10 # + 500#2*prior_unc_alpha_width
    # smooth the widths before anything else, if commanded
    if cfg['width_smooth_size'] is not None:
        stretch_stack.smooth_widths(size=cfg['width_smooth_size'])
    # first remove outliers allowing typical spread of variability
    use_ptiles = False
    if width_stats is not None:
        # assume it is a pekel, TODO: parameterize this
        use_ptiles = True
    # over all time obs
    stretch_stack.filter_node_outliers(# TODO: pass in scale parameter
        width_stats,
        key='width',
        use_ptiles=use_ptiles,
        IQR_scale=cfg['width_outlier_scale'],
        plot=plot,
        outdir=outdir)
    stretch_stack.filter_node_outliers(
        wse_stats,
        key='wse',
        IQR_scale=cfg['wse_outlier_scale'],
        plot=plot,
        outdir=outdir)
    # now remove outliers considering relative spread 
    stretch_stack.filter_node_outliers(
        width_stats,
        key='width',
        use_ptiles=use_ptiles,
        IQR_scale=cfg['width_outlier_scale'],
        Delta2=True,
        plot=plot,
        outdir=outdir)
    stretch_stack.filter_node_outliers(
        wse_stats,
        Delta2=True,
        key='wse',
        IQR_scale=cfg['wse_outlier_scale'],
        plot=plot,
        outdir=outdir)
    # filter out high dark_frac nodes
    stretch_stack.filter_dark_water('width', cfg['width_dark_thresh'])
    stretch_stack.filter_dark_water('wse', cfg['wse_dark_thresh'])
    # drop rows with too  little data
    #reach_id = 'nope' # TODO: handle this as param
    if cfg['crop']:
        reach_id = stretch_stack.stretch_name
    stretch_stack = drop_stretch_nans(stretch_stack,
        reach_id=reach_id)
    return stretch_stack, reach_id

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


def filter_qual(
        df,
        height=True,
        area=True,
        dark_thresh=1.0,
        kind='OB',
        ):
    """
    filter out swot data based on quality, dark_frac, ice,
    location in swath etc...
    """
    if 'node_q' in df.keys():
        qual_key = 'node_q'
        qual_b_key = 'node_q_b'
    else:
        qual_key = 'reach_q'
        qual_b_key = 'reach_q_b'

    ###
    # compute all the various masks for wse and area
    ###
    # filter out ice
    #ice = df['ice_clim_f']==0
    ice = df['ice_clim_f']<=0 # ignore ice flag if it is negative/fill_value
    # filter out bad qual
    bad  = df[qual_key] < 3
    # drop xovr_cal_q == 2
    xover = df['xovr_cal_q'] < 2
    #    # fill values
    wse_fill = df['wse'] > -99999999.0
    area_fill = df['area_total'] > -99999999.0

    # filter out swath edges
    near = np.bitwise_and(df[qual_b_key], 2**13) == 0
    far = np.bitwise_and(df[qual_b_key], 2**14) == 0

    # geolocation_qual_degraded
    geoloc_deg = np.bitwise_and(df[qual_b_key], 2**19) == 0
    # wse outlier
    wse_outlier = np.bitwise_and(df[qual_b_key], 2**23) == 0
    # classification_qual_degraded
    class_q_deg = np.bitwise_and(df[qual_b_key], 2**18) == 0
    # filter out high dark frac
    dark = df['dark_frac'] <= dark_thresh
    ###
    # always drop ice, xover, and bad_q, and fill and dark_frac
    # for both wse and area
    ###
    wse_keep = np.logical_and.reduce([ice, bad, xover, wse_fill, dark])
    area_keep = np.logical_and.reduce([ice, bad, xover, area_fill, dark])

    if 'OB' in kind:
        # filter out swath edges of both wse and area
        wse_keep = np.logical_and.reduce([wse_keep, near, far])
        area_keep = np.logical_and.reduce([area_keep, near, far])
        # drop wse outliers
        wse_keep = np.logical_and.reduce([wse_keep, wse_outlier])
    if 'no_degraded' in kind:
        # drop degraded wse and wse outliers
        wse_keep = np.logical_and.reduce([wse_keep, geoloc_deg])
        # drop classification_qual_degraded
        area_keep = np.logical_and.reduce([area_keep, class_q_deg])
    # TODO: handle OBIM etc
    #breakpoint()
    # null-out the bad data with nans
    if height:
        wse_drop = np.logical_not(wse_keep)
        df['wse'][wse_drop] = np.nan
        if not area:
            # drop all the nodes with bad wse
            df = df[wse_keep]
    if area:
        area_drop = np.logical_not(area_keep)
        df['area_total'][area_drop] = np.nan
        df['area_det'][area_drop] = np.nan
        df['width'][area_drop] = np.nan
        if not height:
            # drop all the nodes with bad area/width
            df = df[width_keep]
    if (height and area):
        # drop only where both wse and area are bad
        keep_df = np.logical_or(wse_keep, area_keep)
        df = df[keep_df]
    return df


def drop_stretch_nans(stretch_stack, min_nodes=10, reach_id=None):
    """
    Prune out the time/cycle observations with too little
    good data in either wse or width.

    inputs:
        stretch_stack = a StretchStack instance with populated input data
        min_nodes    = min number of valid nodes to consider to keep

    outputs:
        data_out     = copy of streach_data with rows dropped
    """
    # TODO make this a class method?
    #create a new container instance
    #data_out = rivscale.products.RiverStretchData()
    data_out = rivscale.products.stretch_stack.StretchStack()
    # copy the attributes
    data_out.stretch_name = stretch_stack.stretch_name
    wse = stretch_stack['wse']
    width = stretch_stack['width']
    #num = np.sum(np.logical_or(np.isfinite(wse), np.isfinite(width)), axis=0)
    # optionally null out data outside the desired reach
    window_valid = np.ones_like(wse)
    if reach_id is None:
        if stretch_stack.stretch_name.isdigit():
            reach_id = stretch_stack.stretch_name
        else:
            reach_id = 'bad'
    if ((reach_id.isdigit()) and (len(reach_id)==11)):
        print('windowing to reach {}'.format(reach_id))
        reach_ids = np.array(
            [str(n)[0:10]+str(n)[-1] for n in stretch_stack.node_id])
        window_valid[reach_ids!=reach_id] = np.nan
    # go through each data that is populated
    num = np.sum(np.logical_or(
        np.isfinite(wse * window_valid),
        np.isfinite(width * window_valid)), axis=0)
    for key in stretch_stack.variables.keys():
        if 'num_times' in stretch_stack.VARIABLES[key]['dimensions'].keys():
            dim = [*stretch_stack.VARIABLES[key]['dimensions'].keys()].index('num_times')
            #print(key, dim)
            if dim==0:
                data_out[key] = stretch_stack[key][num>min_nodes].copy()
            if dim==1:
                data_out[key] = stretch_stack[key][:,num>min_nodes].copy()
            if dim==2:
                data_out[key] = stretch_stack[key][:,:,num>min_nodes].copy()
        else:
            # just copy the data over to output
            data_out[key] = stretch_stack[key].copy()
    return data_out


