'''
Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent Williams

This is a place to put specialized functions
'''
import numpy as np
import scipy.stats
from sklearn.isotonic import IsotonicRegression
import warnings
from rivscale.misc import swot_time_to_field_time

def isotonic_regression_stack(y,x, kind='linear', fill_value=np.nan):#'extrapolate'):
    """
    y is a stack with possibly nan values (e.g., stack.wse)
    x is the 1D along_dist
    """
    # check that x is increasing
    dx = np.diff(x)
    #breakpoint()
    if np.any(dx<0):
        if np.any(dx>0):
            print("x is not increasing or decreasing")
        else:
            x = -x
    ir = IsotonicRegression(out_of_bounds="clip")
    M,N = np.shape(y)
    y_mono = np.zeros_like(y) + np.nan
    for k in range(N):
        msk = np.isfinite(y[:,k])
        if len(y[msk,k])==0:
            continue
        y_mono[msk,k] = ir.fit_transform(x[msk], y[msk,k])
        # interpolate over holes
        intrp = scipy.interpolate.interp1d(
            x[msk],
            y_mono[msk,k],
            kind=kind,
            bounds_error=False,
            fill_value=fill_value
            )
        y_mono[:,k] = intrp(np.array(x))
    return y_mono


def wse_binned_stats(wses, widths, wse_bin_width=0.2, oversamp_factor=2):
    """
    For each node, take statistics of width for all samples within a wse bin
    wse_bin_width:  the window size to bin similar wses
    oversamp_factor: integer greater than 0
    TODO: also generalize percentiles, maybe add mean/std etc...
    """
    warnings.filterwarnings("ignore")
    mx = np.nanmax(wses) + wse_bin_width * 4
    mn = np.nanmin(wses) - wse_bin_width * 2
    d_wse = wse_bin_width
    n_bins = int(np.floor((mx-mn) / d_wse))
    wse_bins0 = np.arange(n_bins)*d_wse + mn#np.linspace(mn, mx1, n_bins)
    wses_bins = []
    for k in range(oversamp_factor):
        wses_bins.append(wse_bins0[0:-1] + k * d_wse/oversamp_factor)
    N_bins = 0
    for bins in wses_bins:
        N_bins = N_bins + len(bins)
    # only use measurements where wse and width are valid
    wses = wses.copy()
    widths = widths.copy()
    wses[~np.isfinite(widths)] = np.nan
    widths[~np.isfinite(wses)] = np.nan
    valids = np.ones_like(wses)
    valids[~np.isfinite(widths)] = 0
    valids[~np.isfinite(wses)] = 0
    # define some local helper functions
    def perc_25(data):
        return np.nanpercentile(data, 25)
    def perc_75(data):
        return np.nanpercentile(data, 75)
    # aggregate stats over wse bins
    num_nodes = len(wses[:,0].flatten())
    num_samps = N_bins-oversamp_factor
    wse_med = np.zeros((num_nodes, num_samps)) + np.nan
    width_med = np.zeros((num_nodes, num_samps)) + np.nan
    width_p25 = np.zeros((num_nodes, num_samps)) + np.nan
    width_p75 = np.zeros((num_nodes, num_samps)) + np.nan
    count = np.zeros((num_nodes, num_samps))
    for k in range(num_nodes):
        for kk, wse_bins in enumerate(wses_bins):
            skip = len(wses_bins)
            #breakpoint()
            count[k,kk::skip] = scipy.stats.binned_statistic(
                wses[k,:], valids[k,:],
                statistic=np.nansum, bins=wse_bins)[0]
            #
            wse_med[k,kk::skip] = scipy.stats.binned_statistic(
                wses[k,:], wses[k,:],
                statistic=np.nanmedian, bins=wse_bins)[0]
            # 
            width_med[k,kk::skip] = scipy.stats.binned_statistic(
                wses[k,:], widths[k,:],
                statistic=np.nanmedian, bins=wse_bins)[0]
            width_p25[k,kk::skip] = scipy.stats.binned_statistic(
                wses[k,:], widths[k,:],
                statistic=perc_25, bins=wse_bins)[0]
            width_p75[k,kk::skip] = scipy.stats.binned_statistic(
                wses[k,:], widths[k,:],
                statistic=perc_75, bins=wse_bins)[0]
        #
    return count, wse_med, width_med, width_p25, width_p75


def seasonal_stats(stack, keys=['wse', 'width'], bins=np.linspace(0, 366, 10)):
    """
    aggregate stack stats over multple years at similar day-of-year
    TODO: generalize percentiles, maybe also take means/std etc...
    """
    # bin time dim of stretch-stack into doy bins
    dates = dates = swot_time_to_field_time(stack.time_id*60*60)
    doy = [date.timetuple().tm_yday for date in dates]
    # TODO: make mask where data from all keys exist (e.g., height/width)
    def perc_25(data):
        return np.nanpercentile(data, 25)
    def perc_75(data):
        return np.nanpercentile(data, 75)
    dic = {
        'bins':bins,
        'med':[],
        'p25':[],
        'p75':[],
        }
    for key in keys:
        dic['med'].append(scipy.stats.binned_statistic(
            doy, stack[key], statistic=np.nanmedian, bins=bins)[0])
        dic['p25'].append(scipy.stats.binned_statistic(
            doy, stack[key], statistic=perc_25, bins=bins)[0])
        dic['p75'].append(scipy.stats.binned_statistic(
            doy, stack[key], statistic=perc_75, bins=bins)[0])

    #breakpoint
    return dic

