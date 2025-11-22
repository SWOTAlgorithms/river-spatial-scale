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

def var_binned_node_stats(bin_var_in, var1_in, var2_in=None,
        bin_width=0.2, oversamp_factor=2, percentile_list=[25, 50, 75]):
    """
    For each node, take statistics of width for all samples within a bin_var bin
    bin_width:  the window size to bin similar bin_var
    oversamp_factor: integer greater than 0
    
    e.g., this function can be used to estimate width statistics at particular
    wse bins (either node-level wses, or reach/stretch-level, but mapped to
    2D stretch-stack-style array).

    INPUTS:
    bin_var_in: variable to bin var1 (and optionally var1), handles 1D or 2D
                    array (e.g., wse_stretch_average.mean, or just stack.wse)
    var1_in:    primary variable that is binned 2D stretch_stack-style array
                    (e.g. stack.width)
    var2_in:    optional secondary variable to bin
    
    bin_width:  the window size to bin similar bin_var
    oversamp_factor: integer greater than 0
    percentile_list: which percentiles to compute

    OUTPUTS:
    bins: the bin edges used in binning
    count: the number of time-obs nodes for each node tha were aggregated
           to statistics. 1D array, node-length
    var1(2)_stats: the statistics that were computed 3D array
                (num_ptiles, num_times, num_nodes)
    percentile_list: the list of percentiles corresponding to the first row
                     of var1(2) stats
    
    TODO: maybe add mean/std etc...
    """
    warnings.filterwarnings("ignore")
    # make internal copies
    var1 = var1_in.copy()
    bin_var = bin_var_in.copy()
    # handle 1D bin_var (e.g., from stretch_average_stats)
    if len(np.shape(bin_var))==1:
        # if it is 1D broadcast to 2D
        bin_var = np.broadcast_to(bin_var, np.shape(var1)).copy()
    if var2_in is not None:
        var2 = var2_in.copy()
    else:
        var2 = None
    # only use measurements where all are valid
    msk = np.logical_or(~np.isfinite(bin_var), ~np.isfinite(var1))
    var1[msk] = np.nan
    bin_var[msk] = np.nan
    valids = np.ones_like(var1)
    valids[msk] = 0
    if var2 is not None:
        var2 = var2.copy()
        msk = np.logical_or.reduce((
            ~np.isfinite(bin_var),
            ~np.isfinite(var1),
            ~np.isfinite(var2),
            ))
        var1[msk] = np.nan
        bin_var[msk] = np.nan
        var2[msk] = np.nan
        valids = np.ones_like(var1)
        valids[msk] = 0
    # set up the binning
    mx = np.nanmax(bin_var) + bin_width * 4
    mn = np.nanmin(bin_var) - bin_width * 2
    d_bin_var = bin_width
    n_bins = int(np.floor((mx-mn) / d_bin_var))
    bins0 = np.arange(n_bins)*d_bin_var + mn#np.linspace(mn, mx1, n_bins)
    bins = []
    for k in range(oversamp_factor):
        bins.append(bins0[0:-1] + k * d_bin_var/oversamp_factor)
    N_bins = 0
    for tbins in bins:
        N_bins = N_bins + len(tbins)
    # aggregate stats over wse bins
    num_nodes = len(var1[:,0].flatten())
    num_samps = N_bins-oversamp_factor
    num_stats = len(percentile_list)
    var1_stats = np.zeros((num_stats, num_nodes, num_samps)) + np.nan
    if var2 is not None:
        var2_stats = np.zeros((num_stats, num_nodes, num_samps)) + np.nan
    bin_stats = np.zeros((num_stats, num_nodes, num_samps)) + np.nan
    count = np.zeros((num_nodes, num_samps))
    for k in range(num_nodes):
        for kk, tbins in enumerate(bins):
            skip = len(bins)
            count[k,kk::skip] = scipy.stats.binned_statistic(
                bin_var[k,:], valids[k,:],
                statistic=np.nansum, bins=tbins)[0]
            for p, ptile in enumerate(percentile_list):
                percentile_func = lambda arr: np.nanpercentile(arr, ptile)
                # bin var1
                var1_stats[p, k,kk::skip] = scipy.stats.binned_statistic(
                    bin_var[k,:], var1[k,:],
                    statistic=percentile_func, bins=tbins)[0]
                bin_stats[p, k,kk::skip] = scipy.stats.binned_statistic(
                    bin_var[k,:], bin_var[k,:],
                    statistic=percentile_func, bins=tbins)[0]
                # bin the input bin_var too
                if var2 is not None:
                    # bin var2
                    var2_stats[p, k,kk::skip] = scipy.stats.binned_statistic(
                        bin_var[k,:], var2[k,:],
                        statistic=percentile_func, bins=tbins)[0]
        #
    if var2 is None:
        return bins, count, percentile_list, bin_stats, var1_stats
    return bins, count, percentile_list, bin_stats, var1_stats, var2_stats

def seasonal_stats(stack, keys=['wse', 'width'],
        percentile_list=[25, 50, 75], bins=np.linspace(0, 366, 10)):
    """
    aggregate stack stats over multple years at similar day-of-year
    TODO: maybe also take means/std etc, maybe make this a class/object
    """
    # bin time dim of stretch-stack into doy bins
    dates = dates = swot_time_to_field_time(stack.time_id*60*60)
    doy = [date.timetuple().tm_yday for date in dates]
    # TODO: make mask where data from all keys exist (e.g., height/width)
    dic = {'bins':bins}
    for ptile in percentile_list:
        percentile_func = lambda arr: np.nanpercentile(arr, ptile)
        pkey = f'p{ptile}'
        dic[pkey] = []
        for key in keys:
            dic[pkey].append(scipy.stats.binned_statistic(
                doy, stack[key], statistic=percentile_func, bins=bins)[0])
    return dic

