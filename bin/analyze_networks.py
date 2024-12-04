#!/usr/bin/env python
'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

Read in the river tiles, find connected river networks and do
special bayes processing

'''
import glob
import os.path

import argparse

import numpy as np
import pandas as pd

import errtools.plots
import matplotlib.pyplot as plt

import scipy.ndimage
from scipy.linalg import pinv, svd, eigh, norm

from errtools.misc import (split_utc_time, field_time_to_swot_time,
    swot_time_to_field_time, get_dist_to_outlet_from_node_id)

from errtools.plots import plot_cdf

import netCDF4 as nc

import statsmodels.api

import geopandas as gpd

import rivscale.io
import rivscale.estimate 
import rivscale.plot
import rivscale.data
import rivscale.reconstruct
import rivscale.filter

EXAMPLE=''

def create_parser():
    """Create parser for command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate River performance with field data',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('dataframe_dir', default=None,
        help='PT and drift node dataframe directory')
    parser.add_argument('rivertile_dir', default=None,
        help='SWOT rivertile directory')
    parser.add_argument('sword_file', default=None,
        help='SWORD netcdf file')
    parser.add_argument('river_name', default=None,
        help='SWORD river name')
    return parser



def main():
    parser = create_parser()
    args = parser.parse_args()
    fles = glob.glob(os.path.join(args.dataframe_dir,'*.csv'))
    # read SWORD
    sword_df, sword_node_df, d_up, d_down = rivscale.io.read_SWORD(args.sword_file)

    # read the SWOT data
    swot_df, swot_node_df = rivscale.io.read_rivertiles(args.rivertile_dir)
    # drop reaches from swot data not in the remaiing sword_df
    bad_reaches = list(set(swot_node_df['reach_id'])-set(sword_df['reach_id']))
    for rch in bad_reaches:
        swot_node_df = swot_node_df[swot_node_df['reach_id']!=rch]
    """
    # drop sword reaches that are not in the swot data
    bad_reaches = list(set(sword_df['reach_id'])-set(swot_node_df['reach_id']))
    for rch in bad_reaches:
        sword_df = sword_df[sword_df['reach_id']!=rch]
    #sword_df = sword_df[sword_df['reach_id'] in swot_node_df['reach_id']]
    """
    # drop all sword reaches that dont have the desired river_name
    river_name = args.river_name
    #river_name = 'Ocmulgee River'
    #river_name = 'Colorado River'
    sword_df = sword_df[sword_df['river_name']==river_name]
    #sword_node_df = sword_node_df[sword_node_df['river_name']==river_name]
    #breakpoint()
    """
    # drop reaches from swot data not in the remaiing sword_df
    bad_reaches = list(set(swot_node_df['reach_id'])-set(sword_df['reach_id']))
    for rch in bad_reaches:
        swot_node_df = swot_node_df[swot_node_df['reach_id']!=rch]
    """
    print("creating network list")
    network_list = rivscale.data.get_connected_networks(sword_df, d_up, d_down)
    #breakpoint()
    # hack to filter out unneeded things for the yellowstone case TODO generalize this
    #network_list = [network_list[3]]
    #network_list = [network_list[0]]
    # just look at the largest network
    lens = [len(n) for n in network_list]
    network_list = [network_list[np.argmax(lens)]]
    #breakpoint()
    # create the data stack
    full_profile_data0 = rivscale.data.make_swot_data_stack(swot_node_df, sword_node_df)
    #breakpoint()
    # create the network stack
    full_profile_data1 = rivscale.data.network_stack(full_profile_data0, network_list)
    # update the uncertainties
    full_profile_data = rivscale.filter.modify_uncert(full_profile_data1)
    breakpoint()
    # interpolate over holes
    full_profile_data = rivscale.filter.interp_stack(full_profile_data, left=np.nan, right=np.nan)
    # estimate mean and covariance
    stats = rivscale.data.stack_mean_and_covariance(full_profile_data)
    # reconstruct the profiles
    bayes_data = rivscale.reconstruct.reconstruct_stack(stats, full_profile_data)

    # load in the field data
    #pt_df, drift_df = rivscale.io.load_field_data(fles, args.sword_file)
    
    # make Bayes with simple model params
    stats3 = rivscale.filter.replace_signal_stats(stats, full_profile_data, sword_node_df)
    bayes_data3 = rivscale.reconstruct.reconstruct_stack(stats3, full_profile_data)
    
    full_profile_data2 = rivscale.filter.detrend(full_profile_data, sword_node_df)
    anom_data0 = rivscale.misc.compute_anomaly(full_profile_data2, bayes_data=None, sword_node_df=None)
    anom_data = rivscale.misc.compute_anomaly(full_profile_data2, bayes_data=bayes_data3, sword_node_df=None)
    # do bayes, but with med_prof_stack as the mean
    stats2 = {}# estimated cov with full mean profile
    stats4 = {}# exp cov with full mean profile
    stats5 = {}# combo cov with full mean profile
    alpha = 0.2
    for key in stats:
        stats2[key] = []
        stats4[key] = []
        stats5[key] = []
        for k, network in enumerate(stats['network']):
            stats2[key].append(stats[key][k])
            stats4[key].append(stats3[key][k])
            stats5[key].append(stats[key][k])
    for k, network in enumerate(stats['network']):
        stats2['mean'][k] = anom_data['med_prof_stack'][k][:,0]
        stats4['mean'][k] = anom_data['med_prof_stack'][k][:,0]
        stats5['mean'][k] = anom_data['med_prof_stack'][k][:,0]
        stats5['Ry'][k] = (alpha*stats['Ry'][k] + (1-alpha)*stats3['Ry'][k])
    bayes_data2 = rivscale.reconstruct.reconstruct_stack(stats2, full_profile_data)
    bayes_data4 = rivscale.reconstruct.reconstruct_stack(stats4, full_profile_data)
    bayes_data5 = rivscale.reconstruct.reconstruct_stack(stats5, full_profile_data)
    #plot_profile_with_bayes(full_profile_data, bayes_data3, anom_data)
    rivscale.plot.plot_profile_with_bayes(full_profile_data, bayes_data2)
    rivscale.plot.plot_profile_with_bayes(full_profile_data, bayes_data2, anom_data)
    #
    rivscale.plot.plot_profile_with_bayes(full_profile_data, bayes_data4)
    rivscale.plot.plot_profile_with_bayes(full_profile_data, bayes_data4, anom_data)
    #
    rivscale.plot.plot_profile_with_bayes(full_profile_data, bayes_data5)
    rivscale.plot.plot_profile_with_bayes(full_profile_data, bayes_data5, anom_data)
    #plt.show()
    #plt.plot()
    # look at spectra
    # first interpolate over holes, while handling ends
    tmp = full_profile_data['wse_stack_interp'][0]
    msk = ~np.isfinite(tmp)
    tmp0 = full_profile_data['wse_stack'][0]
    msk0 = ~np.isfinite(tmp0)
    tmp[msk] = full_profile_data['wse_stack_linfit'][0][msk]
    tmp2 = tmp - full_profile_data['wse_stack_linfit'][0]
    jnk = tmp0-full_profile_data['wse_stack_linfit'][0]
    n_p = np.nanstd(jnk[0:100,:])
    #n_p = np.nanmedian(full_profile_data['wse_u_stack'][0][~msk0])
    noise = np.random.randn(len(tmp[:,0]), len(tmp[0,:]))*n_p
    # add fake noise to the interpolated sections so it doesnt shape spectral noise floor
    tmp2[msk0] = tmp2[msk0] + noise[msk0]
    #tmp20 = full_profile_data['wse_stack'][0] - full_profile_data['wse_stack_linfit'][0]
    #dist = full_profile_data['dist_out'][0]
    plt.figure()
    plt.plot(tmp2)

    plt.figure()
    avg = 0
    for k in range(len(tmp[0,:])):
        f, pxx = scipy.signal.periodogram(tmp2[:,k])
        #f=f0[1:]
        #f = np.linspace(0.01, 10, 1000)
        #msk = np.where(np.isfinite(tmp20[:,k]))
        #wse = tmp20[:,k]
        #breakpoint()
        #pxx = scipy.signal.lombscargle(wse[msk], dist[msk], f)
        plt.loglog(f, pxx)
        #plt.semilogy(f, pxx)
        avg = avg + pxx
        #plt.show()
    ylim = (1e-6, 1000)
    plt.grid()
    plt.ylabel('power spectral density')
    plt.xlabel('wave-number/spatial-frequency (1/node)')
    plt.ylim(ylim)
    avg = avg / len(tmp[0,:])
    plt.figure()
    plt.loglog(f, avg)
    #n_p = np.sqrt(np.nanmedian(full_profile_data['wse_u_stack'][0]))
    noise = np.random.randn(len(tmp[:,0]))*n_p
    noise2 = np.random.randn(len(tmp[:,0]))*(n_p**2)
    fn, pxxn = scipy.signal.periodogram(noise)
    fn, pxxn2 = scipy.signal.periodogram(noise2)
    pxxn_c = np.zeros_like(pxxn) + np.mean(pxxn)
    plt.loglog(fn, pxxn_c,'--')
    plt.grid()
    plt.ylabel('power spectral density')
    plt.xlabel('wave-number/spatial-frequency (1/node)')
    #plt.loglog(fn, pxxn_c/np.sqrt(len(tmp[0,:])),'--')
    #plt.loglog(fn, pxxn,'--')
    #plt.loglog(fn, pxxn2,'--')
    # look at lomb-scargle periodogram without interpolating first
    #scipy.signal.lombscargle()

    #breakpoint()
    # check out the exponetial that fits this the best...?
    dist = full_profile_data['dist_out'][0]
    dist2 = dist - dist[int(len(dist)/2)]
    char_length_tau = 20000
    prior_unc_alpha=2
    cov = np.exp(-np.abs(dist2) / char_length_tau)
    cov = cov / np.max(cov) * prior_unc_alpha**2
    ft = np.fft.fft(cov)
    plt.loglog(fn, np.abs(ft[0:len(fn)]))
    plt.loglog(fn, np.abs(ft[0:len(fn)])+np.mean(pxxn))
    plt.legend(['mean psd', 'noise floor', 'exp. cov', 'exp. cov + noise'])
    plt.ylim(ylim)
    #plt.show()

    """
    # plot the PTs at the same time/place as SWOT
    nodes = np.unique(pt_df['node_id'])
    # hack for Yellowstone TODO generalize this
    good_nodes = [71241000100011, 71241000100101, 71241000110411, 71241000110861, 71241000120341, 71241000120211]
    bad_nodes = list(set(nodes) - set(good_nodes))
    # filter out bad pts
    for node in bad_nodes:
        pt_df = pt_df[pt_df['node_id'] != node]

    pt_swot_match = rivscale.filter.find_pt_swot_matches(pt_df, full_profile_data)
    rivscale.data.stuff_pt_stack(pt_swot_match, full_profile_data)
    """
    rivscale.plot.plot_profile_with_bayes(full_profile_data, bayes_data4, anom_data)
    
    rivscale.plot.plot_mean_profile(full_profile_data, anom_data0)
    
    R = stats3['Ry'][0]
    x0 = int(len(R[:,0])/2)
    dst = full_profile_data['dist_out'][0]/1000
    x = dst - dst[x0]
    plt.figure()
    plt.plot(x, R[:, x0])
    plt.grid()
    plt.ylabel('covariance')
    plt.xlabel('distance (km)')
    plt.figure()
    plt.imshow(R)
    plt.colorbar()
    # now plot the posterior cov
    plt.figure()
    plt.imshow(bayes_data4['post_cov'][0][0])
    plt.colorbar()
    plt.title('posterior covariance')
    plt.figure()
    for c,cyc in enumerate(bayes_data['cycle'][0]):
        err = np.diag(bayes_data4['post_cov'][0][c])
        plt.plot(dst, np.sqrt(err))
    plt.xlabel('dist out (km)')
    plt.ylabel('sqrt(post. cov.) (m)')
    plt.grid()
    plt.show()
    #
    breakpoint()
    # now plot some hysteresis stuff
    plt.figure()
    lgnd = []
    nodes = np.unique(pt_df['node_id'])
    # hack for Yellowstone
    good_nodes = [71241000100011, 71241000100101, 71241000110411, 71241000110861, 71241000120341, 71241000120211]
    for node in good_nodes:
        this_pt_df = pt_df[pt_df['node_id']==node]
        plt.plot(this_pt_df['pt_time_UTC'], this_pt_df['mean_node_pt_wse_m'])
        lgnd.append(node)
    plt.xlabel('time UTC')
    plt.ylabel('wse (m)')
    plt.legend(lgnd)
    # 
    cm_km = (1000*100)
    tmp = bayes_data4['wse_stack'][0]-anom_data['med_prof_stack'][0]
    tmp_prime = np.diff(tmp, axis=0) * 100 # in cm
    dst_diff = np.diff(dst)
    #breakpoint()
    for k in range(len(tmp_prime[0,:])):
        tmp_prime[:,k] = tmp_prime[:,k]/dst_diff # in cm/km
    tmp_prime_smooth = scipy.ndimage.uniform_filter1d(tmp_prime, 20, axis=0)
    plt.figure()
    plt.plot(dst[0:-1], tmp_prime)
    plt.xlabel('dist out (km)')
    plt.ylabel('node-level slope anomaly (cm/km)')
    plt.grid()
    plt.figure()
    plt.plot(tmp_prime, tmp[0:-1,:])
    plt.xlabel('wse_anomaly (m)')
    plt.ylabel('node-level slope anomaly (cm/km)')
    plt.grid()
    #
    plt.figure()
    plt.plot(dst[0:-1], tmp_prime_smooth)
    plt.xlabel('dist out (km)')
    plt.ylabel('smoothed node-level slope anomaly (cm/km)')
    plt.grid()
    plt.figure()
    plt.plot(tmp_prime_smooth, tmp[0:-1,:])
    plt.xlabel('wse_anomaly (m)')
    plt.ylabel('smoothed node-level slope anomaly (cm/km)')
    plt.grid()
    #
    plt.figure()
    for node in good_nodes:
        this_pt_df = pt_df[pt_df['node_id']==node]
        mean = anom_data['med_prof_stack'][0]
        nid = anom_data['node_id'][0]
        id0 = np.where(nid == node)
        #mean[id0][0]
        #breakpoint()
        plt.plot(this_pt_df['pt_time_UTC'], np.array(this_pt_df['mean_node_pt_wse_m'])-mean[id0][0][0])
        lgnd.append(node)
    plt.xlabel('time UTC')
    plt.ylabel('wse anomaly (m)')
    plt.legend(lgnd)
    #
    # make a hysteresis plot from field data
    this_node = 71241000110411
    that_node = 71241000100011
    this_pt_df = pt_df[pt_df['node_id']==this_node]
    that_pt_df = pt_df[pt_df['node_id']==that_node]
    this_wse = np.array(this_pt_df['mean_node_pt_wse_m'])
    that_wse = np.array(that_pt_df['mean_node_pt_wse_m'])
    nid = anom_data['node_id'][0]
    this_nid = np.where(nid == this_node)
    that_nid = np.where(nid == that_node)
    #breakpoint()
    this_time = field_time_to_swot_time(this_pt_df['pt_time_UTC_str'])
    that_time = field_time_to_swot_time(that_pt_df['pt_time_UTC_str'])
    # resample to same time
    that_wse2 = np.interp(this_time, that_time, that_wse)
    this_wse_anom = this_wse - anom_data['med_prof_stack'][0][this_nid[0]][0][0]
    that_wse2_anom = that_wse2 - anom_data['med_prof_stack'][0][that_nid[0]][0][0]
    d_wse_anom = this_wse_anom - that_wse2_anom
    plt.figure()
    plt.plot(this_pt_df['pt_time_UTC'], d_wse_anom)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('pressure transducer slope anomaly')
    plt.figure()
    plt.plot(this_pt_df['pt_time_UTC'], this_wse_anom)
    plt.plot(this_pt_df['pt_time_UTC'], that_wse2_anom)
    plt.grid()
    plt.ylabel('pressure transducer wse anomaly')
    plt.xlabel('time')
    plt.figure()
    plt.plot(d_wse_anom, this_wse_anom)
    plt.grid()
    plt.xlabel('pressure transducer wse anomaly')
    plt.ylabel('pressure transducer slope anomaly')
    plt.show()
    # 
    breakpoint()
    # TODO: plot full_extent anomaly (vs dist_out?) smashiing all consecutive reaches together
    #plot_anomaly_per_cycle(anom_data)
    breakpoint()

    # get the mean profiles
    #swot_mean_df = compute_mean_swot_profile(
    #    swot_node_df,
    #    np.unique(swot_node_df['reach_id']),
    #    sword_node_df)
    ##estimate_offsets()
    ## get the covariance
    #full_profile_data = estimate_swot_mean_and_covariances(swot_node_df, sword_node_df, plotem=False)
    #breakpoint()
    ##
    #plot_swot_profiles(swot_df, swot_node_df, full_profile_data, swot_mean_df)
    ##swot_df, swot_node_df = filter_bad_swot(swot_df, swot_node_df)
    ##
    #sword_df, sword_node_df = read_SWORD(args.sword_file)
    #pt_df, drift_df = load_field_data(fles, args.sword_file)

    # filter out bad drifts
    good_drift_df = flag_drifts(drift_df, pt_df, sword_node_df)
    match_df = find_pt_drift_matches(pt_df, drift_df)

    # fix the jumps in the PT data
    #fix_pt_jumps(pt_df, drift_df, good_drift_df, match_df)
   
    # get the pt reach average dataframe
    reach_average_df = get_reach_average_df(pt_df, np.unique(match_df['reach_id'])) 
    plot_reach_average_df(reach_average_df)

    # estimate mean and covariances for each reach
    good_drifts = estimate_mean_and_covariances(drift_df, good_drift_df, sword_node_df, match_df)

    breakpoint()
    good_drifts['mean'] = good_drifts['mean_reach_profile']
    bayes_data2 = reconstruct_stack(good_drifts, full_profile_data)
    reach_average_data2 = compute_reach(bayes_data2)
    plot_profile_with_bayes(full_profile_data, bayes_data2)
    plot_reach_average(reach_average_data2, swot_df)
    plt.show()
    #swot_df2 = bayes_to_swot_df(reach_average_data, swot_df)
    # reconstruct profiles from PTs
    #reconst_match_df = reconstruct_profiles_from_pts(pt_df, good_drifts, swot_df2, sword_df, reach_average_df)
    reconst_match_df = reconstruct_profiles_from_pts(pt_df, good_drifts, swot_df, sword_df, reach_average_df)
    plt.show()
    # plot global stats
    #plot_global_stats(reconst_match_df)
 
if __name__ == "__main__":
    main()

