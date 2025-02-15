'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent Williams
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

import rivscale.data

########## Sept 2024 stretch-based processing
def filter_node_qual(df, height=True, area=False, dark_thresh=0.8):
    """
    filter out swot data based on quality, dark_frac, ice,
    location in swath etc...
    """
    # filter out swath edges
    #df = df[np.abs(df['xtrk_dist']) > 10000]
    #df = df[np.abs(df['xtrk_dist']) < 60000]
    df = df[np.bitwise_and(df['node_q_b'], 2**13) == 0]
    df = df[np.bitwise_and(df['node_q_b'], 2**14) == 0]
    # filter out high dark frac
    df = df[df['dark_frac'] < dark_thresh]
    # filter out ice
    df = df[df['ice_clim_f']==0]
    # filter out bad qual
    df = df[df['node_q'] < 3]
    # drop xovr_cal_q == 2
    df = df[df['xovr_cal_q'] < 2]
    # now drop bitwise qual if commanded
    if height:
        # fill values
        df = df[df['wse'] > -99999999.0]
        # geolocation_qual_degraded
        df = df[np.bitwise_and(df['node_q_b'], 2**19) == 0]
        # wse outlier
        df = df[np.bitwise_and(df['node_q_b'], 2**23) == 0]
    if area:
        df = df[df['area_total'] > -99999999.0]
        # classification_qual_degraded
        df = df[np.bitwise_and(df['node_q_b'], 2**18) == 0]
    return df

def filter_bad_stretch_stack(
        stretch_stack_in,
        wse_dark_thresh=0.8,
        width_dark_thresh=0.8):
    """
    Filter the streach data based on comparing uncertainty to local variability
    for both wse and width.

    inputs:
        stretch_data_in = a RiverStretchData instance with populated input data

    outputs:
        streach_data    = copy of stretch_data_in with bad values set to NaN

    NOTE: User should first drop the bad stuff based on node_q_b and
    other standard node quality indicators before creating the
    StretchStack object since we dont carry all those
    indicators in the StretchStack object to do it here
    """
    #TODO: make this a class method
    # find the anomolous wse data by looking for inconsistency between
    # node uncertainty and local node variability (TODO: but ignoring
    # nodes near actual diconstinuities?)
    stretch_stack = stretch_stack_in.copy()
    wse = stretch_stack['wse']
    wse_ref_1d = rivscale.estimate.get_med_profile(
        stretch_stack['wse'], stretch_stack['dist_out'])
    wse_std = rivscale.estimate.get_local_std(wse, wse_ref_1d)
    bad_wse_mask = wse_std / stretch_stack['wse_u'] > 100
    stretch_stack['wse'][bad_wse_mask] = np.nan
    # also drop the very noisy nodes
    noisy_wse_mask = stretch_stack['wse_u'] > 0.5
    stretch_stack['wse'][noisy_wse_mask] = np.nan
    # now do the width
    width = stretch_stack['width']
    width_ref_1d = rivscale.estimate.get_med_profile(
        stretch_stack['width'], stretch_stack['dist_out'])
    width_std = rivscale.estimate.get_local_std(width, width_ref_1d)
    bad_width_mask = width_std / stretch_stack['width_u'] > 100
    stretch_stack['width'][bad_width_mask] = np.nan
    # also do the dark_thresh flagging
    dark_frac = stretch_stack['dark_frac']
    stretch_stack['wse'][dark_frac > wse_dark_thresh] = np.nan
    stretch_stack['width'][dark_frac > width_dark_thresh] = np.nan
    return stretch_stack

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
    data_out = rivscale.products.StretchStack()
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

############ Old code TODO: delete/revise etc
def drop_underobserved_reaches(drift_df, reaches):
    good_reaches = []
    for reach in reaches:
        this_df = drift_df[drift_df['reach_id']==reach].sort_values('node_id')
        nodes = np.unique(this_df['node_id'])
        if len(nodes) < 10:
            continue
        good_reaches.append(reach)
    return good_reaches

def drop_nonptobserved_reaches(pt_df, reaches):
    good_reaches = []
    for reach in reaches:
        this_df = pt_df[pt_df['reach_id']==reach].sort_values('node_id')
        nodes = np.unique(this_df['node_id'])
        if len(nodes) < 1:
            continue
        good_reaches.append(reach)
    return good_reaches

def list_nonmonotonic_drifts(drift_df, reaches):
    bad_drifts = []
    for reach in reaches:
        this_df = drift_df[drift_df['reach_id']==reach].sort_values('node_id')
        drift_ids = np.unique(this_df['drift_id'])
        for drift_id in drift_ids:
            # compute the area under the curve
            df = this_df[this_df['drift_id']==drift_id]
            diff = np.diff(np.array(df['mean_node_drift_wse_m']))
            dist_diff = np.diff(np.array(df['p_dist_out']))
            deriv = diff / dist_diff
            # normalize derivative to typical node length and convert to cm
            deriv = deriv * np.median(dist_diff) * 100
            #breakpoint()
            # find if slope is significanly negative wrt drift noise (~5cm?)
            nonmono = np.abs(np.sum(deriv[deriv < -15]))# -5
            if nonmono > 0:
                #plt.figure()
                #plt.plot(df['mean_node_drift_wse_m'])
                #plt.figure()
                #plt.plot(deriv)
                #plt.show()
                bad_drifts.append(drift_id)
    return np.unique(bad_drifts)

def flag_drifts(drift_df, pt_df, sword_node_df):
    bad_drifts = []
    reaches = np.unique(drift_df['reach_id'])
    #breakpoint()
    # exclude underobserved reaches
    reaches = drop_underobserved_reaches(drift_df, reaches)
    #also drop reaches that have no PTs
    reaches = drop_nonptobserved_reaches(pt_df, reaches)
    # drop nonmonotonic drifts from all reaches
    bad_drifts = list_nonmonotonic_drifts(drift_df, reaches)
    for drift_id in bad_drifts:
        drift_df = drift_df[drift_df['drift_id'] != drift_id]
    # compute the mean profile
    mean_profile_df = compute_mean_profile(drift_df, reaches, sword_node_df)
    #breakpoint()
    # find the drifts whose shape is very different from the mean profile for each reach
    drift_list = []
    reach_list = []
    for k, reach in enumerate(reaches):
        mean_profile = mean_profile_df['mean_profile'][k]
        #mean_profile = this_mean_profile_df['mean_profile']
        mean_profile_local_node_id = mean_profile_df['local_node_id'][k]
        this_df = drift_df[drift_df['reach_id']==reach].sort_values('node_id')
        drift_ids = np.unique(this_df['drift_id'])
        nodes = np.unique(this_df['node_id'])
        local_nodes0 = node_id_to_local_node_id(nodes)
        #local_nodes0 = [int(str(node_id)[-4:-1]) for node_id in nodes]
        # only keep drifts that fit well to the mean_profile
        for drift_id in drift_ids:
            this_drift_df = this_df[this_df['drift_id']==drift_id]
            # exclude underobserved drifts
            if len(this_drift_df) < 10:
                continue
            wse = np.array(this_drift_df['mean_node_drift_wse_m'])
            this_local_node_id = node_id_to_local_node_id(this_drift_df.node_id)
            #this_local_node_id = [int(str(node_id)[-4:-1]) for
            #    node_id in this_drift_df.node_id]
            n_index = this_local_node_id - mean_profile_local_node_id[0]
            offset = np.nanmean(wse - mean_profile[n_index])
            fit_wse_err = np.zeros_like(mean_profile)
            fit_wse_err[n_index] = wse - (mean_profile[n_index] + offset)
            #plt.figure()
            #plt.subplot(2,1,1)
            #plt.plot(this_drift_df['local_node_id'], this_drift_df['mean_node_drift_wse_m'])
            #plt.plot(mean_profile_local_node_id, mean_profile + offset)
            #plt.grid()
            #plt.ylabel('wse (m)')
            #plt.subplot(2,1,2)
            #plt.plot(mean_profile_local_node_id[n_index], fit_wse_err[n_index]*100,'-x')
            #plt.ylabel('error (cm)')
            #plt.grid()
            #plt.show()
            #breakpoint()
            rmse_cm = np.sqrt(np.mean((fit_wse_err[n_index]*100)**2))
            maxe_cm = np.max(np.abs(fit_wse_err[n_index]*100))
            if maxe_cm < 25:#15:
                # exclude this drift for this reach
                drift_list.append(drift_id)
                reach_list.append(reach)
    d = {
        'drift_id':drift_list,
        'reach_id':reach_list
        }
    return pd.DataFrame(d)

def find_pt_drift_matches(pt_df, drift_df):
    # find all PT drift matches
    match_d = {
        'mean_node_pt_wse_m':[],
        'mean_node_drift_wse_m':[],
        'node_id':[],
        'drift_id':[],
        'pt_serial':[],
        'reach_id':[],
        'pt_time_UTC':[]
        #'time_UTC':[]
        }
    drift_ids = np.unique(drift_df['drift_id'])
    for drift_id in drift_ids:
        this_drift_df = drift_df[drift_df['drift_id']==drift_id].sort_values('time_UTC')
        node_ids = np.array(this_drift_df['node_id'])
        for node_id in node_ids:
            this_df = pt_df[pt_df['node_id']==node_id].sort_values('pt_time_UTC')
            if len(this_df) == 0:
                continue
            #breakpoint()
            pt_time = field_time_to_swot_time(np.array(this_df['pt_time_UTC_str']))
            df = this_drift_df[this_drift_df['node_id']==node_id]
            drift_time = field_time_to_swot_time(np.array(df['time_UTC_str']))
            tdiff0 = np.array(pt_time) - drift_time
            tdiff = np.abs(tdiff0)
            print("time_diff (mins):", tdiff0/60)
            match_time_index = np.argmin(tdiff)
            #breakpoint()
            if tdiff[match_time_index] / 60 < 60:
                # append match
                match_d['mean_node_pt_wse_m'].append(np.array(
                    this_df['mean_node_pt_wse_m'])[match_time_index])
                match_d['mean_node_drift_wse_m'].append(float(df['mean_node_drift_wse_m']))
                match_d['node_id'].append(node_id)
                match_d['drift_id'].append(drift_id)
                match_d['pt_serial'].append(np.array(
                    this_df['pt_serial'])[match_time_index])
                match_d['reach_id'].append(np.array(
                    this_df['reach_id'])[match_time_index])
                match_d['pt_time_UTC'].append(np.array(
                    this_df['pt_time_UTC'])[match_time_index])
                #match_d['time_UTC'].append(np.array(df['time_UTC']))
    return pd.DataFrame(match_d)

def filter_bad_swot(swot_df, swot_node_df):
    # plot some stuff
    #reaches = np.unique(np.array(good_drift_df['reach_id']))
    reaches = np.unique(np.array(swot_df['reach_id']))
    bad_cycles = []
    for reach in reaches:
        #plt.figure()
        this_swot_df = swot_node_df[swot_node_df['reach_id']==reach]
        this_swot_r_df = swot_df[swot_df['reach_id']==reach]
        #breakpoint()
        #lgnd = []
        #for cycle in np.unique(this_swot_df['cycle']):
        #    this_df = this_swot_df[this_swot_df['cycle']==cycle].sort_values('node_id')
        #    plt.plot(this_df['node_id'], this_df['wse'])
        #    lgnd.append('{}'.format(cycle))
        #plt.legend(lgnd)
        #plt.gca().set_prop_cycle(None)
        #bad_cycles = []
        median_wse = np.median(this_swot_r_df['wse'])
        Q1 = np.percentile(this_swot_r_df['wse'], 25)
        Q3 = np.percentile(this_swot_r_df['wse'], 75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5*IQR
        lower = Q1 - 1.5*IQR
        for cycle in np.unique(this_swot_df['cycle']):
            this_df = this_swot_df[this_swot_df['cycle']==cycle].sort_values('node_id')
            this_r_df = this_swot_r_df[this_swot_r_df['cycle']==cycle]
            wse = np.array(this_r_df['wse'])
            if len(wse) == 0:
                continue
            #breakpoint()
            #plt.plot(this_df['node_id'], np.zeros_like(np.array(this_df['wse'])) + wse[0], '-')
            outlier = False
            if wse > upper:
                 bad_cycles.append(cycle)
                 outlier = True
            elif wse < lower:
                 bad_cycles.append(cycle)
                 outlier = True
            typ = ':'
            if outlier:
                typ = '-'
            #plt.plot(this_df['node_id'], np.zeros_like(np.array(this_df['wse'])) + wse[0], typ)

        #plt.figure()
        #plt.plot(this_swot_r_df['cycle'], this_swot_r_df['wse'],'o')

    #plt.show()
    # drop bad swot cycles
    for bad_cycle in np.unique(bad_cycles):
         swot_df = swot_df[swot_df['cycle'] != bad_cycle]
         swot_node_df = swot_node_df[swot_node_df['cycle'] != bad_cycle]
    # also throw out the cases with bad qual...
    #swot_df = swot_df[swot_df['reach_q'] < 2]
    return swot_df, swot_node_df


def fix_pt_jumps(pt_df, drift_df, good_drift_df, match_df):
    pt_ids = np.unique(pt_df['pt_serial'])
    pt_times = np.unique(pt_df['pt_time_UTC'])
    time_inds  = np.arange(len(pt_times))
    #mean_diff = np.zeros(len(pt_times))
    #num_diff = np.zeros(len(pt_times)) + np.nan
    diff_arr = np.zeros((len(pt_ids), len(pt_times)))
    diff_t_arr = np.zeros((len(pt_ids), len(pt_times)))
    wse_arr = np.zeros((len(pt_ids), len(pt_times))) + np.nan
    node_id_arr = np.zeros(len(pt_ids))
    reach_id_arr = np.zeros(len(pt_ids))
    pt_serial_arr = np.zeros(len(pt_ids))
    k = 0
    pt_split_df = None
    for pt_id in pt_ids:
        this_df = pt_df[pt_df['pt_serial']==pt_id].sort_values('pt_time_UTC')
        wse = np.array(this_df['mean_node_pt_wse_m'])
        #wse0 = wse.copy()
        #wse = scipy.ndimage.gaussian_filter1d(wse, 5)
        #wse = scipy.ndimage.uniform_filter1d(wse, 31)
        wse = scipy.ndimage.median_filter(wse, 31)
        pt_time = field_time_to_swot_time(np.array(this_df['pt_time_UTC_str']))
        diff = np.diff(wse, prepend=wse[0])
        diff_t = np.diff(pt_time, prepend=pt_time[0])
        diff[np.isnan(diff)==1] = 0
        deriv = diff/diff_t * 100 * 60
        diff2 = np.diff(deriv, prepend=deriv[0])
        #breakpoint()
        #med = wse.copy()
        #med[~np.isfinite(wse)] = 0
        #med = np.median_filter(med, size = 10)
        #diff_med = 
        #diff2_med = np.diff(diff2, prepend=diff2[0])
        # split this hydrograph into separate sections
        #inds = np.where(np.logical_and(np.abs(diff2) > 0.01, np.abs(diff) > 0.01))
        #inds = np.where(np.abs(diff2) > 0.02)
        inds = np.where(np.abs(diff2) > 0.05)
        #inds = np.where(np.abs(diff) > 0.1)
        inds2 = np.array([0]+list(inds[0])+[len(diff2)])
        t = this_df['pt_time_UTC']
        y = this_df['mean_node_pt_wse_m']
        #y = diff2
        #y = diff
        #plt.plot(t, y, '-x')
        #breakpoint()
        i = 0
        for ind0, ind1 in zip(inds2[:-1], inds2[1:]):
            #plt.plot(t[ind0:ind1], y[ind0:ind1], '-x')
            this_pt_split_df = this_df.copy()
            this_pt_split_df['pt_serial'] = str(this_df['pt_serial']) + '_{}'.format(i)
            tmp = np.array(this_df['mean_node_pt_wse_m']) + np.nan
            tmp[ind0:ind1] = y[ind0:ind1]
            this_pt_split_df['mean_node_pt_wse_m'] = tmp
            if pt_split_df is None:
                pt_split_df = this_pt_split_df
            else:
                pt_split_df = pd.concat([pt_split_df, this_pt_split_df], ignore_index=True)
            i = i + 1
        #lgnd.append(pt_id)
        this_pt_times = np.array(this_df['pt_time_UTC'])
        this_time_inds = np.zeros(len(this_pt_times))
        for ind, time in zip(time_inds, pt_times):
            this_ind = np.where(this_pt_times==time)
            this_time_inds[this_ind] = ind
        #breakpoint()
        this_time_inds = this_time_inds.astype(int)
        #mean_diff[this_time_inds] = mean_diff[this_time_inds] + diff
        #num_diff[this_time_inds] = num_diff[this_time_inds] + 1
        diff_arr[k, this_time_inds] = diff
        diff_t_arr[k, this_time_inds] = diff_t
        wse_arr[k, this_time_inds] = wse
        node_id_arr[k] = np.array(this_df['node_id'])[0]
        pt_serial_arr[k] = np.array(this_df['pt_serial'])[0]
        reach_id_arr[k] = np.array(this_df['reach_id'])[0]
        k = k + 1
    median_diff = np.nanmedian(diff_arr, axis=0)
    #mean_diff = mean_diff / num_diff
    #mean_diff[np.isnan(mean_diff)==1] = 0
    mean_diff = np.nanmean(diff_arr, axis=0)
    mean_diff[np.isnan(mean_diff)==1] = 0
    median_diff[np.isnan(median_diff)==1] = 0
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('dwse_dt (cm/min)')
    plt.title('pt dwse/dt vs time')

    plt.figure()
    for pt_id in np.unique(pt_split_df['pt_serial']):
        this_df = pt_split_df[pt_split_df['pt_serial']==pt_id].sort_values('pt_time_UTC')
        plt.plot(this_df['pt_time_UTC'], this_df['mean_node_pt_wse_m'])
    M,N = np.shape(diff_arr)
    plt.figure()
    diff_arr_cpy = diff_arr.copy()
    diff_arr_cpy[np.abs(diff_arr_cpy/diff_t_arr *100*60) > 0.1] = 0
    wse_fix_arr = wse_arr.copy()
 
    for k in range(M):
        wse1 = wse_arr[k,:]
        wse_arr0 = wse1[np.isfinite(wse1)]
        #plt.plot(pt_times, diff_arr_cpy[k,:])
        wse_fix_arr[k,:] =  np.cumsum(diff_arr_cpy[k,:]) + wse_arr0[0]
        plt.plot(pt_times, wse_fix_arr[k,:])
    #plt.gca().set_prop_cycle(None)
    for k in range(M):
        wse1 = wse_arr[k,:]
        wse_arr0 = wse1[np.isfinite(wse1)]
        #plt.plot(pt_times, diff_arr_cpy[k,:])
        plt.plot(pt_times, wse_arr[k,:],'--')
    #plt.figure()
    #plt.gca().set_prop_cycle(None)
    #for k in range(M):
    #    wse1 = wse_arr[k,:]
    #    wse_arr0 = wse1[np.isfinite(wse1)]
    #    plt.plot(pt_times,  np.cumsum(median_diff) + wse_arr0[0], '--')
    
    plt.figure()
    plt.plot(pt_times, mean_diff)
    #plt.plot(pt_times, mean_diff2)
    plt.plot(pt_times, median_diff)
    plt.xlabel('time')
    plt.ylabel('mean dwse_dt (cm/min)')
    plt.title('mean pt dwse/dt vs time')

    #breakpoint()
    plt.figure()
    plt.plot(pt_times, np.cumsum(mean_diff))
    #plt.plot(pt_times, np.cumsum(mean_diff2))
    plt.plot(pt_times, np.cumsum(median_diff))
    plt.xlabel('time')
    plt.ylabel('mean dwse_dt (cm/min)')
    plt.title('mean pt dwse/dt vs time')

    plt.figure()
    for pt_id in pt_ids:
        this_df = pt_df[pt_df['pt_serial']==pt_id].sort_values('pt_time_UTC')
        sigma = np.nanmean(np.array(this_df['sigma_pt_correction_m']))
        typ = '-'
        if sigma > 0.1:
            typ = '-x'
        plt.plot(this_df['pt_time_UTC'], this_df['mean_node_pt_wse_m'], typ)
        #lgnd.append(pt_id)
    # 
    plt.figure()
    for pt_id in pt_ids:
        this_df = pt_df[pt_df['pt_serial']==pt_id].sort_values('pt_time_UTC')
        plt.plot(this_df['pt_time_UTC'], this_df['mean_node_pt_wse_m'])
        #plt.plot(, )
        #lgnd.append(pt_id)
    #breakpoint()
    # TODO: break up the hydrographs into segments and adjust/estimate
    # an offset for each based on the drifts

    node_ids = np.unique(pt_df['node_id'])
    lgnd = []
    plt.figure();
    for node_id in node_ids:
        this_df = pt_df[pt_df['node_id']==node_id].sort_values('pt_time_UTC')
        plt.plot(this_df['pt_time_UTC'], this_df['mean_node_pt_wse_m']);
        lgnd.append(node_id)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('wse (m)')
    plt.title('pt wse vs time')

    #reaches = np.unique(drift_df['reach_id'])
    #good_reaches = []
    #for reach in reaches:
    #    # discard reaches with not enough drift coverage
    #    this_drift_df = drift_df[drift_df['reach_id']==reach]
    #    if len(this_drift_df['node_id']) >10:
    #        # discard reaches with no PTs
    #        this_pt_df = pt_df[pt_df['reach_id']==reach]
    #        if len(this_pt_df) > 0:
    #            good_reaches.append(reach)
    #reaches = good_reaches
    #breakpoint()
    # WM good reaches (used in the PT comparisons)
    #reaches = [78220000211,78220000221,78220000231]

    reaches = np.unique(np.array(good_drift_df['reach_id']))
    node_ids = np.unique(pt_df['node_id'])
    for reach in reaches:
        this_pt_df = pt_df[pt_df['reach_id']==reach]
        #this_pt_df = pt_split_df[pt_split_df['reach_id']==reach]
        this_match_df = match_df[match_df['reach_id']==reach]
        lgnd = []
        plt.figure();
        for node_id in node_ids:
            this_df = this_pt_df[this_pt_df['node_id']==node_id].sort_values('pt_time_UTC')
            plt.plot(this_df['pt_time_UTC'], this_df['mean_node_pt_wse_m']);
            lgnd.append(node_id)
        plt.gca().set_prop_cycle(None)
        for node_id in node_ids:
            this_df = this_match_df[this_match_df['node_id']==node_id].sort_values('pt_time_UTC')
            plt.plot(this_df['pt_time_UTC'], this_df['mean_node_drift_wse_m'], 'o')
        plt.grid()
        plt.xlabel('time')
        plt.ylabel('wse (m)')
        plt.title('pt wse vs time, reach: {}'.format(reach))

    for reach in reaches:
        this_df = drift_df[
            drift_df['reach_id']==reach].sort_values('local_node_id')
        # drop the known bad drifts for this reach
        this_good_drift_df = good_drift_df[good_drift_df['reach_id']==reach]
        good_drifts_t = np.unique(this_good_drift_df['drift_id'])
        all_drifts_t = np.unique(this_df['drift_id'])
        bad_drifts_t = list(set(all_drifts_t) - set(good_drifts_t))
        for bad_drift in bad_drifts_t:
            this_df = this_df[this_df['drift_id'] != bad_drift]
        # now plot
        plt.figure()
        drift_ids = np.unique(this_df['drift_id'])
        lgnd = []
        for drift_id in drift_ids:
            df = this_df[this_df['drift_id']==drift_id]
            lgnd.append('{}'.format(np.array(df['time_UTC'])[0])[0:10])
            #lgnd.append('{}'.format(drift_id))
            plt.plot(df['local_node_id'], df['mean_node_drift_wse_m'], '-x')
        plt.legend(lgnd)
        plt.grid()
        plt.xlabel('node_id (local)')
        plt.ylabel('wse (m)')
        plt.title('reach: {}'.format(reach))

    #
    #reaches = np.unique(np.array(good_drift_df['reach_id']))
    #node_ids = np.unique(pt_df['node_id'])
    for reach in reaches:
        #this_pt_df = pt_df[pt_df['reach_id']==reach]
        this_pt_df = pt_split_df[pt_split_df['reach_id']==reach]
        lgnd = []
        plt.figure();
        #for node_id in node_ids:
        #    this_df = this_pt_df[this_pt_df['node_id']==node_id].sort_values('pt_time_UTC')
        #    plt.plot(this_df['pt_time_UTC'], this_df['mean_node_pt_wse_m']);
        #    lgnd.append(node_id)
        for pt_id in np.unique(this_pt_df['pt_serial']):
            this_df = this_pt_df[this_pt_df['pt_serial']==pt_id].sort_values('pt_time_UTC')
            plt.plot(this_df['pt_time_UTC'], this_df['mean_node_pt_wse_m']);
        this_match_df = match_df[match_df['reach_id']==reach]
        #breakpoint()
        for node_id in node_ids:
            this_df = this_match_df[this_match_df['node_id']==node_id].sort_values('pt_time_UTC')
            plt.plot(this_df['pt_time_UTC'], this_df['mean_node_drift_wse_m'], 'o')
        plt.grid()
        plt.xlabel('time')
        plt.ylabel('wse (m)')
        plt.title('pt wse vs time, reach: {}'.format(reach))
    #
    for reach in reaches:
        plt.figure();
        lgnd = []
        for k in range(M):
            if reach_id_arr[k] == reach:
                plt.plot(pt_times, wse_fix_arr[k,:])
        this_match_df = match_df[match_df['reach_id']==reach]
        for node_id in node_ids:
            this_df = this_match_df[this_match_df['node_id']==node_id].sort_values('pt_time_UTC')
            plt.plot(this_df['pt_time_UTC'], this_df['mean_node_drift_wse_m'], 'o')
        plt.grid()
        plt.xlabel('time')
        plt.ylabel('wse (m)')
        plt.title('pt wse vs time, reach: {}'.format(reach))

    # TODO: adjust "fixed" pts using the drifts 
    # identify sections that diverge from the typical shape...
    #    also fit the "mean/median" curve to the drifts...?
    #    also see if each PT can be modeled as a shifted/offest version of the same hydrograph?
    #    also reconstruct the two end PTs that are broken...?
    wse_fix_smooth = scipy.ndimage.uniform_filter1d(wse_fix_arr, 31)
    mx_time = np.argmax(wse_fix_smooth[:,0:1000], axis=1) # this alg. only good for NS
    mx_val = np.max(wse_fix_smooth[:,0:1000], axis=1)
    wse_model = wse_fix_arr.copy()
    for k in range(M):
        tmp = np.zeros(len(wse_model[k,:])) + np.nan
        ti = mx_time[k] - np.min(mx_time)
        tmp[0:len(wse_model[k,ti:])] = wse_model[k,ti:]
        wse_model[k,:] = tmp - mx_val[k]
    median_wse_model = np.nanmedian(wse_model, axis=0)

    plt.figure()
    plt.plot(wse_model.T)
    plt.plot(median_wse_model,'--')
    #plt.show()
    # Fit the median_model to the drifts for each pt_node
    #plt.figure()
    wse_est_all = np.zeros_like(wse_model)
    for reach in reaches:
        plt.figure()
        this_match_df = match_df[match_df['reach_id']==reach]
        for node_id in np.unique(this_match_df['node_id']):
            k = np.where(node_id_arr==node_id)[0][0]
            ti = mx_time[k] - np.min(mx_time)
            wse_est0 = np.zeros_like(median_wse_model)
            wse_est0[ti:] = median_wse_model[0:len(wse_est0[ti:])]
            wse_est = np.zeros_like(median_wse_model) + np.nan
            wse_est[ti:] = median_wse_model[0:len(wse_est[ti:])]
            this_df = this_match_df[this_match_df['node_id']==node_id].sort_values('pt_time_UTC')
            this_times = np.array(this_df['pt_time_UTC'])
            this_wses = np.array(this_df['mean_node_drift_wse_m'])
            wse_est_sampled = []
            for t in this_times:
                wse_est_sampled.append(wse_est[pt_times == t][0])
            wse_est_sampled = np.array(wse_est_sampled)
            offset = np.nanmean(this_wses - wse_est_sampled)
            print("k = {}, offset = {}, node_id = {}".format(k, offset, node_id))
            wse_est_all[k,:] = wse_est0 + offset
            #breakpoint()
            plt.plot(pt_times, wse_est + offset)
            plt.plot(this_times, wse_est_sampled + offset, 'x')
            plt.plot(this_times, this_wses, 'o')
            #plt.plot(this_df['pt_time_UTC'], this_df['mean_node_drift_wse_m'], 'o')
        plt.grid()
        plt.xlabel('time')
        plt.ylabel('wse (m)')
        plt.title('pt wse vs time, reach: {}'.format(reach))
    # write out new "fixed PT files"
    for pt_id in np.unique(pt_df['pt_serial']):
        this_df = pt_df[pt_df['pt_serial']==pt_id].sort_values('pt_time_UTC')
        k = np.where(pt_serial_arr==pt_id)[0][0]
        #print("k = ",k)
        wse_out = []
        for t in np.array(this_df['pt_time_UTC']):
            j = np.where(pt_times == t)[0][0]
            wse_out.append(wse_est_all[k, j])
        wse_out = np.array(wse_out)
        #print('wse_est_all[k,:] = ', wse_est_all[k,:])
        #print('wse_out = ', wse_out)
        this_df['mean_node_pt_wse_m'] = wse_out
        fle_name = 'NS_node_dataframes/node_fixed/NS_{}PT_node_wse.csv'.format(pt_id)
        #this_df.to_csv(fle_name)
    #breakpoint()


def flag_swot_node_outliers(swot_node_df):
    node_qs = np.array(swot_node_df['node_q'])
    node_ids = np.array(swot_node_df['node_id'])
    wses = np.array(swot_node_df['wse'])
    reaches = np.unique(swot_node_df['reach_id'])
    for reach in reaches:
        this_df = swot_node_df[swot_node_df['reach_id']==reach].sort_values('node_id')
        these_nodes = np.array(this_df['local_node_id']) - 1
        full_nodes = this_df['node_id']
        #all_nodes = np.unique(these_nodes)
        #swot_node_ids = np.array(this_sword_df['node_id'])
        #swot_nodes = np.array(node_id_to_local_node_id(swot_node_ids)) - 1
        ##swot_nodes = np.array([int(str(node_id)[-4:-1]) for
        ##    node_id in swot_node_ids])
        #num_drifts = np.zeros_like(swot_nodes)
        ## find the nodes with the most drifts (hopefully all of them)
        #for node in swot_nodes:
        #    num_drifts[node] = len(these_nodes[these_nodes==node])
        #drift_ids = np.unique(this_df['cycle'])
        ## just average for now... 
        #mean_profile = np.ones_like(swot_nodes) + np.nan
        #num_profile = np.zeros_like(swot_nodes)
        
        for node, node_id in zip(these_nodes, full_nodes):
            node_df = this_df[this_df['local_node_id']==node]
            node_df = node_df[node_df['node_q']<=1]
            if len(node_df)==0:
               continue
            Q1 = np.percentile(node_df['wse'], 25)
            Q3 = np.percentile(node_df['wse'], 75)
            IQR = Q3 - Q1
            upper = Q3 + 1.5*IQR
            lower = Q1 - 1.5*IQR
            # set node_q of outliers to special value
            msk_upper = np.logical_and(node_ids==node_id, wses>upper)
            node_qs[msk_upper] = 6
            msk_lower = np.logical_and(node_ids==node_id, wses<lower)
            node_qs[msk_upper] = 5
            #breakpoint()
            # exclude outliers
            #node_df = node_df[node_df['wse']<upper]
            #node_df = node_df[node_df['wse']>lower]
            #mean_node = np.mean((node_df['wse']))
            #mean_profile[swot_nodes==node] = mean_node
            #num_profile[swot_nodes==node] = len(node_df) 
    swot_node_df['node_q'] = node_qs
    return swot_node_df


def null_outliers(wse_stack):
    wse_stack_out = wse_stack.copy()
    # compute the inter-quartile range of the node wse at each node
    p25 = np.nanpercentile(wse_stack, 25, axis=1)
    p75 = np.nanpercentile(wse_stack, 75, axis=1)
    IQR = p75 - p25
    upper = p75 + 1.5*IQR
    lower = p25 - 1.5*IQR
    # null out individual outlier node measurements
    M,N = np.shape(wse_stack)
    for k in range(M):
        this_node = wse_stack[k,:]
        #breakpoint()
        wse_stack_out[k, this_node > upper[k]] = np.nan
        wse_stack_out[k, this_node < lower[k]] = np.nan
    # now null out entire nodes that are sytematically problematic
    IQR25 = np.nanpercentile(IQR[np.abs(IQR)>0], 25)
    IQR75 = np.nanpercentile(IQR[np.abs(IQR)>0], 75)
    upper2 = IQR75 + 0.5*(IQR75 - IQR25)
    wse_stack_out[IQR > upper2, :] = np.nan
    #for k in range(M):
    #    this_node = wse_stack[k,:]
    #    #breakpoint()
    #    wse_stack_out[k, this_node > upper2] = np.nan
    return wse_stack_out, upper, lower

def filter_stack(full_profile_data, plotem=False):
    #nodes = full_profile_data['nodes']
    wse_stack_out = []
    for k,network in enumerate(full_profile_data['network']):
        wse_stack = full_profile_data['wse_stack'][k]
        wse_stack0, upper0, lower0 = null_outliers(wse_stack)
        if plotem:
            plt.figure()
            plt.plot(wse_stack0, 'o')
            plt.plot(wse_stack, 'x')
            plt.plot(upper0)
            plt.plot(lower0)
        wse_stack_out.append(wse_stack0)
    full_profile_data['wse_stack'] = wse_stack_out
    # TODO: full_profile_data['node_q_stack'] = 
    return full_profile_data

def exclude_bad_stack_cycles(full_profile_data):
    out_data = rivscale.data.init_full_profile_data(full_profile_data.keys())
    for k,network in enumerate(full_profile_data['network']):
        wse_stack = full_profile_data['wse_stack'][k]
        cycles = full_profile_data['cycle'][k]
        this_out_data = rivscale.data.init_full_profile_data(
            full_profile_data.keys())
        for j,cycl in enumerate(cycles):
            wse = wse_stack[:,j]
            if np.sum(np.isfinite(wse)) > len(wse) / 2:
                for key in this_out_data.keys():
                    if 'stack' in key:
                        this_out_data[key].append(full_profile_data[key][k][:,j])
                this_out_data['cycle'].append(cycl)
        #breakpoint()
        for key in this_out_data.keys():
            # check if there are no good data in the reach and exclude the entire reach
            if len(this_out_data['cycle'])==0:
                continue
            # populate arrays for each reach 
            if 'stack' in key:
                out_data[key].append(np.array(this_out_data[key]).T)
            elif key=='cycle':
                out_data[key].append(np.array(this_out_data[key]))
            elif key=='reach':
                out_data[key].append(full_profile_data['reach'][k])
            elif key=='network':
                out_data[key].append(network)
            else:
                out_data[key].append(full_profile_data[key][k])
                
    #breakpoint()
    return out_data

def interp_stack(full_profile_data,
        signal_key='wse_stack', left=None, right=None):
    wse_stack_out = []
    for k, network in enumerate(full_profile_data['network']):
        wse_stack = full_profile_data[signal_key][k]
        #nodes = full_profile_data[node_key][k]
        dist_out = full_profile_data['dist_out'][k]
        this_wse_stack = []
        time_key = 'time_id'
        try:
            full_profile_data[time_key]
        except KeyError:
            time_key='cycle'
        for j,cycl in enumerate(full_profile_data[time_key][k]):
            #breakpoint()
            wse = wse_stack[:,j]
            msk = np.isfinite(wse)
            #breakpoint()
            #print(j)
            #wse_interp = np.zeros(np.shape(nodes)) + np.nan
            #try:
            wse_interp = np.interp(dist_out, dist_out[msk], wse[msk], left=left, right=right)
            #except ValueError:
            #    breakpoint()
            #    wse_interp = np.zeros(np.shape(nodes)) + np.nan
            this_wse_stack.append(wse_interp)
        wse_stack_out.append(np.array(this_wse_stack).T)
    full_profile_data[signal_key+'_interp'] = wse_stack_out
    return full_profile_data


def replace_signal_stats(stats, full_profile_data, sword_node_df):
    stats_out = {}
    for key in stats:
        stats_out[key] = stats[key].copy()
    for k,network in enumerate(stats['network']):
        # replace the Ry
        #this_sword_node_df = sword_node_df[sword_node_df['network']==network].sort_values('dist_out')
        #p_dist_out_list=[]
        #for reach in full_profile_data['reach'][k]:
        #    this_sword_node_df = sword_node_df[sword_node_df['reach_id']==reach].sort_values('dist_out')
        #    p_dist_out_list.append(np.array(this_sword_node_df['dist_out']))
        #p_dist_out = np.concatenate(p_dist_out_list)
        #stats_out['Ry'][k] = exponential_cov(p_dist_out)
        # replace the mean with a linear fit
        ind = np.where(np.array(full_profile_data['network'])==network)
        if len(ind)==0:
            continue #dont replace mean
        ind = ind[0][0]
        wse_stack = full_profile_data['wse_stack'][ind]
        wse_u_stack = full_profile_data['wse_u_stack'][ind]
        p_dist_out = full_profile_data['dist_out'][ind]
        stats_out['Ry'][k] = rivscale.reconstruct.exponential_cov(p_dist_out)
        M,N = np.shape(wse_stack)
        prior_wse = np.zeros_like(wse_stack)
        for j in range(N):
            prior_wse[:,j] = rivscale.reconstruct.linear_fit_mean(p_dist_out, wse_stack[:,j], wse_u_stack[:,j])
        stats_out['mean'][k] = np.mean(prior_wse, axis=1)
    return stats_out

def detrend(full_profile_data, sword_node_df):
    prior_wse_fit = []
    detrended = []
    for k,network in enumerate(full_profile_data['network']):
        #this_sword_node_df = sword_node_df[sword_node_df['reach_id']==reach].sort_values('dist_out')
        #p_dist_out = np.array(this_sword_node_df['dist_out'])
        # replace the mean with a linear fit
        ind = np.where(np.array(full_profile_data['network'])==network)
        if len(ind)==0:
            continue #dont replace mean
        ind = ind[0][0]
        wse_stack = full_profile_data['wse_stack'][ind]
        wse_u_stack = full_profile_data['wse_u_stack'][ind]
        p_dist_out = full_profile_data['dist_out'][ind]
        M,N = np.shape(wse_stack)
        prior_wse = np.zeros_like(wse_stack)
        for j in range(N):
            prior_wse[:,j] = rivscale.reconstruct.linear_fit_mean(p_dist_out, wse_stack[:,j], wse_u_stack[:,j])
        prior_wse_fit.append(prior_wse)
        wse_stack = full_profile_data['wse_stack'][k]
        detrend = wse_stack - prior_wse
        detrend[np.isnan(wse_stack)] = 0
        detrended.append(detrend)
    full_profile_data['wse_stack_linfit'] = prior_wse_fit
    full_profile_data['wse_stack_detrend'] = detrended
    return full_profile_data



def get_connected_networks(sword_df, d_up, d_down):
    this_sword_df = sword_df.sort_values('dist_out')
    reach_ids = np.array(this_sword_df['reach_id'])
    network_list = []
    while len(reach_ids)>0:
        lst = [reach_ids[0],] # start with the closest to outlet
        while True:
            #breakpoint()
            rch = str(lst[-1])
            if rch.endswith('4'):
                # break connectivity at dams and dont include dam reaches
                break
            if rch not in d_up:
                # break when next reach not in list of remaining reaches
                break
            this_d_up = d_up[rch][d_up[rch]>0]
            if len(this_d_up)==0:
                # break if there are no more upstream reaches
                break
            # TODO handle multiple up-stream reaches
            lst.append(this_d_up[0])
            #breakpoint()
        reach_ids = list(set(reach_ids) - set(lst))
        network_list.append(lst)
    return network_list

def modify_uncert(full_profile_data):
    """
    set the uncert high for nodes close to bad/nan ones 
    """
    out_profile_data = rivscale.data.init_full_profile_data(
        full_profile_data.keys())
    for key in full_profile_data:
        out_profile_data[key] = full_profile_data[key].copy()
    for k, network in enumerate(full_profile_data['network']):
        wse_stack = full_profile_data['wse_stack'][k]
        wse_u_stack = full_profile_data['wse_u_stack'][k]
        bad_msk = np.zeros_like(wse_stack)
        bad_msk[np.isnan(wse_stack)] = 1
        tmp = scipy.ndimage.uniform_filter1d(bad_msk, 10, axis=0)
        scale = 1 + 40 * tmp
        wse_u_stack = wse_u_stack * scale
        wse_u_stack[np.isnan(wse_stack)] = 1e3
        out_profile_data['wse_u_stack'][k] = wse_u_stack
    return out_profile_data

def find_pt_swot_matches(pt_df, full_profile_data):
    # find all PT drift matches
    match_d = {
        'pt_wse_m':[],
        'swot_wse_m':[],
        'node_id':[],
        'cycle':[],
        'pt_serial':[],
        'reach_id':[],
        'pt_time_UTC':[],
        'dist_out':[],
        'network':[]
        #'time_UTC':[]
        }
    for k,network in enumerate(full_profile_data['network']):
        #breakpoint()
        for n,node_id in enumerate(full_profile_data['node_id'][k]):
            this_df = pt_df[pt_df['node_id']==node_id].sort_values('pt_time_UTC')   
            #breakpoint()
            if len(this_df) == 0:
                continue
            #breakpoint()
            pt_time = field_time_to_swot_time(np.array(this_df['pt_time_UTC_str'])) 
            #swot_time = field_time_to_swot_time(full_profile_data['time_stack'][k])
            swot_time = np.nanmedian(full_profile_data['time_stack'][k], axis=0)
            cycle = full_profile_data['cycle'][k]
            
            for j,cycl in enumerate(cycle):
                tdiff0 = np.array(pt_time) - swot_time[j]
                tdiff = np.abs(tdiff0)
                print("time_diff (mins):", tdiff0/60)
                match_time_index = np.argmin(tdiff)
                #breakpoint()
                if tdiff[match_time_index] / 60 < 60:
                    # append match
                    match_d['pt_wse_m'].append(np.array(
                        this_df['mean_node_pt_wse_m'])[match_time_index])
                    match_d['swot_wse_m'].append(full_profile_data['wse_stack'][k][n,j])
                    match_d['dist_out'].append(full_profile_data['dist_out'][k][n])
                    match_d['network'].append(full_profile_data['network'][k])
                    match_d['node_id'].append(node_id)
                    match_d['cycle'].append(cycl)
                    match_d['pt_serial'].append(np.array(
                        this_df['pt_serial'])[match_time_index])
                    match_d['reach_id'].append(np.array(
                        this_df['reach_id'])[match_time_index])
                    match_d['pt_time_UTC'].append(np.array(
                        this_df['pt_time_UTC'])[match_time_index])
                #match_d['time_UTC'].append(np.array(df['time_UTC']))
    #breakpoint()
    return pd.DataFrame(match_d)
