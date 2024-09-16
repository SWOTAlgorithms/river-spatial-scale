
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

########## Sept 2024, modified to use product class for river stretch processing
def get_med_profile(signal, dist_out, kernel_size=35):
    """
    This function estimates a river profile from multitemporal SWOT measurements
    over a connected river stretch (potentially mutu-reach section of river sampled at
    the nodes).

    inputs:
    signal   = 2D multitemporal array of along-river data (e.g., node wse, node width)
    dist_out = 1D along-river array of connected node distance to outlet
    
    output:
    med_filt = median profile with holes onterpolated over and spatially median-filter smoothed
    """
    med = np.nanmedian(signal, axis=1)
    msk = ~np.isnan(med)
    # interplate over holes (but don't extrapolate. e.g., nan-fill outside)
    med_interp = np.interp(
        dist_out, dist_out[msk], med[msk], left=np.nan, right=np.nan)
    # do along-river smoothing, preserving discontinuitites
    med_filt = scipy.signal.medfilt(med_interp, kernel_size=kernel_size)
    return med_filt

def get_local_std(signal, signal_ref_1d, size=10):
    """
    This function computes the std of neighborhood of connected nodes (for each node).

    inputs:
    signal        = 2D multitemporal array of along-river data (e.g., node wse, node width)
    signal_ref_1d = reference profile, e.g., output from get_med_profile()
    size          = length of neighborhood window
    
    output:
    std           = the local standard deviation
    """
    ref = np.broadcast_to(signal_ref_1d, np.shape(signal.T)).T
    anom = signal - ref
    anom_fill = anom.copy()
    anom_fill[~np.isfinite(anom)] = 0
    mask = np.ones_like(anom)
    mask[~np.isfinite(anom)] = 0
    cnt = scipy.ndimage.uniform_filter1d(mask, size, axis=0)
    mn = scipy.ndimage.uniform_filter1d(anom_fill, size, axis=0) / cnt
    mn[cnt==0] = 0
    sm = scipy.ndimage.uniform_filter1d((anom_fill - mn)**2, size, axis=0) / cnt
    std = np.sqrt(sm)
    return std

def get_stretch_stats(stretch_data_in, signal_key='wse', percentiles=[5, 25, 32, 50, 68, 75, 95]):
    """
    This function computes statistics of the multitemporal data along the time dimension.

    inputs:
    streach_data_in = a RiverStreachData object that is populated with input data
    signal_key      = 'wse' or 'width' etc
    percentiles     = list of percentiles to compute
    
    output:
    streach_data    = copy of streach_data_in with the mean std, and percentile fields populated
    """
    stretch_data = stretch_data_in.copy()
    stretch_data['{}_mean'.format(signal_key)] = np.nanmean(stretch_data[signal_key], axis=1)
    stretch_data['{}_std'.format(signal_key)] = np.nanstd(stretch_data[signal_key], axis=1)
    stretch_data['percentiles'] = np.array(percentiles)
    ptiles = np.zeros((
        len(stretch_data['{}_mean'.format(signal_key)]),
        len(percentiles),
        )) + np.nan
    for k, ptile in enumerate(percentiles):
        ptiles[:,k] = np.nanpercentile(stretch_data[signal_key], ptile, axis=1)
    stretch_data['{}_percentiles'.format(signal_key)] = ptiles
    return stretch_data

# TODO: put functions to estimate scale parameters from the multitemporal streach data

########## Older code ... TODO: clean-up/delete/revise
def compute_mean_profile(drift_df, reaches, sword_node_df):
    # compute the mean profile handling missing data
    d = {
        'mean_profile':[],
        'reach_id':[],
        'node_id':[],
        'local_node_id':[],
        }
    for reach in reaches:
        this_sword_df = sword_node_df[sword_node_df['reach_id']==reach].sort_values('node_id')
        this_df = drift_df[drift_df['reach_id']==reach].sort_values('node_id')
        these_nodes = np.array(this_df['local_node_id']) - 1
        all_nodes = np.unique(these_nodes)
        swot_node_ids = np.array(this_sword_df['node_id'])
        swot_nodes = node_id_to_local_node_id(swot_node_ids)
        #swot_nodes = np.array([int(str(node_id)[-4:-1]) for
        #    node_id in swot_node_ids])
        num_drifts = np.zeros_like(this_sword_df['node_id'])
        # find the nodes with the most drifts (hopefully all of them)
        for node in all_nodes:
            num_drifts[node] = len(these_nodes[these_nodes==node])
        drift_ids = np.unique(this_df['drift_id'])
        # just average for now... 
        mean_profile = np.ones_like(swot_nodes) + np.nan
        for node in swot_nodes:
            node_df = this_df[this_df['local_node_id']==node]
            mean_node = np.mean((node_df['mean_node_drift_wse_m']))
            mean_profile[swot_nodes==node] = mean_node
        mean_profile = np.array(mean_profile)
        d['mean_profile'].append(mean_profile)
        d['reach_id'].append(np.zeros_like(swot_nodes) + reach)
        d['node_id'].append(swot_node_ids)
        d['local_node_id'].append(swot_nodes)
        
        #plt.figure()
        #for drift_id in drift_ids:
        #    df = this_df[this_df['drift_id']==drift_id]
        #    plt.plot(df['local_node_id'], df['mean_node_drift_wse_m'])
        #    plt.plot(swot_nodes, mean_profile,'-x')
        #plt.show()
        #breakpoint()
        #mean_profiles.append(mean_profile)
    return pd.DataFrame(d)

def compute_mean_swot_profile(swot_node_df, reaches, sword_node_df):
    # compute the mean profile handling missing data
    d = {
        'mean_profile':[],
        'num_profile':[],
        'IQR_profile':[],
        'reach_id':[],
        'node_id':[],
        'local_node_id':[],
        }
    for reach in reaches:
        this_sword_df = sword_node_df[sword_node_df['reach_id']==reach].sort_values('node_id')
        this_df = swot_node_df[swot_node_df['reach_id']==reach].sort_values('node_id')
        these_nodes = np.array(this_df['local_node_id']) - 1
        all_nodes = np.unique(these_nodes)
        swot_node_ids = np.array(this_sword_df['node_id'])
        swot_nodes = np.array(node_id_to_local_node_id(swot_node_ids)) - 1
        #swot_nodes = np.array([int(str(node_id)[-4:-1]) for
        #    node_id in swot_node_ids])
        num_drifts = np.zeros_like(swot_nodes)
        # find the nodes with the most drifts (hopefully all of them)
        for node in swot_nodes:
            num_drifts[node] = len(these_nodes[these_nodes==node])
        drift_ids = np.unique(this_df['cycle'])
        # just average for now... 
        mean_profile = np.ones_like(swot_nodes) + np.nan
        num_profile = np.zeros_like(swot_nodes)
        # first comput IQR
        IQR_profile = np.zeros(np.shape(swot_nodes))
        for node in swot_nodes:
            node_df = this_df[this_df['local_node_id']==node]
            node_df = node_df[node_df['node_q']<=1]
            if len(node_df)==0:
               continue
            Q1 = np.percentile(node_df['wse'], 25)
            Q3 = np.percentile(node_df['wse'], 75)
            IQR = Q3 - Q1
            IQR_profile[swot_nodes==node] = IQR
        IQR_profile = np.array(IQR_profile)
        # also flag things outside the median IQR
        IQR_med = np.median(IQR_profile[IQR_profile>0])
        print(IQR_med)
        for node in swot_nodes:
            node_df = this_df[this_df['local_node_id']==node]
            node_df = node_df[node_df['node_q']<=1]
            if len(node_df)==0:
               continue
            Q1 = np.percentile(node_df['wse'], 25)
            Q3 = np.percentile(node_df['wse'], 75)
            # flag things outside median IQR
            upper = Q3 + 1.5*IQR_med
            lower = Q1 - 1.5*IQR_med
            #breakpoint()
            # exclude outliers
            node_df = node_df[node_df['wse']<upper]
            node_df = node_df[node_df['wse']>lower]
            mean_node = np.mean((node_df['wse']))
            mean_profile[swot_nodes==node] = mean_node
            num_profile[swot_nodes==node] = len(node_df)
            #IQR_profile[swot_nodes==node] = IQR
        mean_profile0 = np.array(mean_profile)
        num_profile = np.array(num_profile)
        # mask out 
        Q3_IQR = np.percentile(IQR_profile[IQR_profile>0], 75)
        Q1_IQR = np.percentile(IQR_profile[IQR_profile>0], 25)
        Q3_num = np.percentile(num_profile, 75)
        Q1_num = np.percentile(num_profile, 25)
        IQR_IQR = Q3_IQR - Q1_IQR
        IQR_num = Q3_num - Q1_num
        #good_node_mask = np.logical_and(IQR_profile < Q3_IQR, num_profile > Q3_num)
        #good_node_mask = IQR_profile < Q3_IQR + 0.8*IQR_IQR
        #mean_profile = mean_profile + np.nan
        #mean_profile[good_node_mask] = mean_profile0[good_node_mask] 
        d['mean_profile'] = d['mean_profile'] + list(mean_profile)
        d['num_profile'] = d['num_profile'] + list(num_profile)
        d['IQR_profile'] = d['IQR_profile'] + list(IQR_profile)
        d['reach_id'] = d['reach_id'] + list(np.zeros_like(swot_nodes) + reach)
        d['node_id'] = d['node_id'] + list(swot_node_ids)
        d['local_node_id'] = d['local_node_id'] + list(swot_nodes)

    return pd.DataFrame(d)

def estimate_swot_mean_and_covariances(swot_node_df, sword_node_df, plotem=False):
    reaches = np.unique(swot_node_df['reach_id'])
    full_profile_data = {
        'signal':[],
        'signal_orig':[],
        'node_id':[],
        'nodes':[],
        'Ry':[],
        'B':[],
        'reach':[],
        'cycle':[],
        'mean_reach_profile':[]}
    # set up system identification problem for each reach
    for reach in reaches:
        # get "full" profile by linearly interpolating over missing nodes...
        this_df = swot_node_df[
            swot_node_df['reach_id']==reach].sort_values('local_node_id')
        #if reach == 78220000211:
        #    breakpoint()
        # drop the known bad drifts for this reach
        #this_good_drift_df = good_drift_df[good_drift_df['reach_id']==reach]
        #good_drifts_t = np.unique(this_good_drift_df['drift_id'])
        #all_cycles_t = np.unique(this_df['cycle'])
        #bad_drifts_t = list(set(all_drifts_t) - set(good_drifts_t))
        #for bad_drift in bad_drifts_t:
        #    this_df = this_df[this_df['drift_id'] != bad_drift]
        #
        local_node_ids = np.unique(this_df['local_node_id'])
        #breakpoint()
        #nodes = np.arange(np.min(local_node_ids), np.max(local_node_ids)+1)
        this_sword_df = sword_node_df[sword_node_df['reach_id']==reach].sort_values('node_id')
        nodes = node_id_to_local_node_id(this_sword_df['node_id'])
        node_ids = this_sword_df['node_id']
        cycles = np.sort(np.unique(this_df['cycle']))
        signal = []
        signal_orig = []
        signal_cycle = []
        for cycl in cycles:
            df = this_df[this_df['cycle']==cycl]
            # drop bad node data
            this_node_q = np.array(df['local_node_id'])
            df = df[df['node_q']<=1]
            # interpolate to full extent
            this_nodes = np.array(df['local_node_id'])
            this_wse = np.array(df['wse'])
            
            # TODO: actually use the Bayes approach implemented in RiverObs?
            # for now just linearly interpolate
            #full_wse = np.zeros_like(nodes)+np.nan
            #full_wse[this_nodes] = this_wse
            # if fewer than min nodes in drift, exclude
            if len(this_wse) < 10:#len(nodes)*3/4:
                continue
            #breakpoint()
            full_wse0 = np.ones(np.shape(nodes)) + np.nan
            for n,w in zip(this_nodes, this_wse):
                full_wse0[nodes==n] = w
            full_wse = np.interp(nodes, this_nodes, this_wse)
            signal.append(full_wse.T)
            signal_orig.append(full_wse0.T)
            signal_cycle.append(cycl)
        signal = np.array(signal)
        signal_orig = np.array(signal_orig)
        signal_cycle = np.array(signal_cycle)
        if len(signal)==0:
            continue
        plt.figure()
        plt.plot(signal.T)
        plt.title('reach: {}'.format(reach))
        # compute the mean height profile
        mn = np.mean(signal, axis=0)
        # define the zero-mean signal
        #breakpoint()
        M,N = np.shape(signal)
        Y = signal.T-np.tile(mn,(M,1)).T
        # estimate the covariance of the zero-mean signal
        Ry = np.inner(Y, Y) / M
        U, S, V = svd(Ry)
        # create the orthonornmal basis from eigen vectors
        #breakpoint()
        L = len(S[S>1e-10])
        B = U[:,0:L] # use truncated basis
        full_profile_data['signal'].append(signal)
        full_profile_data['signal_orig'].append(signal_orig)
        full_profile_data['node_id'].append(node_ids)
        full_profile_data['nodes'].append(nodes)
        full_profile_data['Ry'].append(Ry)
        full_profile_data['B'].append(B)
        full_profile_data['reach'].append(reach)
        full_profile_data['cycle'].append(np.array(cycles))
        full_profile_data['mean_reach_profile'].append(mn)    
    return full_profile_data

def estimate_mean_and_covariances(drift_df, good_drift_df, sword_node_df, match_df, plotem=False):
    reaches = np.unique(match_df['reach_id'])
    good_drifts = {'signal':[], 'nodes':[], 'Ry':[], 'B':[], 'reach':[], 'mean_reach_profile':[]}
    # set up system identification problem for each reach
    for reach in reaches:
        # get "full" profile by linearly interpolating over missing nodes...
        this_df = drift_df[
            drift_df['reach_id']==reach].sort_values('local_node_id')
        #if reach == 78220000211:
        #    breakpoint()
        # drop the known bad drifts for this reach
        this_good_drift_df = good_drift_df[good_drift_df['reach_id']==reach]
        good_drifts_t = np.unique(this_good_drift_df['drift_id'])
        all_drifts_t = np.unique(this_df['drift_id'])
        bad_drifts_t = list(set(all_drifts_t) - set(good_drifts_t))
        for bad_drift in bad_drifts_t:
            this_df = this_df[this_df['drift_id'] != bad_drift]
        #
        if len(this_df)==0:
            continue
        local_node_ids = np.unique(this_df['local_node_id'])
        #breakpoint()
        #nodes = np.arange(np.min(local_node_ids), np.max(local_node_ids)+1)
        this_sword_df = sword_node_df[sword_node_df['reach_id']==reach].sort_values('node_id')
        nodes = node_id_to_local_node_id(this_sword_df['node_id'])
        drift_ids = np.unique(this_df['drift_id'])
        signal = []
        signal_drift_id = []
        for drift_id in drift_ids:
            df = this_df[this_df['drift_id']==drift_id]
            # interpolate to full extent
            this_nodes = np.array(df['local_node_id'])
            this_wse = np.array(df['mean_node_drift_wse_m'])
            # TODO: actually use the Bayes approach implemented in RiverObs?
            # for now just linearly interpolate
            #full_wse = np.zeros_like(nodes)+np.nan
            #full_wse[this_nodes] = this_wse
            # if fewer than min nodes in drift, exclude
            if len(this_wse) < 10:#len(nodes)*3/4:
                continue
            #breakpoint()
            full_wse = np.interp(nodes, this_nodes, this_wse)
            signal.append(full_wse.T)
            signal_drift_id.append(drift_id)
        signal = np.array(signal)
        signal_drift_id = np.array(signal_drift_id)
        plt.figure()
        plt.plot(signal.T)
        plt.title('reach: {}'.format(reach))
        # compute the mean height profile
        mn = np.mean(signal, axis=0) 
        # define the zero-mean signal
        shp = np.shape(signal)
        if len(shp)!=2:
            breakpoint()
            signal = np.array([signal, signal])
        M,N = np.shape(signal)
        Y = signal.T-np.tile(mn,(M,1)).T
        # estimate the covariance of the zero-mean signal
        Ry = np.inner(Y, Y) / M
        # take the eigen-value decmposition of the signal covariance
        #eigval, W = eigh(Ry)
        U, S, V = svd(Ry)
        # plot the eigen values and principle eigen vectors
        #plt.figure()
        ##plt.plot(10*np.log10(eigval))
        #plt.plot(10*np.log10(S))
        #plt.figure()
        #plt.plot(U[:,0:3])
        #plt.show()
        #breakpoint()
        # create the orthonornmal basis from eigen vectors
        #breakpoint()
        L = len(S[S>1e-10])
        B = U[:,0:L] # use truncated basis
        good_drifts['signal'].append(signal)
        good_drifts['nodes'].append(nodes)
        good_drifts['Ry'].append(Ry)
        good_drifts['B'].append(B)
        good_drifts['reach'].append(reach)
        good_drifts['mean_reach_profile'].append(mn)
        # estimate basis parameters from the pt vector
        this_match_df = match_df[match_df['reach_id']==reach]
        this_drift_ids = np.unique(this_match_df['drift_id'])
        print('reach_id', reach)
        #print(this_drift_ids)
        #breakpoint()
        for drift_id in this_drift_ids:
            this_match_df2 = this_match_df[this_match_df['drift_id']==drift_id].sort_values('node_id')
            L2 = len(this_match_df2)
            if L2 < 1:
                continue
            if L2 < L:
                # truncate even more
                B = U[:,0:L2]
            local_node_id = node_id_to_local_node_id(this_match_df2.node_id)
            #local_node_id = [int(str(node_id)[-4:-1]) for
            #    node_id in this_match_df2.node_id]
            node_index = np.array(local_node_id)-nodes[0]
            drift_wse = np.array(this_match_df2['mean_node_drift_wse_m'])
            pt_wse = np.array(this_match_df2['mean_node_pt_wse_m'])
            #breakpoint()
            #method = 'truncated_basis'
            #method = 'linear_estimator'
            method = 'bayes_estimator'
            if method =='truncated_basis':
                # set up the linear algebra estimators
                signal_hat = truncated_basis_estimator(
                    drift_wse, node_index, mn, B)
                signal_hat_pt = truncated_basis_estimator(
                    pt_wse, node_index, mn, B)
            elif method == 'linear_estimator':
                signal_hat = optimal_linear_estimator(
                    drift_wse, node_index, mn, Ry,
                    np.ones_like(drift_wse)*0.05)
                signal_hat_pt = optimal_linear_estimator(
                    pt_wse, node_index, mn, Ry,
                    np.abs(pt_wse - drift_wse))
            elif method == 'bayes_estimator':
                # do Bayes Reconstruction
                signal_hat, A_inv = bayes_estimator(
                    drift_wse, node_index, mn, Ry,
                    np.ones_like(drift_wse)*0.05)
                signal_hat_pt, A_inv_pt = bayes_estimator(
                    pt_wse, node_index, mn, Ry,
                    #np.ones_like(drift_wse)*0.05)
                    np.abs(pt_wse - drift_wse))
            # plot
            ind0 = np.where(signal_drift_id==drift_id)[0]
            if len(ind0) > 0:
                ind = ind0[0]
            else:
                continue
            # compute the reach average height
            fit_wse = signal[ind,node_index]
            drift_wse_err = drift_wse - fit_wse
            pt_wse_err = pt_wse - fit_wse
            mean_drift_wse = np.mean(drift_wse)
            mean_pt_wse = np.mean(pt_wse)
            mean_fit_wse = np.mean(fit_wse)
            mean_signal = np.mean(signal[ind,:])
            mean_signal_hat = np.mean(signal_hat_pt)
            # plot
            if plotem:
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(signal[ind,:])
                plt.plot(signal_hat,'--')
                plt.plot(signal_hat_pt,':')
                #plt.plot(mn)
                plt.plot(node_index, drift_wse, '^')
                plt.plot(node_index, pt_wse, 'o')
                plt.plot(nodes, np.ones_like(nodes)*mean_drift_wse, ':')
                plt.plot(nodes, np.ones_like(nodes)*mean_pt_wse, ':')
                plt.plot(nodes, np.ones_like(nodes)*mean_signal_hat)
                plt.plot(nodes, np.ones_like(nodes)*mean_signal)
                _, drft0 = os.path.split(drift_id)
                drft_parts = drft0.split('_')
                drft = drft_parts[1] + '_' + drft_parts[4] + '_' + drft_parts[5]
                plt.title('reach: {}, drift:{}'.format(reach, drft))
                plt.ylabel('wse (m)')
                plt.grid()
                plt.legend(['drift', 'drift fit', 'pt fit', 'drift (at pt nodes)',
                    'pt meas.', 'mean drift (at pt nodes)', 'mean pt', 'mean fit',
                    'mean drift'])
                #plt.figure()
                plt.subplot(2,1,2)
                plt.plot((signal[ind,:] - signal[ind,:])*100)
                plt.plot((signal_hat - signal[ind,:])*100, '--')
                plt.plot((signal_hat_pt - signal[ind,:])*100,':')
                plt.plot(node_index, drift_wse_err*100, '^')
                plt.plot(node_index, pt_wse_err*100, 'o')
                #plt.plot(nodes, np.ones_like(nodes)*(mean_signal_hat - mean_signal)*100)
                plt.plot(nodes, np.ones_like(nodes)*(mean_drift_wse - mean_signal)*100, ':')
                plt.plot(nodes, np.ones_like(nodes)*(mean_pt_wse - mean_signal)*100, ':')
                plt.plot(nodes, np.ones_like(nodes)*(mean_signal_hat - mean_signal)*100)
                plt.grid()
                plt.ylabel('wse error (cm)')
    return good_drifts


def stack_mean_and_covariance(full_profile_data):
    stats = {'mean':[], 'median':[], 'Ry':[], 'reach':[], 'network':[]}
    for k, network in enumerate(full_profile_data['network']):
        signal = full_profile_data['wse_stack_interp'][k]
        signal_u = full_profile_data['wse_u_stack'][k]
        mn = np.nanmean(signal, axis=1)
        med = np.nanmedian(signal, axis=1)
        # define the zero-mean signal
        #breakpoint()
        M,N = np.shape(signal)
        Y = signal-np.tile(mn,(N, 1)).T
        Y[np.isnan(signal)] = 0
        count = np.ones((M,N))
        count[np.isnan(signal)] = 0
        # estimate the covariance of the zero-mean signal
        #Ry = np.inner(Y, Y) / N
        C = np.inner(count, count)
        Ry = np.inner(Y, Y) / C
        Ry[C==0] = 0
        
        #breakpoint()
        ## subtract off the known noise covariance
        ## handling resulting negaitve diagonal elements
        #Rv = np.diag(np.nanmean(signal_u, axis=1))
        #Ry_diag = np.diag(np.diag(Ry))
        #R1 = Ry_diag - Rv
        #R1_full = Ry - Rv
        #Rmax = np.diag(np.max(np.abs(R1_full), axis=0))
        ## replace negative diag values with the next biggest magnitude element
        ## in each row
        ##R1[R1<0] = Rmax[R1<0]
        #Ry = Ry - Rv
        #Ry[R1<0] = Rmax[R1<0] + 1e-3
        
        
        #breakpoint()
        stats['mean'].append(mn)
        stats['median'].append(med)
        stats['Ry'].append(Ry)
        stats['reach'].append(full_profile_data['reach'][k])
        stats['network'].append(network)
    return stats

