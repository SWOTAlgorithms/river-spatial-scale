
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

####### Sept 2024, updated for river stretch processing

def reconstruct_stretch(stretch_data_in,
        signal_key='wse', uncert_key='wse_u'):
    # copy the input data to output data
    stretch_data = stretch_data_in.copy()
    # get the bayes parameters
    mn = stretch_data['{}_reference'.format(signal_key)]
    Ry = stretch_data['{}_cov'.format(signal_key)]
    # TODO: should this be local_node_id (i.e., indexing starting at 1)?
    nodes = np.arange(len(stretch_data['node_id']), dtype=int)
    signal_hats = []
    post_covs = []
    bayes_us = []
    time_key = 'time_id'
    # go through each time/cycle observation in the stack 
    for j,cycl in enumerate(stretch_data[time_key]):
        meas = stretch_data[signal_key][:,j]
        meas_u = stretch_data[uncert_key][:,j]
        meas_u[~np.isfinite(meas_u)] = 10**5#10^5
        msk = np.isfinite(meas)
        #breakpoint()
        # check for not enough data
        if np.sum(msk) < 2:
            # dont call bayes, just append nans
            nan_array = np.ones_like(meas) + np.nan
            unc_array = np.zeros_like(meas_u) + 10**5
            signal_hats.append(nan_array)
            post_covs.append(np.diag(unc_array))
            bayes_us.append(unc_array.copy())
        else:
            signal_hat, A_inv = bayes_estimator(
                meas[msk], nodes[msk], mn, Ry,
                np.sqrt(meas_u[msk]))
            signal_hats.append(signal_hat)
            post_covs.append(A_inv)
            bayes_us.append(np.diag(A_inv).copy())
    #breakpoint()
    stretch_data['bayes_{}'.format(signal_key)] = np.array(signal_hats).T
    stretch_data['bayes_{}_post_cov'.format(signal_key)] = np.moveaxis(
        np.array(post_covs), 0, -1)
    stretch_data['bayes_{}_u'.format(signal_key)] = np.array(bayes_us).T
    #breakpoint()
    return stretch_data

def bayes_estimator(meas_wse, node_index, mn, Ry0, meas_noise):
    N = len(mn)
    H = np.zeros((len(node_index), N))
    for i,ni in enumerate(node_index):
        if ni == -30:
            breakpoint()
        H[i,ni] = 1
    X = meas_wse - mn[node_index]
    Rv = np.diag(meas_noise**2)
    Ry_inv = pinv(Ry0)
    Rv_inv = pinv(Rv)
    A = Ry_inv + H.T @ Rv_inv @ H
    A_inv = np.linalg.inv(A)
    K = A_inv @ H.T @ Rv_inv
    K_bar = A_inv @ Ry_inv
    yp = K @ meas_wse
    y_bar =  K_bar @ mn
    signal_hat = yp + y_bar
    return signal_hat, A_inv

def exponential_cov(p_dist_out, char_length_tau=20000, prior_unc_alpha=2.0):
    """
    copied snippets from RiverObs:
    p_dist_out      : Node-level distance to outlet from prior database
    char_length_tau : characteristic length (m)
    prior_unc_alpha : scale to apply to the magnitude of Ry
    """
    Ry0 = np.zeros((len(p_dist_out), len(p_dist_out)))
    for k, d0 in enumerate(p_dist_out):
        t = p_dist_out - d0
        Ry0[k, :] = np.exp(-np.abs(t) / char_length_tau)
    # scale the covariance to trade-off noise.vs "spectral resolution"
    Ry = Ry0 / np.max(Ry0) * prior_unc_alpha ** 2
    return Ry

######## Old data TODO: delete/revise etc

def truncated_basis_estimator(meas_wse, node_index, mn, B):
    X = meas_wse - mn[node_index]
    HB = B[node_index,:]
    HBp = pinv(HB)
    a_hat = np.inner(HBp, X)
    Y_hat = np.inner(B, a_hat)
    signal_hat =  Y_hat + mn
    return signal_hat

def optimal_linear_estimator(meas_wse, node_index, mn, Ry0, meas_noise):
    # this doesnt really work...dont use it...
    #Ry = Ry0 + np.outer(mn, mn)
    Ry = Ry0
    RyH_T = Ry[:,node_index]
    HRyH_T = Ry[node_index,node_index]
    Rv = np.diag((meas_noise)**2)
    X = meas_wse - mn[node_index]
    #X = meas_wse
    HRyH_T_plus_Rv_inv = np.linalg.inv(HRyH_T + Rv)
    A_hat = np.inner(RyH_T, HRyH_T_plus_Rv_inv)
    # normalize A_hat?
    A_hat = A_hat/np.linalg.norm(A_hat)
    signal_hat = np.inner(A_hat, X) + mn
    #signal_hat = np.inner(A_hat, X)
    return signal_hat

def bayes_estimator_old(meas_wse, node_index, mn, Ry0, meas_noise):
    N = len(mn)
    H = np.zeros((len(node_index), N))
    for i,ni in enumerate(node_index):
        if ni == -30:
            breakpoint()
        H[i,ni] = 1
    X = meas_wse - mn[node_index]
    Rv = np.diag(meas_noise**2)
    # Ry is not full rank...how to handle...?
    #tau = 
    #prior_unc_alpha = 
    #Ry_exp = np.zeros((N, N))
    #for k, d0 in enumerate(ss):
    #    t = ss - d0
    #    Ry0[k, :] = np.exp(-np.abs(t) / tau)
    #Ry_exp = Ry_exp / np.max(Ry_exp) * prior_unc_alpha ** 2
    #Ry_reg = np.eye(N)*1e-10
    ####
    #Ry_reg = np.eye(N)*1e-8
    #Ry = Ry0 + Ry_reg # Tikhonov regularization...?
    #Ry_inv = np.linalg.inv(Ry)
    Ry_inv = pinv(Ry0)
    Rv_inv = pinv(Rv)
    A = Ry_inv + H.T @ Rv_inv @ H
    A_inv = np.linalg.inv(A)
    K = A_inv @ H.T @ Rv_inv
    K_bar = A_inv @ Ry_inv
    yp = K @ meas_wse
    y_bar =  K_bar @ mn 
    signal_hat = yp + y_bar
    return signal_hat, A_inv


def reconstruct_profiles_from_pts(pt_df, good_drifts, swot_df, sword_df, reach_average_df):
    # get the SWOT data as well as recontruct the profiles for all tile from the PTs
    d = {
        'wse_err':[],
        'wse_swot':[],
        'wse':[],
        'slope_err':[],
        'slope2_err':[],
        'reach_id':[],
        'time_UTC':[]
        }
    reaches = np.unique(good_drifts['reach'])
    for reach in reaches:
        this_swot_df = swot_df[swot_df['reach_id'] == reach].sort_values('time_UTC')
        this_sword_df = sword_df[sword_df['reach_id'] == reach]
        this_reach_average_df = reach_average_df[reach_average_df['reach_id']==reach]
        reach_ind = good_drifts['reach'].index(reach)
        #breakpoint()
        signal = good_drifts['signal'][reach_ind]
        nodes = good_drifts['nodes'][reach_ind]
        Ry = good_drifts['Ry'][reach_ind]
        B = good_drifts['B'][reach_ind]
        mn = good_drifts['mean_reach_profile'][reach_ind]
        this_pt_df = pt_df[pt_df['reach_id']==reach].sort_values('pt_time_UTC')
        # reconstruct the full profiles at every PT time for this reach
        times = np.unique(this_pt_df['pt_time_UTC'])
        num_nodes = len(nodes)
        WSE = np.ma.zeros((num_nodes, len(times))) + np.ma.masked
        DWSE = np.ma.zeros((num_nodes, len(times))) + np.ma.masked
        for k, time in enumerate(times):
            this_df = this_pt_df[this_pt_df['pt_time_UTC'] == time]
            # reconstruct the full profile from PTs at this time
            M,N = np.shape(B)
            L = len(this_df)
            if L < N:
                B = B[:,0:L]
            local_node_id = node_id_to_local_node_id(this_df.node_id)
            #local_node_id = [int(str(node_id)[-4:-1]) for
            #    node_id in this_df.node_id]
            node_index = np.array(local_node_id)-nodes[0]
            pt_wse = this_df['mean_node_pt_wse_m']
            #wse = truncated_basis_estimator(pt_wse, node_index, mn, B)
            wse, A_inv = bayes_estimator(pt_wse, node_index, mn, Ry, np.ones_like(pt_wse)*0.05)
            WSE[:,k] = wse
            DWSE[:,k] = wse - mn
        plt.figure()
        plt.imshow(WSE, aspect='auto', interpolation='none', cmap='jet')
        plt.colorbar()
        plt.title('reach: {}'.format(reach))
        plt.figure()
        plt.imshow(DWSE, aspect='auto', interpolation='none', cmap='jet')
        plt.colorbar()
        plt.title('reach: {}'.format(reach))
        # compute reach wse and slope vs time
        mean_wse = np.mean(WSE,axis=0)
        plt.figure()
        plt.plot(times, mean_wse)
        plt.plot(this_reach_average_df['reach_times'], this_reach_average_df['reach_wse'])
        plt.plot(this_swot_df['time_UTC'], this_swot_df['wse'], 'o')
        plt.grid()
        plt.title('reach: {}'.format(reach))
        plt.ylabel('reach wse (m)')
        plt.legend(['reconstructed', 'pt', 'swot'])

        reach_length = float(this_sword_df['reach_length'])
        cm_km = (1000*100)
        slp = (WSE[-1,:]-WSE[0,:])/reach_length
        plt.figure()
        plt.plot(times, slp * cm_km)
        plt.plot(this_reach_average_df['reach_times'], this_reach_average_df['reach_slope'] * cm_km)
        plt.plot(this_swot_df['time_UTC'], this_swot_df['slope'] * cm_km, 'o')
        plt.plot(this_swot_df['time_UTC'], this_swot_df['slope2'] *cm_km, '^')
        plt.grid()
        plt.ylabel('reach_slope (cm/km)')
        plt.legend(['reconstructed', 'pt', 'swot slope1', 'swot slope2'])
        plt.title('reach: {}'.format(reach))

        plt.figure()
        plt.scatter(mean_wse, slp * cm_km)
        plt.scatter(this_reach_average_df['reach_wse'], this_reach_average_df['reach_slope'] * cm_km)
        plt.scatter(this_swot_df['wse'], this_swot_df['slope'] * cm_km)
        plt.scatter(this_swot_df['wse'], this_swot_df['slope2'] * cm_km)
        plt.xlabel('reach wse (m)')
        plt.ylabel('reach slope (cm/km)')
        plt.legend(['reconstructed', 'pt', 'swot slope1', 'swot slope2'])
        plt.grid()
        plt.title('reach: {}'.format(reach))
        plt.show()
        #breakpoint()

        # create the error dataframes
        times_swot = np.unique(this_swot_df['time_UTC'])
        #d = {
        #    'wse_err':[],
        #    'slope_err':[],
        #    'slope2_err'
        #    'reach_id':[],
        #    'time_UTC':[]
        #    }
        for time in times_swot:
            # find closest withing half hour
            t = pd.to_datetime(time).strftime('%Y-%m-%d %H:%M:%S')
            ts = pd.to_datetime(times).strftime('%Y-%m-%d %H:%M:%S')
            t0 = field_time_to_swot_time([t])
            ts0 = field_time_to_swot_time(ts)
            t_diff = np.array(ts0) - np.array(t0)
            if np.min(np.abs(t_diff)) < 30*60*60: # minimum within 30 mins
                ind = np.argmin(np.abs(t_diff))
                #breakpoint()
                # add this times "truth" reach WSE and slope(s)
                df = this_swot_df[this_swot_df['time_UTC']==time]
                d['wse_err'].append(float(df['wse']) - mean_wse[ind])
                d['wse_swot'].append(float(df['wse']))
                d['wse'].append(mean_wse[ind])
                d['slope_err'].append(float(df['slope']) - slp[ind])
                d['slope2_err'].append(float(df['slope2']) - slp[ind])
                d['reach_id'].append(reach)
                d['time_UTC'].append(time)
    for var in d.keys():
        d[var] = np.array(d[var])
    reconst_match_df = pd.DataFrame(d)
    return reconst_match_df

def reconstruct_stack(stats, full_profile_data,
        signal_key='wse_stack', uncert_key='wse_u_stack'):
    data_out = rivscale.data.init_full_profile_data()
    # copy the input data to output data
    for key in data_out.keys():
        try:
            data_out[key] = full_profile_data[key].copy()
        except KeyError:
            pass
    data_out[signal_key] = full_profile_data[signal_key].copy()
    data_out[uncert_key] = full_profile_data[uncert_key].copy()
    data_out['post_cov'] = []
    # reconstruct
    for k0,network in enumerate(stats['network']):
        #breakpoint()
        msk = np.where(np.array(stats['network'])==network)
        if len(msk)==0:
            continue
        msk = msk[0]
        if len(msk)==0:
            continue
        msk = msk[0]
        #breakpoint()
        mn = stats['mean'][msk]
        Ry = stats['Ry'][msk]
        # need to handle Ry and full_data different size...
        #breakpoint()
        k = np.where(np.array(full_profile_data['network'])==network)
        if len(k)==0:
            continue
        else:
            k = k[0][0]
        #nodes = full_profile_data['nodes'][k]-1
        nodes = np.arange(len(full_profile_data['nodes'][k]), dtype=int)
        this_signal_hat = []
        this_post_cov = []
        time_key = 'time_id'
        try:
            full_profile_data[time_key]
        except KeyError:
            time_key = 'cycle'
        for j,cycl in enumerate(full_profile_data[time_key][k]):
            meas_wse = full_profile_data[signal_key][k][:,j]
            meas_wse_u = full_profile_data[uncert_key][k][:,j]
            meas_wse_u[~np.isfinite(meas_wse_u)] = 10**5#10^5
            msk = np.isfinite(meas_wse)
            #breakpoint()
            signal_hat, A_inv = bayes_estimator(
                    meas_wse[msk], nodes[msk], mn, Ry,
                    np.sqrt(meas_wse_u[msk]))
            this_signal_hat.append(signal_hat)
            this_post_cov.append(A_inv)
        #data_out['wse_stack'].append(np.array(this_signal_hat).T)
        data_out[signal_key][k] = np.array(this_signal_hat).T
        data_out['post_cov'].append(this_post_cov)
        # TODO: update the wse_u_stack with posterior uncertainty estimate
    return data_out



def exponential_cov_old(p_dist_out, char_length_tau=20000, prior_unc_alpha=2.0):#prior_unc_alpha=0.25):#char_length_tau=10000, prior_unc_alpha=0.5):#1.5):
    """
    copied snippets from RiverObs:
    p_dist_out      : Node-level distance to outlet from prior database
    char_length_tau : characteristic length (m)
    prior_unc_alpha : scale to apply to the magnitude of Ry
    """
    Ry0 = np.zeros((len(p_dist_out), len(p_dist_out)))
    for k, d0 in enumerate(p_dist_out):
        t = p_dist_out - d0
        Ry0[k, :] = np.exp(-np.abs(t) / char_length_tau)
    # scale the covariance to trade-off noise.vs "spectral resolution"
    Ry = Ry0 / np.max(Ry0) * prior_unc_alpha ** 2
    return Ry

def linear_fit_mean(p_dist_out, wse, wse_r_u):
    """
    snippets taken from RiverObs
    """
    mask = np.where(np.isfinite(wse))
    ww = 1/wse_r_u**2
    SS = np.c_[p_dist_out, np.ones(len(p_dist_out), dtype=p_dist_out.dtype)]
    wse_fit = statsmodels.api.WLS(wse[mask], SS[mask], weights=ww[mask]).fit()
    prior_wse = wse_fit.predict(SS)
    return prior_wse

def RiverObs_reconstruct(p_dist_out, wse, wse_r_u):
    # mimics the Bayes estimation done in riverobs
    # make prior wse from linear fit
    mask = np.ones_like(wse) # TODO: actually find the mask of good data...
    ww = 1/wse_r_u**2
    SS = np.c_[ss, np.ones(len(p_dist_out), dtype=p_dist_out.dtype)]
    wse_fit = statsmodels.api.WLS(wse[mask], SS[mask], weights=ww[mask]).fit()
    prior_wse = wse_fit.predict(SS)
    # get the sampling operator
    # find where the data is not masked out or NaN
    ind = np.where(np.logical_not(np.logical_and(mask, np.isfinite(wse))))
    # get vector with 1 for valid data elements
    h = np.ones(np.shape(wse))
    h[ind] = 0
    # make full sampling matrix
    H = np.diag(h)
    # now remove the zero rows
    num_removed = 0
    for k, val in enumerate(h):
        if val == 0:
            row = k - num_removed
            H = np.delete(H, row, 0)
            num_removed = num_removed + 1
    # get the covariance matrices
    #Rv = self.get_noise_autocov(wse, wse_r_u, mask, full_noise_cov)
    Ry = exponential_cov(p_dist_out)
    # TODO: do the Bayes Estimate
    #stats

