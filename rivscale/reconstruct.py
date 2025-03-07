'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

'''
import glob
import os.path

import argparse

import numpy as np
import pandas as pd

import errtools.plots
import matplotlib.pyplot as plt

import scipy.ndimage
from scipy import odr
from scipy.linalg import pinv, svd, eigh, norm

from errtools.misc import (split_utc_time, field_time_to_swot_time,
    swot_time_to_field_time, get_dist_to_outlet_from_node_id)

from errtools.plots import plot_cdf

import netCDF4 as nc

import statsmodels.api

import geopandas as gpd

import rivscale.data

######## Feb 2025

def reconstruct_filter_data(
        cfg,
        stretch_stack,
        wse_stats,
        width_stats,
        plot=False):
    """
    def process_bayes_reconstruction(
        cfg,
        stretch_stack,
        wse_stats,
        width_stats,
        height_width):
    
    TODO: enable overwriting of char_length_tau and prior_unc_alpha if commanded
    """
    # first handle optional config params
    if 'wse_dark_thresh' not in cfg.keys():
        cfg['wse_dark_thresh'] = '0.8'
    if 'wse_dark_thresh' not in cfg.keys():
        cfg['width_dark_thresh'] = '0.3'
    if 'width_smooth_size' not in cfg.keys():
        cfg['width_smooth_size'] = 'None'
    if 'wse_outlier_scale' not in cfg.keys():
        cfg['wse_outlier_scale'] = '5.0'
    if 'width_outlier_sclae' not in cfg.keys():
        cfg['width_outlier_scale'] = '5.0'
    """
    if 'wse_char_length_tau' not in cfg.keys():
        cfg['wse_char_length_tau'] = '100000.0'
    if 'wse_prior_unc_alpha' not in cfg.keys():
        cfg['wse_prior_unc_alpha'] = '1.5'
    if 'width_char_length_tau' not in cfg.keys():
        cfg['width_char_length_tau'] = '100000.0'
    if 'width_prior_unc_alpha' not in cfg.keys():
        cfg['width_prior_unc_alpha'] = '50.0'
    if 'rho_wse_width' not in cfg.keys():
        cfg['rho_wse_width'] = 50
    """
    if 'crop' not in cfg.keys():
        cfg['crop'] = 'False'
    #
    # populate witdh_u
    node_len = stretch_stack['area_total'] / stretch_stack['width']
    stretch_stack['width_u'] = stretch_stack['area_tot_u'] / node_len
    # make measurement uncert at least as much as signal uncert we assume
    stretch_stack['width_u'] = stretch_stack['width_u'] + 10 # + 500#2*prior_unc_alpha_width
    # filter out bad data
    #stretch_stack = rivscale.filter.filter_width_node_outliers(
    #    stretch_stack, width_stats, width_outlier_scale, plot=True)
    #plot = False
    #stretch_stack.filter_node_outliers(wse_stats, key='wse', plot=plot)
    #stretch_stack.filter_node_outliers(width_stats, key='width', plot=plot)
    # first remove outliers allowing typical spread of variability
    # over all time obs
    stretch_stack.filter_node_outliers(
        width_stats,
        key='width',
        use_ptiles=True,
        plot=plot)
    stretch_stack.filter_node_outliers(
        wse_stats,
        key='wse',
        plot=plot)
    # now remove outliers considering relative spread 
    stretch_stack.filter_node_outliers(
        width_stats,
        key='width',
        Delta2=True,
        plot=plot)
    stretch_stack.filter_node_outliers(
        wse_stats,
        Delta2=True,
        key='wse',
        plot=plot)

    #if plot:
    #    plt.show()
    # filter out high dark_frac nodes
    stretch_stack.filter_dark_water('width', cfg['width_dark_thresh'])
    stretch_stack.filter_dark_water('wse', cfg['wse_dark_thresh'])
    #stretch_stack.width[
    #    stretch_stack.dark_frac>width_dark_thresh] = np.nan
    #stretch_stack.wse[
    #    stretch_stack.dark_frac>wse_dark_thresh] = np.nan
    # drop rows with too  little data
    #reach_id = None
    reach_id = 'nope'
    if cfg['crop']:
        reach_id = stretch_stack.stretch_name
    stretch_stack = rivscale.filter.drop_stretch_nans(stretch_stack,
        reach_id=reach_id)
    return stretch_stack

def process_bayes_reconstruction(
        cfg,
        stretch_stack,
        wse_stats,
        width_stats,
        height_width):
    # first handle optional config params
    if 'wse_dark_thresh' not in cfg.keys():
        cfg['wse_dark_thresh'] = '0.8'
    if 'wse_dark_thresh' not in cfg.keys():
        cfg['width_dark_thresh'] = '0.3'
    if 'width_smooth_size' not in cfg.keys():
        cfg['width_smooth_size'] = 'None'
    if 'wse_outlier_scale' not in cfg.keys():
        cfg['wse_outlier_scale'] = '5.0'
    if 'width_outlier_sclae' not in cfg.keys():
        cfg['width_outlier_scale'] = '5.0'
    if 'wse_char_length_tau' not in cfg.keys():
        cfg['wse_char_length_tau'] = '100000.0'
    if 'wse_prior_unc_alpha' not in cfg.keys():
        cfg['wse_prior_unc_alpha'] = '1.5'
    if 'width_char_length_tau' not in cfg.keys():
        cfg['width_char_length_tau'] = '100000.0'
    if 'width_prior_unc_alpha' not in cfg.keys():
        cfg['width_prior_unc_alpha'] = '50.0'
    if 'rho_wse_width' not in cfg.keys():
        cfg['rho_wse_width'] = 50
    if 'crop' not in cfg.keys():
        cfg['crop'] = 'False'

    # filter the data
    stretch_stack = reconstruct_filter_data(
        cfg,
        stretch_stack,
        wse_stats,
        width_stats,
        plot=False)
    # now do the reconstruction
    joint_bayes = rivscale.products.BayesData.joint(
        stretch_stack,
        wse_stats,
        width_stats,
        height_width,
        rho_wse_width=cfg['rho_wse_width'])
    return joint_bayes


def generate_cov_matrix(
        stretch_stack,
        char_length_tau,
        prior_unc_alpha,
        time_key='time_id'):
    signal_cov = []
    for j,cycl in enumerate(stretch_stack[time_key]):
        R = exponential_cov(
            stretch_stack['dist_out'], # should probably use the actual node distances?
            char_length_tau=char_length_tau,
            prior_unc_alpha=prior_unc_alpha)
        signal_cov.append(R)
    return np.moveaxis(np.array(signal_cov), 0, -1)

####### Jan 2025, updated for joint height/width river stretch processing
def piecewise_linear(p, x):
    """
    """
    x0, y0, slope1, slope2 = p
    return np.piecewise(
            x,
            [x < x0],
            [lambda x: slope1 * x + y0 - slope1 * x0,
                lambda x: slope2 * x + y0 - slope2 * x0]
            )


def fit_model(x, y, xu, yu, beta=[0, 0, 0, 0]):
    # fits a curve of type model() assuming errror in both dimensions
    data = odr.RealData(x, y, sx=xu, sy=yu)
    #my_model = odr.Model(poly_model)
    my_model = odr.Model(piecewise_linear)
    fitter = odr.ODR(data, my_model, beta0=beta)
    out = fitter.run()
    p_est = out.beta
    p_err = out.sd_beta
    return p_est, p_err

def get_height_width_fit(
        stretch_data_in,
        beta=[0, 0, 50/1.5, 50/1.5],
        n_sig=3, # num of stds to consider outlier
        n_good=5):# num of good points needed to do a fit
    stretch_data= stretch_data_in.copy()
    # handle reach-level outliers
    bad_wse_mean = np.abs(stretch_data.stretch_wse_mean) > \
        n_sig * stretch_data.stretch_wse_std
    bad_width_mean = np.abs(stretch_data.stretch_width_mean) > \
        n_sig * stretch_data.stretch_width_std
    bad_wse_std = stretch_data.stretch_wse_std > \
        n_sig * np.median(stretch_data.stretch_wse_std)
    bad_width_std = stretch_data.stretch_width_std > \
        n_sig * np.median(stretch_data.stretch_width_std)
    good = np.logical_not(np.logical_or.reduce([
        bad_wse_mean, bad_width_mean, bad_wse_std, bad_width_std]))
    if np.sum(good)<n_good:
        stretch_data['hw_params'] = np.array(beta)
        stretch_data['hw_params_err'] = np.array([1,1,1,1]) * 1e10
        return stretch_data
    p_est, p_err = fit_model(
        stretch_data.stretch_wse_mean[good],#mn,
        stretch_data.stretch_width_mean[good],#mn_w,
        stretch_data.stretch_wse_std[good],#std_res,
        stretch_data.stretch_width_std[good],#std_w_res,
        beta=beta)
    stretch_data['hw_params'] = p_est
    stretch_data['hw_params_err'] = p_err
    return stretch_data

def reach_average(stretch_data_in, keys=['wse', 'width']):
    stretch_data = stretch_data_in.copy()
    for key in keys:
        data = stretch_data[key]
        ref = stretch_data['{}_reference'.format(key.split('_')[-1])]
        ref2 = np.broadcast_to(ref, np.shape(data.T)).T
        anom = data - ref2
        mn = np.nanmean(anom, axis=0)
        med = np.nanmedian(anom, axis=0)
        std = np.nanstd(anom, axis=0)
        mask = np.isfinite(anom)
        cnt = np.sum(mask, axis=0)
        # set output
        stretch_data['stretch_{}_mean'.format(key)] = mn
        stretch_data['stretch_{}_median'.format(key)] = med
        stretch_data['stretch_{}_std'.format(key)] = std
        stretch_data['stretch_{}_count'.format(key)] = cnt
    return stretch_data

def get_dw_dh_from_model(wse_anom, wse_u, height_width):
    # just look up the average dw_dh slope for a buffer around
    # the measured data
    wse_plus = wse_anom + wse_u
    wse_minus = wse_anom - wse_u
    if isinstance(wse_plus, np.ma.MaskedArray):
        wse_plus = wse_plus.filled(np.nan)
    if isinstance(wse_minus, np.ma.MaskedArray):
        wse_minus = wse_minus.filled(np.nan)
    #breakpoint()
    #width_hat = height_width.sample(wse, x_key='wse')
    width_hat_plus = height_width.sample(wse_plus, x_key='wse')
    width_hat_minus = height_width.sample(wse_minus, x_key='wse')
    dw_dh = (width_hat_plus - width_hat_minus) / (wse_plus - wse_minus)
    # fill in the mising data with average
    dw_dh[~np.isfinite(dw_dh)] = np.nanmean(dw_dh)
    return dw_dh

def get_dw_dh_from_model_defunkt(stretch_data, time_index, plot=False):
    x0, y0, dw_dh1, dw_dh2 = stretch_data.hw_params
    b1 = y0 - dw_dh1 * x0
    b2 = y0 - dw_dh2 * x0
    wse = stretch_data.stretch_wse_mean[time_index]
    width = stretch_data.stretch_width_mean[time_index]
    wse_std = stretch_data.stretch_wse_std[time_index]
    width_std = stretch_data.stretch_width_std[time_index]
    # define params of 3rd line related to noise std of wse and width
    # passing thorugh the point (width, wse)
    dw_dh3 = -width_std / wse_std
    b3 = width - dw_dh3 * wse
    # compute intersections of line 3 with line 1 and line 2
    # intersection of line 1 and 3
    x13 = (b3 - b1) / (dw_dh1 - dw_dh3)
    y13 = dw_dh1 * x13 + b1
    # intersection of line 2 and 3
    x23 = (b3 - b2) / (dw_dh2 - dw_dh3)
    y23 = dw_dh2 * x23 + b2
    # find which one is "closest" to (width, wse)
    #dist13 = np.sqrt((x13-wse)**2+(y13-width)**2)
    #dist23 = np.sqrt((x23-wse)**2+(y23-width)**2)
    # find the one that has the x on the line in the correct regime
    dw_dh = dw_dh1 # default the first regime
    if x23>x0:#dist23 < dist13:
        # The other regime
        dw_dh = dw_dh2
    if plot:
       # plot
       wses = stretch_data.stretch_wse_mean
       widths = stretch_data.stretch_width_mean
       x = np.linspace(np.nanmin(wses),np.nanmax(wses))
       y = rivscale.reconstruct.piecewise_linear(stretch_data.hw_params, x)
       y1 = dw_dh1 * x + b1#y0 - dw_dh1 * x0
       y2 = dw_dh2 * x + b2#y0 - dw_dh2 * x0 
       #dw_dh3 = -width_std / wse_std
       #b3 = width - dw_dh3 * wse
       y3 = dw_dh3 * x + b3 
       # compute intersections
       # plot
       plt.figure()
       plt.scatter(wses, widths)
       plt.errorbar(wse, width, xerr=wse_std, yerr=width_std, c='k')
       plt.plot(x, y,c='r')
       #plt.plot(x, y1)
       #plt.plot(x, y2)
       plt.plot(x, y3, c='g')
       plt.plot(x13, y13, 'x')
       plt.plot(x23, y23, 'x')
       if x23>x0:#dist23 < dist13:
           plt.plot(x23, y23, '^')
       else:
           plt.plot(x13, y13, '^')
       #plt.show()
       #breakpoint()
    return dw_dh

def joint_reconstruct_stretch(stretch_data_in):
    # copy the input data to output data
    stretch_data = stretch_data_in.copy()
    # get the mean and cov of the stacked wse and width
    N = len(stretch_data['wse_reference'])
    mn = np.concatenate([
        stretch_data['wse_reference'], stretch_data['width_reference']
        ])    
    nodes = np.arange(2*len(stretch_data['node_id']), dtype=int)
    wse_hats = []
    wse_hats_u = []
    width_hats = []
    width_hats_u = []
    wse_post_covs = []
    width_post_covs = []
    wse_width_post_covs = []
    time_key = 'time_id'
    # go through each time/cycle observation in the stack 
    for j,cycl in enumerate(stretch_data[time_key]):
        meas = np.concatenate([
            stretch_data['wse'][:,j], stretch_data['width'][:,j]
            ])
        meas_u = np.concatenate([
            stretch_data['wse_u'][:,j], stretch_data['width_u'][:,j]
            ])
        Ry = np.block([
            [stretch_data['wse_cov'][:,:,j], stretch_data['wse_width_cov'][:,:,j].T],
            [stretch_data['wse_width_cov'][:,:,j], stretch_data['width_cov'][:,:,j]]
            ])
        #print(np.linalg.cond(Ry))
        #if np.linalg.cond(Ry) > 1e
        # call the estimator
        signal_hat, post_cov = reconstruct_one_time_obs(
            meas, meas_u, Ry, mn, nodes)
        # populate the output arrays
        wse_hats.append(signal_hat[0:N])
        wse_post_covs.append(post_cov[0:N,0:N])
        wse_hats_u.append(np.diag(post_cov[0:N,0:N]))
        width_hats.append(signal_hat[N:])
        width_post_covs.append(post_cov[N:,N:])
        width_hats_u.append(np.diag(post_cov[N:,N:]))
        wse_width_post_covs.append(post_cov[0:N,N:])
    stretch_data['bayes_wse'] = np.array(wse_hats).T
    stretch_data['bayes_width'] = np.array(width_hats).T
    stretch_data['bayes_wse_u'] = np.array(wse_hats_u).T
    stretch_data['bayes_width_u'] = np.array(width_hats_u).T
    #breakpoint()
    stretch_data['bayes_wse_post_cov'] = np.moveaxis(
        np.array(wse_post_covs), 0, -1)
    stretch_data['bayes_width_post_cov'] = np.moveaxis(
        np.array(width_post_covs), 0, -1)
    stretch_data['bayes_wse_width_post_cov'] = np.moveaxis(
        np.array(wse_width_post_covs), 0, -1)
    return stretch_data

def joint_reconstruct_stretch_defunkt(stretch_data_in):
    # copy the input data to output data
    stretch_data = stretch_data_in.copy()
    # get the mean and cov of the stacked wse and width
    N = len(stretch_data['wse_reference'])
    mn = np.concatenate([
        stretch_data['wse_reference'], stretch_data['width_reference']
        ])
    Ry = np.block([
        [stretch_data['wse_cov'], stretch_data['wse_width_cov'].T],
        [stretch_data['wse_width_cov'], stretch_data['width_cov']]
        ])
    # make a fake stretch with the stacked data stuffed into the wse terms
    # and then call the standard reconsruct routine
    this_stretch_data = rivscale.products.RiverStretchData()
    this_stretch_data['wse_reference'] = mn
    this_stretch_data['wse_cov'] = Ry
    this_stretch_data['wse_u'] = np.concatenate([
        stretch_data['wse_u'], stretch_data['width_u']
        ])
    this_stretch_data['node_id'] = np.concatenate([
        stretch_data['node_id'], stretch_data['node_id']
        ])
    this_stretch_data['wse'] = np.concatenate([
        stretch_data['wse'], stretch_data['width']
        ])
    this_stretch_data['time_id'] = stretch_data['time_id']

    # call the reconstruct routine
    this_stretch_data = rivscale.reconstruct.reconstruct_stretch(
        this_stretch_data, signal_key='wse', uncert_key='wse_u')
    # unpack arrays into the outputs
    stretch_data['bayes_wse'] = this_stretch_data['bayes_wse'][0:N]
    stretch_data['bayes_width'] = this_stretch_data['bayes_wse'][N:]
    stretch_data['bayes_wse_post_cov'] = this_stretch_data['bayes_wse_post_cov'][0:N,0:N]
    stretch_data['bayes_width_post_cov'] = this_stretch_data['bayes_wse_post_cov'][N:,N:]
    stretch_data['bayes_wse_width_post_cov'] = this_stretch_data['bayes_wse_post_cov'][0:N,N:]
    stretch_data['bayes_wse_u'] = this_stretch_data['bayes_wse_u'][0:N]
    stretch_data['bayes_width_u'] = this_stretch_data['bayes_wse_u'][N:]
    return stretch_data

####### Sept 2024, updated for river stretch processing
def reconstruct_one_time_obs(meas, meas_u_in, Ry, mn, nodes):
    meas_u = meas_u_in.copy()
    meas_u[~np.isfinite(meas_u)] = 10**5#10^5
    msk = np.isfinite(meas)
        #breakpoint()
    # check for not enough data
    if np.sum(msk) < 2:
        # dont call bayes, just append nans
        nan_array = np.ones_like(meas) + np.nan
        unc_array = np.zeros_like(meas_u) + 10**5
        signal_hat = nan_array
        post_cov = np.diag(unc_array)
        bayes_u = unc_array.copy()
    else:
        signal_hat, A_inv = bayes_estimator(
            meas[msk], nodes[msk], mn, Ry,
            np.sqrt(meas_u[msk]))
        post_cov = A_inv
        #bayes_u = np.diag(A_inv).copy()
    return signal_hat, post_cov#, bayes_u

#
def MAP_gradient_one_time_obs(
        meas, meas_u_in, Ry, mn, nodes, signal_hat):
    meas_u = meas_u_in.copy()
    meas_u[~np.isfinite(meas_u)] = 10**5#10^5
    msk = np.isfinite(meas)
        #breakpoint()
    # check for not enough data
    if np.sum(msk) < 2:
        # dont call bayes, just append nans
        grad = np.zeros_like(meas)
        post_cov = np.diag(np.ones_like(meas_u)*10**5)
    else:
        grad, post_cov = MAP_gradient(
            meas[msk], nodes[msk], mn, Ry,
            np.sqrt(meas_u[msk]), signal_hat)
    return grad, post_cov

def reconstruct_stretch(stretch_data, bayes_data_in,
        signal_key='wse', uncert_key='wse_u'):
    # copy the input data to output data
    bayes_data = bayes_data_in.copy()
    # get the bayes parameters
    mn = bayes_data.signal_mean
    #Ry = bayes_data['signal_cov']
    # TODO: should this be local_node_id (i.e., indexing starting at 1)?
    nodes = np.arange(len(stretch_data['node_id']), dtype=int)
    signal_hats = []
    post_covs = []
    bayes_us = []
    time_key = 'time_id'
    # go through each time/cycle observation in the stack 
    for j,cycl in enumerate(stretch_data[time_key]):
        Ry = bayes_data.signal_cov[:,:,j]
        meas = stretch_data[signal_key][:,j].copy()
        meas_u = stretch_data[uncert_key][:,j].copy()
        signal_hat, post_cov = reconstruct_one_time_obs(
            meas, meas_u, Ry, mn, nodes)
        signal_hats.append(signal_hat)
        post_covs.append(post_cov)
        bayes_us.append(np.diag(post_cov).copy())
    bayes_data.signal = np.array(signal_hats).T
    bayes_data.signal_post_cov = np.moveaxis(
        np.array(post_covs), 0, -1)
    bayes_data.signal_u = np.array(bayes_us).T
    return bayes_data

def reconstruct_stretch_defunkt(stretch_data_in,
        signal_key='wse', uncert_key='wse_u', constrain_hw=True):
    # copy the input data to output data
    stretch_data = stretch_data_in.copy()
    # get the bayes parameters
    mn = stretch_data['{}_reference'.format(signal_key)]#.filled(np.nan)
    #Ry = stretch_data['{}_cov'.format(signal_key)]#.filled(0)
    # TODO: should this be local_node_id (i.e., indexing starting at 1)?
    nodes = np.arange(len(stretch_data['node_id']), dtype=int)
    signal_hats = []
    post_covs = []
    bayes_us = []
    time_key = 'time_id'
    # go through each time/cycle observation in the stack 
    for j,cycl in enumerate(stretch_data[time_key]):
        Ry = stretch_data['{}_cov'.format(signal_key)][:,:,j]
        meas = stretch_data[signal_key][:,j]
        meas_u = stretch_data[uncert_key][:,j]
        signal_hat, post_cov = reconstruct_one_time_obs(
            meas, meas_u, Ry, mn, nodes)
        signal_hats.append(signal_hat)
        post_covs.append(post_cov)
        bayes_us.append(np.diag(post_cov).copy())
    stretch_data['bayes_{}'.format(signal_key)] = np.array(signal_hats).T
    stretch_data['bayes_{}_post_cov'.format(signal_key)] = np.moveaxis(
        np.array(post_covs), 0, -1)
    stretch_data['bayes_{}_u'.format(signal_key)] = np.array(bayes_us).T
    return stretch_data

def noise_and_prior_terms(
        node_index, mn, Ry0, meas_noise):
    N = len(mn)
    H = np.zeros((len(node_index), N))
    for i,ni in enumerate(node_index):
        if ni == -30:
            breakpoint()
        H[i,ni] = 1
    #X = meas_wse - mn[node_index]
    Rv = np.diag(meas_noise**2)
    Ry_inv = pinv(Ry0)
    Rv_inv = pinv(Rv)
    A = Ry_inv + H.T @ Rv_inv @ H
    A_inv = np.linalg.inv(A)
    return H, Rv, Rv_inv, Ry_inv, A, A_inv

def bayes_estimator(meas, node_index, mn, Ry0, meas_noise):
    """
    N = len(mn)
    H = np.zeros((len(node_index), N))
    for i,ni in enumerate(node_index):
        if ni == -30:
            breakpoint()
        H[i,ni] = 1
    #X = meas_wse - mn[node_index]
    Rv = np.diag(meas_noise**2)
    Ry_inv = pinv(Ry0)
    Rv_inv = pinv(Rv)
    A = Ry_inv + H.T @ Rv_inv @ H
    A_inv = np.linalg.inv(A)
    """
    H, Rv, Rv_inv, Ry_inv, A, A_inv = noise_and_prior_terms(
        node_index, mn, Ry0, meas_noise)
    K = A_inv @ H.T @ Rv_inv
    K_bar = A_inv @ Ry_inv
    yp = K @ meas
    y_bar =  K_bar @ mn
    signal_hat = yp + y_bar
    return signal_hat, A_inv

def MAP_gradient(meas, node_index, mn, Ry0, meas_noise, meas_hat):
    H, Rv, Rv_inv, Ry_inv, A, A_inv = noise_and_prior_terms(
        node_index, mn, Ry0, meas_noise)
    grad = A @ meas_hat - Ry_inv @ mn - H.T @ Rv_inv @ meas
    return grad, A_inv

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
    if isinstance(prior_unc_alpha, np.ndarray):
        #breakpoint()
        Ry = np.outer(prior_unc_alpha,prior_unc_alpha) * (Ry0 / np.max(Ry0))
    else:
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
    #A_inv = np.linalg.inv(A)
    A_inv = pinv(A)
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

