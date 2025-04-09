'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

'''
import numpy as np
from scipy import odr
from scipy.linalg import pinv
import rivscale.filter
import rivscale.products.bayes_data

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
    stretch_stack, reach_id = rivscale.filter.filter_stretch_stack(
        cfg,
        stretch_stack,
        wse_stats,
        width_stats,
        plot=False)
    # now do the reconstruction
    joint_bayes = rivscale.products.bayes_data.BayesData.joint(
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

###
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



