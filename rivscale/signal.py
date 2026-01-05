'''
Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

This module is a place for signal processing, modeling and estimation
including convariance/spectral estimation and data-driven modeling etc
'''
import numpy as np


def estimate_signal_mean(var_in, ref=None):
    """
    estimate the mean of the signal for each row in the stack

    var_in is a 2D array (e.g., stretch_stack-like)
    """
    signal_in = np.array(var_in.copy())
    # TODO: may need to account for missing samples in the mean estimate
    if ref is not None:
        ref_2D = np.broadcast_to(ref, np.shape(signal_in.T)).T
        ref_mean = np.nanmean(ref_2D, axis=0)
        signal_mean = np.nanmean(signal_in - ref_2D, axis=0) + ref_mean
    else:
        signal_mean = np.nanmean(signal_in, axis=0)
    signal_mean_2D = np.broadcast_to(signal_mean, np.shape(signal_in))
    return signal_mean, signal_mean_2D

def estimate_along_cov_from_stack_var(var_in, noise_in, ref=None, min_noise = 0.1):
    """
    estimate covariance

    var_in is a 2D array (e.g., stretch_stack-like)
    """
    signal_in = np.array(var_in.copy())
    noise_var = np.array(noise_in.copy())
    noise_var[noise_var<=min_noise] = 1e4
    noise_var[~np.isfinite(noise_var)] = 1e4
    M,N = np.shape(signal_in)
    # TODO: may need to account for missing samples in the mean estimate
    #signal_mean, signal_mean_2D = estimate_signal_mean(signal_in, ref=ref)
    if ref is None:
        sig = signal_in
        signal_mean_2D = np.zeros_like(sig)
    else:
        signal_mean_2D = np.broadcast_to(ref, np.shape(signal_in.T)).T
        # subtract the mean
        sig = signal_in - signal_mean_2D
    # get the mask of missing values to create the sampling operator
    h_msk = ~np.isfinite(sig)
    h = np.ones_like(sig)
    h[h_msk] = 0
    # Zero out the invalid entries
    #sig = signal.copy()
    sig[h_msk] = 0
    # go through each signal in stack and accumulate rank-1 Cov estimates
    Cov = np.zeros((M,M))
    Count = np.zeros((M,M))
    Norm = np.zeros((M,M))
    for k in range(N):
        this_sig = sig[:,k]
        this_h = h[:,k]
        this_wgt = this_h / noise_var[:,k]
        #this_H = np.diag(h[n,:])
        # since we filled with zero, dont need to mutliply by H
        this_cov = np.outer(this_sig*this_wgt, this_sig*this_wgt)#this_sig.T@this_sig
        #this_norm = this_H.T@this_H
        #this_norm = np.diag(H[n,:]) # outer product of qusi-Identity is itself
        this_count = np.outer(this_h, this_h)
        this_weight = np.outer(this_wgt, this_wgt)
        Cov = Cov + this_cov
        Norm = Norm + this_weight
        Count = Count + this_count
        #breakpoint()
    Cov = Cov / Norm
    return Cov, Count, Norm, signal_mean_2D

def KL_model(Cov, var_in):
    """
    Perform a Karhunen-Loeve Model
    using low-order basis functions generated from Cov

    var_in is a 2D array (e.g., stretch_stack-like)
    """

    signal_in = np.array(var_in.copy())
    signal_mean, signal_mean_2D = estimate_signal_mean(signal_in)
    sig = signal_in - signal_mean_2D
    h_msk = ~np.isfinite(sig)
    # Zero out the invalid entries
    sig[h_msk] = 0
    # now take the generalized spectrum (SVD)
    U,S,V = np.linalg.svd(Cov)
    # do a low-order fit to the first n basis vectors
    # TODO: pick number of basis vectors by the signal
    # noise level projected into the U basis 
    Order = len(S[S>10])
    if Order > 10:# TODO: make these parameters
        Order = 10
    if Order < 2:
        Order = 2
    B = np.array(U[:,0:Order].copy())
    #breakpoint()
    B_inv = np.linalg.pinv(B)
    C = np.array(B_inv @ sig)
    sig_hat = B @ C + signal_mean_2D
    return sig_hat, U, S, C, Order


def weighted_sampled_fit(B_in, var_in, H=None):
    """
    var_in is a 1D or 2D array of data (e.g., along-river wse)
    B = a linear Basis matrix/vector
    H is a sampling/weigthing vector for each row of var_in
    """
    signal_in = np.array(var_in).copy()
    B = np.array(B_in).copy()
    # check shape of B
    if np.shape(B)[0] != np.shape(signal_in)[0]:
        B = B.T
    if H is None:
        # use the Nan Mask as the sampling operator
        H = np.zeros_like(signal_in)
        H[np.isfinite(signal_in)] = 1
        signal_in[H==0] = 0
    M,N = np.shape(signal_in)
    fit = []
    params = []
    for k in range(N):
        # do a weighted fit for each row
        #breakpoint()
        sig = signal_in[:,k]
        W = np.diag(H[:,k])
        C = np.linalg.pinv(B.T @ W @ B) @ B.T @ W @ sig
        fit.append(B @ C)
        params.append(C)
    return np.array(fit).T, np.array(params).T



