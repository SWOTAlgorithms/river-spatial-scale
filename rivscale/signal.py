'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

This mdule is a place for signal processing, moddleing and estimation
including convariance/spectral estimation and data-driven modeling etc
'''

def estimate_signal_mean(var_in):
    """
    estimate the mean of the signal for each row in the stack

    var_in is a 2D array (e.g., stretch_stack-like)
    """
    signal_in = np.array(var_in.copy())
    # TODO: may need to account for missing samples in the mean estimate
    signal_mean = np.nanmean(signal_in, axis=0)
    signal_mean_2D = np.broadcast_to(signal_mean, np.shape(signal_in))
    return signal_mean, signal_mean_2D

def estimate_along_cov_from_stack_var(var_in):
    """
    estimate covariance

    var_in is a 2D array (e.g., stretch_stack-like)
    """
    signal_in = np.array(var_in.copy())
    M,N = np.shape(signal_in)
    # TODO: may need to account for missing samples in the mean estimate
    signal_mean, signal_mean_2D = estimate_signal_mean(signal_in)
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
    Norm = np.zeros((M,M))
    for k in range(N):
        this_sig = sig[:,k]
        this_h = h[:,k]
        #this_H = np.diag(h[n,:])
        # since we filled with zero, dont need to mutliply by H
        this_cov = np.outer(this_sig, this_sig)#this_sig.T@this_sig
        #this_norm = this_H.T@this_H
        #this_norm = np.diag(H[n,:]) # outer product of qusi-Identity is itself
        this_norm = np.outer(this_h, this_h)
        Cov = Cov + this_cov
        Norm = Norm + this_norm
    Cov = Cov / Norm
    return Cov, signal_mean_2D

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



