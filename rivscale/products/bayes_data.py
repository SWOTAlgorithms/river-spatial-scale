'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author (s): Brent Williams

Container for Bayes reconstruction of height, width, or joint (or other
variables).  

Mimics the product class from RiverObs and the SWOT project "python" repo
'''
from collections import OrderedDict as odict
from rivscale.misc import textjoin
from rivscale.products.constants import (
        DIMENSIONS_ALL, DIMENSIONS_SAVG, DIMENSIONS_ALONG, DIMENSIONS_2D,
        DIMENSIONS_PCNT, DIMENSIONS_PCNT2, DIMENSIONS_COV, DIMENSIONS_POSTCOV)

from SWOTWater.products.product import Product, ProductTesterMixIn
from SWOTWater.products.constants import FILL_VALUES

import rivscale.estimate
import rivscale.plot
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
import os.path
import numpy as np
import rivscale.products.along_stretch
import rivscale.products.stretch_stack

class BayesData(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding Bayes reconstruction parameters and the
            reconstructed signals for each time observation in the
            multitemproal stack of data.
            """)}],
        ['stretch_name',{'dtype':'str', 'value': textjoin("""
            Name given to this stretch instance (e.g., center reach
            or river name)
            """)}],
        ['signal_key',{'dtype':'str', 'value':'wse, width, or dark_frac'}],
        ])
    DIMENSIONS = DIMENSIONS_ALL
    VARIABLES = odict([
        ['reaches', odict([['dimensions', odict([['num_reaches', 0]])]])],
        ['dist_out', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['node_length', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['along_dist', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['cross_track', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['local_node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['p_lat', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['p_lon', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['time_id', odict([['dimensions', odict([['num_times', 0]])]])],
        ['granule_id', odict([['dimensions', odict([['num_times', 0]])]])],
        ['signal', odict([['dimensions', DIMENSIONS_2D]])],
        ['signal_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['signal_post_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        #['bayes_wse_width_post_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['signal_mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        #['signal_cov', odict([['dimensions', odict([['num_nodes', 0]])]])],
        #['wse_cov', odict([['dimensions', DIMENSIONS_COV]])],
        #['width_cov', odict([['dimensions', DIMENSIONS_COV]])],
        #['wse_width_cov', odict([['dimensions', DIMENSIONS_COV]])],
        ['signal_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        #['width_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        #['wse_width_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
    ])

    def plot(
            self,
            x_key='along_dist',#'dist_out', # or 'time_id'
            outdir=None,
            show=False,
            title_tag=''):
        title_tag = title_tag+'Bayes'
        # create reference object
        wse_reference = rivscale.products.along_stretch.AlongStretchStats()
        width_reference = rivscale.products.along_stretch.AlongStretchStats()
        # cast to a stretch_stack object and use its plotter
        stretch_stack = rivscale.products.stretch_stack.StretchStack()
        stretch_stack.stretch_name = self.stretch_name
        if 'joint' not in self.signal_key:
            stretch_stack.dist_out = self.dist_out
            stretch_stack.along_dist = self.along_dist
            stretch_stack.p_lon = self.p_lat
            stretch_stack.p_lat = self.p_lon
        stretch_stack.time_id = self.time_id
        if self.signal_key == 'wse':
            title_tag = title_tag + ' wse'
            stretch_stack.wse = self.signal
            stretch_stack.wse_u = self.signal_u
            wse_reference.reference = self.signal_mean
        if self.signal_key == 'width':
            title_tag = title_tag + ' width'
            stretch_stack.width = self.signal
            stretch_stack.width_u = self.signal_u
            width_reference.reference = self.signal_mean
        if self.signal_key == 'joint_wse_width':
            title_tag = title_tag + ' joint wse width'
            wse_b, width_b, post_cov_b = self.unpack_joint()
            N = len(wse_b.signal_mean)
            stretch_stack.dist_out = self.dist_out[0:N]
            stretch_stack.along_dist = self.along_dist[0:N]
            stretch_stack.p_lon = self.p_lat[0:N]
            stretch_stack.p_lat = self.p_lon[0:N]
            stretch_stack.wse = wse_b.signal
            stretch_stack.wse_u = wse_b.signal_u
            stretch_stack.width = width_b.signal
            stretch_stack.width_u = width_b.signal_u
            # unpack reference
            wse_reference.reference = wse_b.signal_mean
            width_reference.reference = width_b.signal_mean
        #breakpoint()
        stretch_stack.plot(
            wse_reference,
            width_reference,
            title_tag=title_tag,
            outdir=outdir,
            show=show)


    @classmethod
    def simple(
            cls,
            stretch_stack,
            stats,
            signal_key,
            char_length_tau=None,
            prior_unc_alpha=None):
        #    
        bayes = cls()
        bayes.stretch_name = stretch_stack.stretch_name
        bayes.reaches = stretch_stack.reaches
        bayes.dist_out = stretch_stack.dist_out
        bayes.dist_out = stretch_stack.along_dist
        bayes.dist_out = stretch_stack.node_length
        bayes.time_id = stretch_stack.time_id
        bayes.granule_id = stretch_stack.granule_id
        bayes.signal_key = signal_key
        bayes.signal_mean = stats.reference.copy()
        time_key = 'time_id'
        # do some validity checks
        num_valid = len(stretch_stack[signal_key][
            np.isfinite(stretch_stack[signal_key])])
        if num_valid < 2:
            return bayes
        # create the cov
        if (char_length_tau is not None) and (prior_unc_alpha is not None):
            bayes.signal_cov = rivscale.reconstruct.generate_cov_matrix(
                stretch_stack,
                char_length_tau,
                prior_unc_alpha)
        elif isinstance(stats, rivscale.products.along_stretch.AlongStretchStats):
            # use what is in the stats
            #breakpoint()
            try:
                bayes.signal_cov = rivscale.reconstruct.generate_cov_matrix(
                    stretch_stack,
                    stats.char_length_tau,
                    stats.prior_unc_alpha)
            except AssertionError as e:
                print(e)
            # TODO: use the cov in there...
        # actually run it
        bayes = rivscale.reconstruct.reconstruct_stretch(
            stretch_stack, bayes,
            signal_key=signal_key,
            uncert_key=signal_key+'_u')
        return bayes

    @classmethod
    def joint(
            cls,
            stretch_stack,
            wse_along_stats,
            width_along_stats,
            height_width,
            rho_wse_width=0.7,
            maxiter=5
            ):
        bayes = cls()
        bayes.signal_key = 'joint_wse_width'
        bayes.stretch_name = stretch_stack.stretch_name
        N = len(wse_along_stats.reference)
        bayes.reaches = stretch_stack.reaches
        bayes.dist_out = np.concatenate([
            stretch_stack.dist_out,
            stretch_stack.dist_out])
        bayes.along_dist = np.concatenate([
            stretch_stack.along_dist,
            stretch_stack.along_dist])
        bayes.node_length = np.concatenate([
            stretch_stack.node_length,
            stretch_stack.node_length])
        bayes.time_id = stretch_stack.time_id
        bayes.granule_id = stretch_stack.granule_id
        # get the mean and cov of the stacked wse and width
        #N = len(wse_along_stats.reference)
        # create the stacked mean
        mn = np.concatenate([
            wse_along_stats.reference, width_along_stats.reference
            ])
        wse_cov = rivscale.reconstruct.exponential_cov(
            stretch_stack['along_dist'],#stretch_stack['dist_out'],
            char_length_tau=wse_along_stats.char_length_tau,
            prior_unc_alpha=wse_along_stats.prior_unc_alpha)
        width_cov = rivscale.reconstruct.exponential_cov(
            stretch_stack['along_dist'],#stretch_stack['dist_out'],
            char_length_tau=width_along_stats.char_length_tau,
            prior_unc_alpha=width_along_stats.prior_unc_alpha)
        bayes.signal_mean = mn
        nodes = np.arange(2*len(stretch_stack['node_id']), dtype=int)
        Signal_hat = []
        Signal_hat_u = []
        Post_cov = []
        time_key = 'time_id'
        #
        # go through each time/cycle observation in the stack
        for j,cycl in enumerate(stretch_stack[time_key]):
            meas = np.concatenate([
                stretch_stack['wse'][:,j], stretch_stack['width'][:,j]
                ])
            meas_u = np.concatenate([
                stretch_stack['wse_u'][:,j], stretch_stack['width_u'][:,j]
                ])
            """
            wse_cov = rivscale.reconstruct.exponential_cov(
                stretch_stack['dist_out'],
                char_length_tau=wse_along_stats.char_length_tau,
                prior_unc_alpha=wse_along_stats.prior_unc_alpha)
            width_cov = rivscale.reconstruct.exponential_cov(
                stretch_stack['dist_out'],
                char_length_tau=width_along_stats.char_length_tau,
                prior_unc_alpha=width_along_stats.prior_unc_alpha)
            """
            # constrain the height and width std magnitudes using the h/w-model
            # init with prior mean
            signal_hat = np.concatenate([
                    wse_along_stats.reference,
                    width_along_stats.reference])
            # compute the hw-independent bayes estimate
            R1 = wse_cov.copy()
            R2 = width_cov.copy()
            R12 = np.zeros_like(R1)
            Ry = np.block([
                [R1, R12],
                [R12.T, R2],
                ])
            signal_b, post_cov = rivscale.reconstruct.reconstruct_one_time_obs(
                meas, meas_u, Ry, mn, nodes)

            #grad, post_cov = rivscale.reconstruct.MAP_gradient_one_time_obs(
            #    meas, meas_u, Ry, mn, nodes, signal_hat)
            #
            # init the nonlinear search with the bayes estimate
            # that does not impose the hw relation?
            signal_hat = signal_b.copy()
            '''
            # replace estimate with measurement when available
            #signal_hat[np.isfinite(meas)] = meas[np.isfinite(meas)]
            #this_wse_anom = np.array(
            #    stretch_stack['wse'][:,j] - wse_along_stats.reference).copy()
            ## init the wse_anom to the measurement or prior
            ## where there is no measurement
            #this_wse_anom[~np.isfinite(this_wse_anom)] = 0
            this_wse_u = np.array(stretch_stack['wse_u'][:,j]).copy()
            for k in range(maxiter+1):
                # loop moving toward the bayes estimate while updating the
                # dw_dh along the way to handle nonlinear height/width models
                #dw_dh = rivscale.reconstruct.get_dw_dh_from_model(
                #    stretch_stack, height_width, j)
                this_wse_anom = (
                    signal_hat[0:N] - wse_along_stats.reference).copy()
                this_width_anom = (
                    signal_hat[N] - width_along_stats.reference).copy()
                #dw_dh = rivscale.reconstruct.get_dw_dh_from_model(
                #    this_wse_anom, this_wse_u, height_width)
                dw_dh = height_width.sample_deriv(this_wse_anom, x_key='wse')
                dh_dw = height_width.sample_deriv(this_width_anom, x_key='width')
                # experiment with constant dw_dh and dh_dw
                dw_dh[:] = np.median(dw_dh)
                #dh_dw[:] = np.median(dh_dw)
                dw_dh = 1/dh_dw
                #
                f_of_h = height_width.sample(this_wse_anom, x_key='wse')
                g_of_w = height_width.sample(this_width_anom, x_key='width')
                K_of_y = np.concatenate([g_of_w, f_of_h])
                signal_anom = np.concatenate([this_wse_anom, this_width_anom])
                hw_model_diff = np.concatenate([
                    this_wse_anom - g_of_w,
                    this_width_anom - f_of_h])
                # TODO: for some reason allowing each node to vary in dw_dh
                #       doesnt seem to work...need to investigate
                #dw_dh = np.median(dw_dh)
                this_prior_unc_alpha_width = dw_dh * wse_along_stats.prior_unc_alpha
                #breakpoint()
                """
                width_cov = rivscale.reconstruct.exponential_cov(
                    stretch_stack['dist_out'],
                    char_length_tau=width_along_stats.char_length_tau,
                    prior_unc_alpha=this_prior_unc_alpha_width)

                wse_width_cov = (
                    rho_wse_width * np.real(scipy.linalg.sqrtm(wse_cov) @ (
                        scipy.linalg.sqrtm(width_cov.T))))

                Ry = np.block([
                    [wse_cov, wse_width_cov.T],
                    [wse_width_cov, width_cov],
                    ])
                """
                """
                #width_cov = wse_cov.copy()
                #wse_width_cov = wse_cov.copy()
                Rg = wse_cov.copy() #/ np.max(wse_cov)
                Rf = width_cov.copy() #/ np.max(width_cov)
                #Rf = wse_cov.copy()
                #Rg = width_cov.copy()
                A = np.diag(dw_dh)
                C = np.diag(1/dw_dh)
                a = rho_wse_width
                b = rho_wse_width
                R1 = (1-a)**2 * b**2 * A @ Rg @ A.T + (1-b)**2 * Rf
                R2 = (1-b)**2 * a**2 * C @ Rf @ C.T + (1-a)**2 * Rg
                R12 = (1-a)**2 * b * A @ Rg + (1-b)**2 * a * Rf @ C.T
                Ry = 1/(1-a*b)**2 * np.block([
                    [R1, R12],
                    [R12.T, R2],
                    ])
                """
                """
                A = np.diag(dw_dh)
                #sig = rho_wse_width
                sig = 5000
                R1 = wse_cov.copy()
                R2 = A @ R1 @ A.T + sig * np.eye(N)
                R12 = A @ R1
                Ry = np.block([
                    [R1, R12],
                    [R12.T, R2],
                    ])
                """
                """
                R1 = wse_cov.copy()
                R2 = width_cov.copy()
                R12 = np.zeros_like(R1)
                Ry = np.block([
                    [R1, R12],
                    [R12.T, R2],
                    ])
                #breakpoint()
                # call the estimator
                #signal_hat, post_cov = rivscale.reconstruct.reconstruct_one_time_obs(
                #    meas, meas_u, Ry, mn, nodes)
                signal_b, post_cov = rivscale.reconstruct.reconstruct_one_time_obs(
                    meas, meas_u, Ry, mn, nodes)
                signal_b2, post_cov2 = rivscale.reconstruct.reconstruct_one_time_obs(
                    meas - mn, meas_u, Ry, np.zeros_like(mn), nodes)
                """
                grad, post_cov = rivscale.reconstruct.MAP_gradient_one_time_obs(
                        meas, meas_u, Ry, mn, nodes, signal_hat)
                #grad_norm = grad / np.linalg.norm(grad)
                wgt = 1
                Lamda  = wgt * np.block([
                    [np.eye(N) * (1/5), np.zeros((N,N))],
                    [np.zeros((N,N)), np.eye(N) * (1/500)],
                    ])
                Ry_inv = np.linalg.inv(Ry)
                #Lamda  = Ry_inv
                D = np.block([
                    [np.zeros((N,N)), np.diag(dh_dw)],
                    [np.diag(dw_dh), np.zeros((N,N))],
                    ])
                G = np.eye(2*N) - D
                #G = np.block([
                #    [np.eye(N), -np.diag(dw_dh)],
                #    [-np.diag(dh_dw), np.eye(N)],
                #    ])
                grad_hw = G.T @ Lamda @ hw_model_diff
                #grad_hw = G @ Lamda @ hw_model_diff
                grad_hw_norm = post_cov @ grad_hw
                grad_norm = post_cov @ grad
                grad_tot = grad_norm + grad_hw_norm
                #

                A_b = np.linalg.inv(post_cov)
                A_inv = np.linalg.inv(A_b + G.T @ Lamda @ G)
                signal_hat2 = A_inv @ (A_b @ signal_b - \
                    G.T @ Lamda @ (
                        -G @ signal_hat - K_of_y  + signal_hat- mn))
                """
                ### linear, fully constraind in deltah, deltaw
                good_inds = np.where(np.isfinite(meas))
                H0 = np.eye(2*N)
                H = H0[good_inds,:].squeeze()
                Rv_inv  = np.diag(1/meas_u[good_inds]**2)
                Ry_inv = np.linalg.inv(Ry)
                #Rh_inv = np.linalg.inv(wse_cov)
                #Rw_inv = np.linalg.inv(width_cov)
                x = (meas - mn)[good_inds].squeeze()
                y0 = signal_hat - mn
                b0 = K_of_y
                A = H.T @ Rv_inv @ H + Ry_inv + G.T @ Lamda @ G
                A_inv = np.linalg.inv(A)
                y_hat = A_inv @ (H.T @ Rv_inv @ x - G.T @ Lamda @ (D @ y0 - b0))
                signal_hat2 = y_hat + mn
                ###
                """
                """
                signal_hat2 = signal_b - post_cov @ G.T @ Lamda @ (
                    signal_anom - K_of_y)
                """
                """
                ###
                # do fully contrained linear Bayes/MAP
                ###
                good_inds = np.where(np.isfinite(meas))
                D = np.diag(dh_dw)
                H0 = np.block([
                    [np.eye(N)],
                    [D],
                    ])
                Ht = H0[good_inds,:].squeeze()
                Rv_inv  = np.diag(1/meas_u[good_inds]**2)
                Rh_inv = np.linalg.inv(wse_cov)
                Rw_inv = np.linalg.inv(width_cov)
                A = Ht.T @ Rv_inv @ Ht + Rh_inv + D.T @ Rw_inv @ D
                A_inv = np.linalg.inv(A)
                b = f_of_h - D @ signal_hat[0:N]
                x0 = np.concatenate([
                    meas[0:N],
                    meas[N:] - mn[N:] - b])
                xt = x0[good_inds].squeeze()
                wse_hat = A_inv @ (
                    Ht.T @ Rv_inv @ xt + Rh_inv @ mn[0:N] - D.T @ Rw_inv @ b)
                #width_hat = D @ wse_hat + b + mn[N:]
                width_hat = height_width.sample(
                    wse_hat - mn[0:N], x_key='wse') + mn[N:]
                signal_hat2 = np.concatenate([wse_hat, width_hat])
                """
                ###
                ###
                #breakpoint()
                mu = 0.9
                #signal_hat1 = signal_hat - mu * grad_tot
                signal_hat1 = signal_hat
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(signal_b[0:N] - wse_along_stats.reference,
                        label='bayes')
                plt.plot(signal_hat[0:N] - wse_along_stats.reference,
                        label='current est')
                plt.plot(signal_hat1[0:N] - wse_along_stats.reference,
                        label='next est')
                plt.plot(signal_hat2[0:N] - wse_along_stats.reference,
                        label='linear')
                plt.plot(meas[0:N] - wse_along_stats.reference,
                        label='meas')
                plt.legend()
                plt.ylabel('wse')
                plt.subplot(2,1,2)
                plt.plot(grad_norm[0:N], label='MAP term')
                plt.plot(grad_hw_norm[0:N], label='HW term')
                plt.plot(grad_tot[0:N], '--', label='total')
                plt.legend()
                plt.ylabel('wse_gradient')
                #
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(signal_b[N:] - width_along_stats.reference,
                        label='bayes')
                plt.plot(signal_hat[N:] - width_along_stats.reference,
                        label='current est')
                plt.plot(signal_hat1[N:] - width_along_stats.reference,
                        label='next est')
                plt.plot(signal_hat2[N:] - width_along_stats.reference,
                        label='linear')
                plt.plot(meas[N:] - width_along_stats.reference,
                        label='meas')
                plt.legend()
                plt.ylabel('width')
                plt.subplot(2,1,2)
                plt.plot(grad_norm[N:], label='MAP term')
                plt.plot(grad_hw_norm[N:], label='HW term')
                plt.plot(grad_tot[N:], '--', label='total')
                plt.ylabel('width_gradient')
                plt.legend()
                #
                plt.figure()
                plt.plot(
                    signal_b[N:] - width_along_stats.reference,
                    signal_b[0:N] - wse_along_stats.reference,
                    'o', label='bayes')
                plt.plot(
                    signal_hat2[N:] - width_along_stats.reference,
                    signal_hat2[0:N] - wse_along_stats.reference,
                    'x', label='linear')
                plt.plot(
                    signal_hat[N:] - width_along_stats.reference,
                    signal_hat[0:N] - wse_along_stats.reference,
                    'x', label='current est')
                plt.plot(
                    signal_hat1[N:] - width_along_stats.reference,
                    signal_hat1[0:N] - wse_along_stats.reference,
                    'x', label='next est')
                plt.plot(
                    meas[N:] - width_along_stats.reference,
                    meas[0:N] - wse_along_stats.reference,
                    'o', label='meas')
                dw_x = np.linspace(-200,200)
                dh_y = height_width.sample(dw_x, x_key='width')
                plt.plot(dw_x, dh_y)
                plt.legend()
                plt.show()
                breakpoint()
                # do the update
                signal_hat = signal_hat2


            plotem = True
            if plotem:
                plt.figure()
                plt.plot(
                    signal_b[N:] - width_along_stats.reference,
                    signal_b[0:N] - wse_along_stats.reference,
                    'o', label='bayes')
                plt.plot(
                    signal_hat[N:] - width_along_stats.reference,
                    signal_hat[0:N] - wse_along_stats.reference,
                    'x', label='current est')
                plt.plot(
                    signal_hat1[N:] - width_along_stats.reference,
                    signal_hat1[0:N] - wse_along_stats.reference,
                    'x', label='next est')
                plt.plot(
                    meas[N:] - width_along_stats.reference,
                    meas[0:N] - wse_along_stats.reference,
                    'o', label='meas')
                plt.legend()
                plt.show()
            '''
            Signal_hat.append(signal_hat)
            Signal_hat_u.append(np.diag(post_cov))
            Post_cov.append(post_cov)
        #breakpoint()
        try:
            bayes.signal = np.array(Signal_hat).T
        except AssertionError as e:
            print(e)
        try:
            bayes.signal_u = np.array(Signal_hat_u).T
        except AssertionError as e:
            print(e)
        try:
            bayes.signal_post_cov = np.moveaxis(
                np.array(Post_cov), 0, -1)
        except AssertionError as e:
            print(e)
        return bayes


    def unpack_joint(self):
        bayes_wse = BayesData()
        bayes_wse.signal_key = 'wse'
        bayes_wse.stretch_name = self.stretch_name
        bayes_width = BayesData()
        bayes_width.signal_key = 'width'
        bayes_width.stretch_name = self.stretch_name
        N = int(len(self.signal[:,0])/2)
        bayes_wse.signal_mean = self.signal_mean[0:N]
        bayes_width.signal_mean = self.signal_mean[N:]
        wse_hats = []
        wse_hats_u = []
        width_hats = []
        width_hats_u = []
        wse_post_covs = []
        width_post_covs = []
        wse_width_post_covs = []
        # go through each time/cycle observation in the stack
        for j, jnk in enumerate(self.signal[-1]):
            signal_hat = self.signal[:,j]
            post_cov = self.signal_post_cov[:,:,j]
            # populate the output arrays
            wse_hats.append(signal_hat[0:N])
            wse_post_covs.append(post_cov[0:N,0:N])
            wse_hats_u.append(np.diag(post_cov[0:N,0:N]))
            width_hats.append(signal_hat[N:])
            width_post_covs.append(post_cov[N:,N:])
            width_hats_u.append(np.diag(post_cov[N:,N:]))
            wse_width_post_covs.append(post_cov[0:N,N:])
        bayes_wse.signal = np.array(wse_hats).T
        bayes_width.signal = np.array(width_hats).T
        bayes_wse.signal_u = np.array(wse_hats_u).T
        bayes_width.signal_u = np.array(width_hats_u).T
        bayes_wse.post_cov = np.moveaxis(
            np.array(wse_post_covs), 0, -1)
        bayes_width.post_cov = np.moveaxis(
            np.array(width_post_covs), 0, -1)
        bayes_wse_width_post_cov = np.moveaxis(
            np.array(wse_width_post_covs), 0, -1)
        # TODO: handle the node_id, p_lat vars etc...
        return bayes_wse, bayes_width, bayes_wse_width_post_cov






