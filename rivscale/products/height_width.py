'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author (s): Brent Williams

Container for holding height/width relationship information.

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

import rivscale.products.stretch_average
from scipy.stats import spearmanr

class HeightWidthModel(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding stretch_average estimates (over all nodes) of wse
            and width derived from 2D multitemporal data
            """)}],
        ['stretch_name',{'dtype':'str', 'value': textjoin("""
            Name given to this stretch instance (e.g., center reach
            or river name)
            """)}],
        ['width_err',{'dtype':'float', 'value':-1}],
        ['wse_err',{'dtype':'float', 'value':-1}],
        ['tot_err',{'dtype':'float', 'value':-1}],
        ['count',{'dtype':'int', 'value':0}],
        ['spearman_r',{'dtype':'float', 'value':-2}],
        ['spearman_p_value',{'dtype':'float', 'value':-1}],
        ])
    DIMENSIONS = odict([['num_hw_params',0],])
    VARIABLES = odict([
        ['width_coords', odict([['dimensions', odict([['num_hw_params', 0]])]])],
        ['wse_coords', odict([['dimensions', odict([['num_hw_params', 0]])]])],
        #['hw_params', odict([['dimensions', odict([['num_hw_params', 0]])]])],
        #['hw_params_err', odict([['dimensions', odict([['num_hw_params', 0]])]])],
    ])
    #for name, reference in VARIABLES.items():
    #    reference['dimensions'] = DIMENSIONS
    @classmethod
    def from_objects(
            cls,
            cfg,
            wse_stretch_avg,
            width_stretch_avg,
            width_along_stats=None,
            wse_anom=True,
            width_anom=True,
            stretch_name=None,
            ptile_list = [5, 25, 32, 50, 68, 75, 95],
            snapit=True
            ):#TODO: pass in uncertainty as well for snapping
        if 'sigma_n' not in cfg.keys():
            cfg['sigma_n'] = 50
        sigma_n = cfg['sigma_n']
        height_width = cls()
        if stretch_name is not None:
            height_width.stretch_name = stretch_name
        else:
            if isinstance(wse_stretch_avg,
                    rivscale.products.stretch_average.StretchAverageStats):
                height_width.stretch_name = wse_stretch_avg.stretch_name
            elif isinstance(width_stretch_avg,
                    rivscale.products.stretch_average.StretchAverageStats):
                height_width.stretch_name = width_stretch_avg.stretch_name
            else:
                height_width.stretch_name = 'arrays'
        #if width_along_stats is None:
            #ptile_list = [5, 25, 32, 50, 68, 75, 95]
        #else:
        if width_along_stats is not None:
            ptile_list = width_along_stats.percentile_list
        # get percntiles of measured reach data
        #Pm = np.nanpercentile(width_stretch_avg.mean, ptile_list)
        if isinstance(wse_stretch_avg,
                rivscale.products.stretch_average.StretchAverageStats):
            wse = wse_stretch_avg.mean
            wse_reference = wse_stretch_avg.mean_reference
        else:
            # assume it is an array
            wse = wse_stretch_avg.copy()
            wse_reference = np.nanmedian(wse)
        if isinstance(width_stretch_avg,
                rivscale.products.stretch_average.StretchAverageStats):
            width = width_stretch_avg.mean
            width_reference = width_stretch_avg.mean_reference
        else:
            # assume it is an array
            width = width_stretch_avg.copy()
            width_reference = np.nanmedian(width)
        if wse_anom:
            wse = wse - wse_reference
        if width_anom:
            width = width - width_reference
        # check that wse and width are all valid over the same places
        good_msk = np.logical_and(np.isfinite(wse), np.isfinite(width))
        if np.sum(good_msk)==0:
            return height_width
        wse = wse[good_msk]
        width = width[good_msk]
        Ph = np.nanpercentile(wse, ptile_list)
        Pm = np.nanpercentile(width, ptile_list)
        if width_along_stats is None:
            # just use the measuered stats for width
            Pl_hat = Pm
        else:
            # percentile method using input along-stats for width
            # adjusting the distribution mean and std baesed on the
            # measured distribution (using simple Gaussian assumption)
            Pg = np.nanmean(width_along_stats.percentiles, axis=0)
            # now find Pl the no-noise width measurement distribution
            # assuming zeros mean noise with known sqrt(variances) (sigma_n).
            if (50 in ptile_list):
                # get the mean of the prior and measured width distribution
                ind = np.where(ptile_list==50)
                #breakpoint()
                mu_g = Pg[ind]
                mu_m = Pm[ind]
            else:
                print('50th ptile not in prior percentile list')
                return None
            #sigma_g = 1/2*(Pg[4]-Pg[2])
            #sigma_m = 1/2*(Pm[4]-Pm[2])
            if (25 in ptile_list) and (75 in ptile_list):
                # Use IQR to adjust the prior width distribution
                # to the measured distribution (using Gaussian assumptions)
                ind75 = np.where(ptile_list==75)
                ind25 = np.where(ptile_list==25)
                sigma_g = 1/1.349*(Pg[ind75]-Pg[ind25])
                sigma_m = 1/1.349*(Pm[ind75]-Pm[ind25])
            else:
                print('25th or 75th ptile not in prior percentile list')
                return None
            if sigma_n>sigma_m:
                #print('replacing noise std')
                sigma_n = sigma_m * 0.7
            a = np.sqrt(sigma_m**2-sigma_n**2) / sigma_g
            Pl_hat = a*(Pg - mu_g) + mu_m

        # now match-up the height and width percentiles
        # and make a wse a piecewise-linear function of width
        # ignore nosie on height for now (TODO: handle it)
        height_width.width_coords = Pl_hat
        height_width.wse_coords = Ph
        # compute the residual error from model fit
        w_hat = height_width.sample(wse, x_key='wse')
        h_hat = height_width.sample(width, x_key='width')
        dw = w_hat - width
        dh = h_hat - wse
        # use RMSE from fit curve for errors
        # handle length 1 cases
        if len(dw)==1:
            dw = np.array([dw,])
        if len(dh)==1:
            dh = np.array([dh,])
        height_width.width_err = np.sqrt(np.nanmean(dw**2))
        height_width.wse_err = np.sqrt(np.nanmean(dh**2))
        height_width.count = np.sum(good_msk)
        #
        msk = np.logical_and(np.isfinite(wse), np.isfinite(width))
        if np.sum(msk)>0:
            sp, p_val = spearmanr(wse[msk], width[msk])
            height_width.spearman_r = sp
            height_width.spearman_p_value = p_val
        # TODO: compute average bank slope?
        if snapit:
            # TODO: find distance to "closest" point
            sig_wse = height_width.wse_err
            sig_width = height_width.width_err
            wse_est, width_est, err = height_width.snap_to_curve(
                wse, width, sig_wse, sig_width, wse)
            height_width.tot_err = err
            # reestimate the wse_err and width_err using the closest point
            dw = width_est - width
            dh = wse_est - wse
            # use RMSE from fit curve for errors
            # handle length 1 cases
            if len(dw)==1:
                dw = np.array([dw,])
            if len(dh)==1:
                dh = np.array([dh,])
            height_width.width_err = np.sqrt(np.nanmean(dw**2))
            height_width.wse_err = np.sqrt(np.nanmean(dh**2))
        #breakpoint()
        return height_width

    def sample(self, x_in, x_key='width',kind='linear'):
        """
        x: sample location(s) of the 'signal_key' dimension
        returns: the sampled value(s) of other dimension
        e.g., if signal_key=='width' retun the wse at the point
        by interpolating in between the width samples and 
        """
        y_key = 'wse'
        if x_key=='wse':
            y_key='width'
        x = x_in.copy()
        if isinstance(x_in, np.ma.MaskedArray):
            x = x.filled(np.nan)
        intrp = scipy.interpolate.interp1d(
            self[x_key+'_coords'],
            self[y_key+'_coords'],
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate"
            )
        y = intrp(x)
        if isinstance(x_in, np.ma.MaskedArray):
            mask = ~np.isfinite(y)
            y = np.ma.masked_array(y, mask=mask)
        return intrp(x)

    def sample_deriv(self, x, x_key='width', kind='linear', delta=0.1):
        """
        estimate the derivative with finite differences
        """
        y_plus = self.sample(x + delta, x_key, kind)
        y_minus = self.sample(x - delta, x_key, kind)
        return (y_plus -y_minus) / (2 * delta)

    def snap_to_curve(self, wse, width, sig_wse, sig_width, wse0):
        """
        find 'closest' point on the curve to each point with given covariance
        (assuming each point is indep)
        
        Basically just do a non-linear max-liklihood estimate of the
        observations given that the 'true' value lies on the curve
        and the height/width measurements are uncorrelated.
        """
        from scipy.optimize import minimize
        def ml_error(wse_t,
                wse_m, width_m, sig_wse, sig_width, height_width): 
            width_t = height_width.sample(wse_t, x_key='wse')
            return np.sum((wse_m - wse_t)**2 / sig_wse**2 + (
                width_m - width_t)**2 / sig_width**2)
        #
        res = minimize(ml_error, wse0, method='nelder-mead',
                args=(wse, width, sig_wse, sig_width, self),
                options={'xatol': 1e-8, 'disp': True})
        #
        wse_est = res.x
        width_est = self.sample(wse_est, x_key='wse')
        err = ml_error(wse_est, wse, width, sig_wse, sig_width, self)
        return wse_est, width_est, np.sqrt(err)

    def compute_per_pass_data(
            self,
            wse_data,
            width_data,
            granule_id=None,
            cross_track=None,
            wse_ref=0.0,
            width_ref=0.0,
            cfg=None):
        """
        wse_data, width_data can be stretch_average obects
        or arrays, in which case we also need granule_id, cross-track,
        wse_ref and width_ref
        """
        per_pass = {
            'pass_id':[],
            'width_bias':[],
            'width':[],
            'wse':[],
            'hw':[],
            'spearman':[],
            'cross_track':[]
            }
        if isinstance(wse_data, np.ndarray):
            width = width_data.copy()
            wse = wse_data.copy()
            if granule_id is None:
                return None
            if cross_track is None:
                return None
        else:
            #breakpoint()
            width = width_data.mean
            wse = wse_data.mean
            wse_ref = np.nanmedian(wse_data.mean_reference)
            width_ref = np.nanmedian(width_data.mean_reference)
            wse_med = np.median(wse)
            width_med = np.median(width)
            granule_id = wse_data.granule_id.copy()
            cross_track = wse_data.cross_track
        # get the pass_id
        pid = np.array([g.split('_')[1] for g in granule_id])
        upid = np.unique(pid)
        # loop over passes
        for p in upid:
            wse0 = wse[pid==p].flatten()
            width0 = width[pid==p].flatten()
            xtrk = cross_track[pid==p].flatten()
            # also compute spearman
            msk0 = np.logical_and(np.isfinite(width0), np.isfinite(wse0))
            res = scipy.stats.spearmanr(width0[msk0], wse0[msk0])
            # now plot do a height/width model for each pass
            if cfg is None:
                cfg = {'sigma_n':50.0, 'use_pekel':False}
            # add reference that is taken off inside constructor
            this_hw = self.from_objects(cfg, wse0, width0,
                wse_anom=False, width_anom=False)
            width_bias = this_hw.sample(
                np.array([wse_ref,]), x_key='wse')[0] - width_ref
            
            # accummulate
            per_pass['pass_id'].append(p)
            per_pass['cross_track'].append(np.nanmedian(xtrk))
            per_pass['width_bias'].append(width_bias)
            per_pass['width'].append(width0)
            per_pass['wse'].append(wse0)
            #per_pass['width_ref'].append(width_ref)
            #per_pass['wse_ref'].append(wse_ref)
            per_pass['hw'].append(this_hw)
            per_pass['spearman'].append(res.correlation)
        
        return per_pass 

    def plot(
            self,
            wse_data=None,
            width_data=None,
            granule_id=None,
            outdir=None,
            show=False,
            title_tag=None,
            cfg=None,
            newfig=True,
            line_color='k',
            label_prefix='',
            delta=False,
            show_all_pass=True):
        cross_track = None
        if newfig:
            plt.figure(figsize=(7,7))
        this_label = '{}model fit'.format(label_prefix)
        if (wse_data is not None) and (
                width_data is not None):
            label = None
            #breakpoint()
            per_pass = self.compute_per_pass_data(
                wse_data,
                width_data,
                granule_id,
                )
            if isinstance(wse_data, np.ndarray):
                d_width = width_data.copy()
                d_wse = wse_data.copy()
                wse_ref = 0.0
                width_ref = 0.0
                wse_med = np.median(d_wse)
                width_med = np.median(d_width)
                # make them anomalies
                #d_wse = d_wse - np.nanmedian(d_wse)
                #d_width = d_width - np.nanmedian(d_width)
                if granule_id is not None:
                    gid=granule_id.copy()
            else:
                # assume it is a stretch average
                if title_tag is None:
                    label='stretch average'
                d_width = width_data.mean
                d_wse = wse_data.mean
                wse_ref = np.nanmedian(wse_data.mean_reference)
                width_ref = np.nanmedian(width_data.mean_reference)
                if delta:
                    d_width = d_width - width_data.mean_reference
                    d_wse = d_wse - wse_data.mean_reference
                    wse_ref = 0.0
                    width_ref = 0.0
                wse_med = np.median(d_wse)
                width_med = np.median(d_width)
                gid = wse_data.granule_id.copy()
                #breakpoint()
                cross_track = wse_data.cross_track
            if per_pass is None:
                plt.plot(d_width, d_wse, 'o', label=label)
                plot_all_pass=True
            else:
                #breakpoint()
                for k,p in enumerate(per_pass['pass_id']):
                    print('pass',p)
                    wse0 = per_pass['wse'][k]
                    width0 = per_pass['width'][k]
                    # also compute spearman
                    this_hw = per_pass['hw'][k]
                    width_bias = per_pass['width_bias'][k]
                    spearman = per_pass['spearman'][k]
                    xtrk = per_pass['cross_track'][k]
                    # now plot it
                    this_label = 'pass {}, $\gamma_s$={:1.2f}, offset {:1.2f}'.format(
                        p, spearman, width_bias)
                    
                    this_label = this_label + ', xtrk {:2.1f} (km)'.format(
                        xtrk) # keep sign
                    plt.plot(width0, wse0, 'o', label=this_label)
                    color = plt.gca().lines[-1].get_color()
                    plt.plot(this_hw.width_coords, this_hw.wse_coords,
                        '-',color=color, linewidth=2)
            msk = np.logical_and(np.isfinite(d_width), np.isfinite(d_wse))
            res = scipy.stats.spearmanr(d_width[msk], d_wse[msk])
            this_label = '{}model fit, tot $\gamma_s$={:1.2f}'.format(
                label_prefix, res.correlation)
        if line_color is None: 
            line_color = plt.gca().lines[-1].get_color()
        # plot the model
        if show_all_pass:
            widths = self.width_coords.copy()
            wses = self.wse_coords.copy()
            if not(delta):
                widths = widths + width_ref
                wses = wses + wse_ref
            #breakpoint()
            that_label = f'data median ({wse_med:1.2f}, {width_med:1.2f})'
            plt.plot(widths, wses,'-',
                color=line_color, linewidth=2,
                label=this_label)
            #plt.plot([width_ref],[wse_ref,],'x', color=line_color)
            plt.plot([width_med],[wse_med,],'s', color=line_color, label=that_label)
        #
        if delta:
            plt.xlabel('$\Delta$ width (m)')
            plt.ylabel('$\Delta$ wse (m)')
        else:
            plt.xlabel('width (m)')
            plt.ylabel('wse (m)')
        #plt.legend()
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.2),
            ncol=1,
            fancybox=True,
            shadow=True,
            borderaxespad=0
            )
        plt.grid()

        title = self.stretch_name
        if title_tag is not None:
            title = title + ' ' +title_tag
        fname = title + '_wse_vs_width'
        plt.title(title)
        plt.tight_layout()
        if outdir is not None:
            # create output dir if not exist
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            fname = fname.replace(' ','_')
            plt.savefig(os.path.join(outdir, fname), dpi=300)
            plt.close()
        if show:
            plt.show()






