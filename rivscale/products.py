'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author (s): Brent Williams

Mimics the product class from RiverObs and the SWOT project "python" repo
'''
from collections import OrderedDict as odict
import textwrap
import numpy as np
from datetime import datetime

import swot.constants
from swot.product import Product, ProductTesterMixIn
from SWOTWater.products.constants import FILL_VALUES
from swot.lr.base_classes import AttrFillerMixIn

import rivscale.estimate
import rivscale.plot
import matplotlib.pyplot as plt
import scipy.interpolate

def textjoin(text):
    """Dedent join and strip text"""
    text = textwrap.dedent(text)
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

#DIMENSIONS_4D = odict([
#    ['num_times', 0], ['num_nodes', 0], ['num_reaches',0], ['num_percentiles', 0]])
#DIMENSIONS_2D = odict([['num_times', 0], ['num_nodes', 0]])
#DIMENSIONS_PCNT = odict([['num_percentiles', 0], ['num_nodes', 0]])
#DIMENSIONS_COV = odict([['num_times', 0], ['num_nodes', 0], ['num_nodes', 0]])

DIMENSIONS_ALL = odict([
    ['num_nodes', 0],
    ['num_times', 0],
    ['num_reaches', 0],
    ['num_percentiles', 0],
    ['num_nodes2', 0],
    ['num_hw_params', 0]])
DIMENSIONS_SAVG = odict([
    ['num_reaches', 0],
    ['num_times', 0],
    ['num_percentiles', 0]])
DIMENSIONS_ALONG = odict([
    ['num_reaches', 0],
    ['num_nodes', 0],
    ['num_percentiles', 0]])
DIMENSIONS_2D = odict([
    ['num_nodes', 0],
    ['num_times', 0]])
DIMENSIONS_PCNT = odict([
    ['num_nodes', 0],
    ['num_percentiles', 0]])
DIMENSIONS_PCNT2 = odict([
    ['num_times', 0],
    ['num_percentiles', 0]])
DIMENSIONS_COV = odict([
    ['num_nodes', 0],
    ['num_nodes2', 0]])
DIMENSIONS_POSTCOV = odict([
    ['num_nodes', 0],
    ['num_nodes2', 0],
    ['num_times', 0]])

class StretchStack(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding 2D multitemporal data
            """)}],
        ['stretch_name',{'dtype':'str', 'value': textjoin("""
            Name given to this stretch instance (e.g., center reach
            or river name)
            """)}],
        ])
    DIMENSIONS = DIMENSIONS_ALL
    VARIABLES = odict([
        ['reaches', odict([['dimensions', odict([['num_reaches', 0]])]])],
        ['dist_out', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['local_node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['time_id', odict([['dimensions', odict([['num_times', 0]])]])],
        ['cycle_id', odict([['dimensions', odict([['num_times', 0]])]])],
        #['swot_time', odict([['dimensions', odict([['num_times', 0]])]])],
        ['date_hour', odict([['dimensions', odict([['num_times', 0]])]])],
        ['wse', odict([['dimensions', DIMENSIONS_2D]])],
        ['width', odict([['dimensions', DIMENSIONS_2D]])],
        ['area_total', odict([['dimensions', DIMENSIONS_2D]])],
        ['wse_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['width_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['area_tot_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['node_q_b', odict([['dimensions', DIMENSIONS_2D]])],
        ['dark_frac', odict([['dimensions', DIMENSIONS_2D]])],
    ])

    def plot(
            self,
            wse_reference=None,
            width_reference=None,
            x_key='dist_out', # or 'time_id'
            outdir=None,
            show=False):
        if wse_reference is not None:
            # plot the wse and wse_anom together
            rivscale.plot.plot_stretch_stack(
                self,
                x_key=x_key,
                y_keys=['wse','wse'],
                y_reference=[wse_reference, wse_reference],
                y_anom=[False, True],
                outdir=outdir)
        if width_reference is not None:
            # plot the width and width_anom together
            rivscale.plot.plot_stretch_stack(
                self,
                x_key=x_key,
                y_keys=['width','width'],
                y_reference=[width_reference, width_reference],
                y_anom=[False, True],
                outdir=outdir)
        if (wse_reference is None) and (width_reference is None):
            # plot the height and width together
            rivscale.plot.plot_stretch_stack(
                self,
                x_key=x_key,
                y_keys=['wse','width'],
                y_reference=[None, None],
                y_anom=[False, False],
                outdir=outdir) 
        """
        rivscale.plot.plot_stretch_stack(
            self,
            x_key='time_id',
            y_keys=['wse','width'],
            outdir=outdir,
            marker='o')
        """
        if show:
            plt.show()

class AlongStretchStats(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding along-river statistics and spatial-scale covariance
            estimates derived from 2D multitemporal data
            """)}],
        ['signal_key',{'dtype':'str', 'value':'wse, width, or dark_frac'}],
        ['stretch_name',{'dtype':'str', 'value': textjoin("""
            Name given to this stretch instance (e.g., center reach
            or river name)
            """)}],
        ['char_length_tau',{'dtype':'float', 'value':100000}],
        ['prior_unc_alpha',{'dtype':'float', 'value':1.5}],
        ])
    DIMENSIONS = DIMENSIONS_ALL
    VARIABLES = odict([
        ['reaches', odict([['dimensions', odict([['num_reaches', 0]])]])],
        ['dist_out', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['local_node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['reference', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['std', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['count', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['percentiles', odict([['dimensions', DIMENSIONS_PCNT]])],
        ['percentile_list', odict([['dimensions', odict([['num_percentiles', 0]])]])],
    ])


    def plot(self, outdir=None):
        rivscale.plot.plot_stretch_stats(
            self,
            x_key='dist_out',
            outdir=outdir)

    @classmethod
    def from_StretchStack(
            cls,
            stretch_stack,
            signal_key,
            percentiles=[5, 25, 32, 50, 68, 75, 95],
            kernel_size=35,
            char_length_tau=None,
            prior_unc_alpha=None):
        stats = cls()
        stats.signal_key = signal_key
        # copy over common items
        stats.stretch_name = stretch_stack.stretch_name
        stats.reaches = stretch_stack.reaches.copy()
        stats.dist_out = stretch_stack.dist_out.copy()
        stats.node_id = stretch_stack.node_id.copy()
        stats.local_node_id = stretch_stack.local_node_id.copy()
        # get the reference profile
        stats.reference = rivscale.estimate.get_med_profile(
            stretch_stack[signal_key],
            stretch_stack['dist_out'],
            kernel_size = kernel_size)
        # TODO: get the spatial covariance estimate from data
        if char_length_tau is not None:
            stats.char_length_tau = char_length_tau
        if prior_unc_alpha is not None:
            stats.prior_unc_alpha = prior_unc_alpha
        # get the stats
        stats.mean = np.nanmean(stretch_stack[signal_key], axis=1)
        stats.std = np.nanstd(stretch_stack[signal_key], axis=1)
        mask = np.zeros(np.shape(stretch_stack[signal_key]))
        mask[np.isfinite(stretch_stack[signal_key])] = 1
        stats.count = np.nansum(mask, axis=1)
        stats.percentiles_list = percentiles
        ptiles = np.zeros((
            len(stats.mean),
            len(percentiles),
            )) + np.nan
        for k, ptile in enumerate(percentiles):
            ptiles[:,k] = np.nanpercentile(stretch_stack[signal_key], ptile, axis=1)
        stats.percentiles = np.array(ptiles)
        stats.percentile_list = np.array(percentiles)
        return stats

    @classmethod
    def from_pekel_df(cls, df_list, reaches, name, in_stats):
        stats = cls()
        stats.stretch_name = name
        stats.reaches = np.array(reaches)
        stats.signal_key = 'width'
        def df_to_dict(df, node_id, dist_out):
            #df = df_in.sort_values('node_id')
            d = {'percentiles':[], 'percentile_list':[]}
            # Pekel occurrence threshold is packed in cycle field
            ptiles = np.sort(100-np.unique(df.cycle))
            for ptile in ptiles:
                this_df = df[df.cycle==100-ptile]
                w = np.zeros(np.shape(node_id))+np.nan
                this_node_id = np.array(this_df.node_id)
                for wid, nid in zip(this_df.width, this_df.node_id):
                    w[node_id==nid] = wid
                # fill in any missing nodes with the average
                avg_w = np.nanmean(w)
                w[np.isnan(w)] = avg_w
                d['percentiles'].append(w)
            d['percentile_list'] = ptiles
            return d
        # stack up all the variables
        d = {
            'node_id':None,
            'percentiles':None,
            'percentile_list':None,
            'dist_out':None}
        in_nodes = in_stats.node_id
        in_reaches = np.array([int(str(n)[0:10]+str(n)[-1]) for n in in_nodes])
        in_dist_out = in_stats.dist_out
        for k,reach in enumerate(reaches):
            this_df = df_list[k]
            #this_d = df_to_dict(this_df)
            this_msk = np.where(in_reaches==reach)
            this_node_id = in_nodes[this_msk]
            this_dist_out = in_dist_out[this_msk]
            this_d = df_to_dict(this_df, this_node_id, this_dist_out)
            if k ==0:
                d['node_id'] = this_node_id
                d['dist_out'] = this_dist_out
                # assumes same percentile list for every df
                d['percentile_list'] = np.array(this_d['percentile_list'])
                d['percentiles'] = np.array(this_d['percentiles'])
            else:
                d['node_id'] = np.append(d['node_id'], this_node_id)
                d['dist_out'] = np.append(d['dist_out'], this_dist_out)
                d['percentiles'] = np.append(
                    d['percentiles'],
                    np.array(this_d['percentiles']), axis=1)
        for key in d.keys():
            if key == 'percentiles':
                stats[key] = d[key].T
            else:
                stats[key] = d[key]
        # put the 50%ile in as the reference
        msk = stats.percentile_list==50
        if np.sum(msk)>0:
            stats.reference = stats.percentiles[:,msk].squeeze()
        return stats

    def crop_to_reach(
            self,
            reach_id=None):
        stats = AlongStretchStats()
        if reach_id is None:
            if self.stretch_name.isdigit():
                reach_id = self.stretch_name
            else:
                reach_id = 'bad'
        if not((reach_id.isdigit()) and (len(reach_id)==11)):
            print('reach_id is not a valid value, not cropping')
            return None
        # now get the mask
        reach_ids = np.array([str(n)[0:10]+str(n)[-1] for n in self.node_id])
        mask = np.where(reach_ids==reach_id)[0]
        stats.reaches = np.array([int(reach_id),])
        stats.signal_key = self.signal_key
        for key in set(self.variables.keys()) - set(['reaches',]):
            stats[key] = self[key][mask[0]:mask[-1]]
        return stats

class StretchAverageStats(Product):
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
        ['signal_key',{'dtype':'str', 'value':'wse, width, or dark_frac'}],
        #['reference_mean',{'dtype':'float', 'value':0.0}],
        #['reference_slope',{'dtype':'float', 'value':-9999.0}],
        ])
    DIMENSIONS = DIMENSIONS_ALL
    VARIABLES = odict([
        ['reaches', odict([['dimensions', odict([['num_reaches', 0]])]])],
        ['dist_out', odict([['dimensions', odict([['num_times', 0]])]])],
        ['time_id', odict([['dimensions', odict([['num_times', 0]])]])],
        ['cycle_id', odict([['dimensions', odict([['num_times', 0]])]])],
        ['mean', odict([['dimensions', odict([['num_times', 0]])]])],
        ['mean_reference', odict([['dimensions', odict([['num_times', 0]])]])],
        ['std', odict([['dimensions', odict([['num_times', 0]])]])],
        ['count', odict([['dimensions', odict([['num_times', 0]])]])],
        ['percentiles', odict([['dimensions', DIMENSIONS_PCNT]])],
        ['percentile_list', odict([['dimensions', odict([['num_percentiles', 0]])]])],
        ['slope', odict([['dimensions', odict([['num_times', 0]])]])],
        ['slope_reference', odict([['dimensions', odict([['num_times', 0]])]])],
    ])
    def plot(self, outdir=None, show=False):
        rivscale.plot.plot_stretch_stats(
            self,
            x_key='time_id',
            outdir=outdir)
        if show:
            plt.show()

    @classmethod
    def from_reach_df(cls, df_in, reach, signal_key):
        stats = cls()
        stats.signal_key = signal_key
        stats.reaches = [reach,]
        stats.stretch_name = '{}'.format(reach)
        #
        df = df_in.copy()
        df[df[signal_key]<-1e5] = np.nan
        # crop out all reaches except those in the reach list
        #df = df[~df['reach_id'].isin(reaches)]
        df = df[df['reach_id']==int(reach)]
        #
        dist_out = []
        time_id = []
        mean = []
        std = []
        if signal_key=='wse':
            df[df['slope']<-1e5] = np.nan
            slope = []
        time_ids = np.unique(np.array(np.floor(df['time']/60/60))).astype(int)
        for time_i in time_ids:
            times_id = np.floor(df['time']/60/60).astype(int)
            this_df = df[times_id==time_i]
            # TODO handle window over desired reach
            dist_out.append(this_df['dist_out'])
            mean.append(this_df[signal_key])
            std.append(this_df[signal_key+'_u'])
            time_id.append(time_i)
            if signal_key=='wse':
                slope.append(this_df['slope'])
        #breakpoint()
        stats.time_id = np.array(time_id).squeeze()
        stats.dist_out = np.array(dist_out).squeeze()
        stats.mean = np.array(mean).squeeze()
        stats.std = np.array(std).squeeze() 
        # just use the mean of the data as reference
        stats.mean_reference = np.nanmean(
            mean) * np.ones_like(stats.mean)
        if signal_key=='wse':
            stats.slope_reference = np.nanmean(
                slope) * np.ones_like(stats.mean)
        return stats

    @classmethod
    def from_StretchStack(
            cls,
            stretch_stack,
            signal_key,
            along_stats=None,
            percentiles=[5, 25, 32, 50, 68, 75, 95],
            reach_id=None,
            average_method='weighted',
            slope_method='bayes'):
        # average_method = 'simple', 'weighted', 'bayes_weighted'
        # slope_method = 'simple', 'bayes'
        # TODO: add method to fit a line to the deviation from reference
        stats = cls()
        # copy common things
        stats.stretch_name = stretch_stack.stretch_name
        stats.signal_key = signal_key
        stats.reaches = stretch_stack.reaches.copy()
        stats.time_id = stretch_stack.time_id.copy()
        stats.cycle_id = stretch_stack.cycle_id.copy()
        # get stats
        signal = stretch_stack[signal_key]
        signal_u = stretch_stack[signal_key+'_u']
        first_node = 0
        last_node = -1
        if along_stats is not None:
            if ('bayes' in average_method) or ('bayes' in slope_method):
                bayes = BayesData.simple(
                    stretch_stack,
                    along_stats,
                    signal_key)
                    #char_length_tau,
                    #prior_unc_alpha)
            if 'bayes' in average_method:
                signal = bayes.signal
                signal_u = bayes.signal_u
        # get weighting mask for reach average
        window = np.ones_like(signal)
        window_valid = window.copy()
        #window[~np.isfinite(stretch_stack[signal_key])] = 0
        #window_valid[~np.isfinite(stretch_stack[signal_key])] = np.nan
        if reach_id is None:
            if stretch_stack.stretch_name.isdigit():
                reach_id = stretch_stack.stretch_name
            else:
                reach_id = 'bad'
        if ((reach_id.isdigit()) and (len(reach_id)==11)):
            print('windowing to reach {}'.format(reach_id))
            reach_ids = np.array(
                [str(n)[0:10]+str(n)[-1] for n in stretch_stack.node_id])
            #window_mask = np.where(reach_ids==reach_id)[0]
            window[reach_ids!=reach_id] = 0
            window_valid[reach_ids!=reach_id] = np.nan
            inds = np.where(
                np.array(reach_ids) == stretch_stack.stretch_name)[0]
            first_node = inds[0]
            last_node = inds[-1]
        
        # inverse variance weighting with along-river windowing
        weight = 1/signal_u**2 * window
        sum_weight = np.nansum(weight, axis=0)
        sum_window = np.nansum(window, axis=0)
        #
        stats.dist_out = np.nanmean(stretch_stack.dist_out, axis=0)
        if along_stats is None:
            ref = np.zeros_like(signal[:,0])
        elif isinstance(along_stats, AlongStretchStats):
            ref = along_stats.reference.copy()
        reference = np.broadcast_to(
            ref, np.shape(signal.T)).T
        #stats.reference_mean = np.nanmean(
        #        np.array(reference) * weight, axis=0) / mean_weight
        # always do simple mean of reference
        #stats.mean_reference = np.nanmean(np.array(reference), axis=0)
        # weighted mean of ref too
        stats.mean_reference = np.nansum(
            reference * window, axis=0) / (
                sum_window)#).filled(np.nan)
        if isinstance(stats.mean_reference, np.ma.MaskedArray):
            stats.mean_reference.filled(np.nan)
        if 'simple' in average_method:
            stats.mean = np.nanmean(signal*window_valid - reference, axis=0) \
                + stats.mean_reference
        elif 'weighted' in average_method:
            stats.mean = np.nansum(
                (signal-reference) * weight, axis=0) / (
                    sum_weight) + stats.mean_reference#).filled(np.nan)
        if isinstance(stats.mean, np.ma.MaskedArray):
            stats.mean.filled(np.nan)
        #stats_mean = np.nansum(
        #        (stretch_stack[signal_key]-reference)*weight, axis=0) / (
        #                sum_weight) + stats.reference_mean
        #stats.mean = stats_mean.filled(np.nan)
        # TODO: output the estimate of the sqrt(variance)
        stats.std = np.nanstd((
            signal-reference) * window_valid, axis=0)
        mask = np.zeros(np.shape(signal))
        mask[np.isfinite((
            signal-reference) * window_valid)] = 1
        stats.count = np.nansum(mask, axis=0)
        stats.percentile_list = np.array(percentiles)
        ptiles = np.zeros((
            len(stats.mean),
            len(percentiles),
            )) + np.nan
        for k, ptile in enumerate(percentiles):
            ptiles[:,k] = np.nanpercentile(
                (signal-reference) * window_valid,
                ptile, axis=0) + stats.mean_reference
        stats.percentiles = ptiles
        # now compute slope
        if 'bayes' in slope_method:
            signal = bayes.signal
        else:
            signal = stretch_stack[signal_key]
        dist_out = np.broadcast_to(
            stretch_stack.dist_out, np.shape(signal.T)).T
        slope = (
            signal[last_node,:] - signal[first_node,:]) / (
            dist_out[last_node,:] - dist_out[first_node,:])
        slope_ref = (
            reference[last_node,:] - reference[first_node,:]) / (
            dist_out[last_node,:] - dist_out[first_node,:])
        stats.slope = slope
        stats.slope_reference = slope_ref
        # TODO: handle missing data in endpoints
        # TODO: get estimate of slope_u
        return stats

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
    """
    def plot(self): 
        # plot the wse and wse_anom together
        rivscale.plot.plot_stretch_stack(
            self,
            x_key=x_key,
            y_keys=['wse','wse'],
            y_reference=[wse_reference, wse_reference],
            y_anom=[False, True],
            outdir=outdir)
    """
    @classmethod
    def simple(
            cls,
            stretch_stack,
            stats,
            signal_key,
            char_length_tau=None,
            prior_unc_alpha=None):
        bayes = cls()
        bayes.stretch_name = stretch_stack.stretch_name
        bayes.signal_key = signal_key
        bayes.signal_mean = stats.reference.copy()
        time_key = 'time_id'
        if (char_length_tau is not None) and (prior_unc_alpha is not None):
            bayes.signal_cov = rivscale.reconstruct.generate_cov_matrix(
                stretch_stack,
                char_length_tau,
                prior_unc_alpha)
        elif isinstance(stats, AlongStretchStats):
            # use what is in the stats
            bayes.signal_cov = rivscale.reconstruct.generate_cov_matrix(
                stretch_stack,
                stats.char_length_tau,
                stats.prior_unc_alpha)
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
            rho_wse_width=0.7
            ):
        bayes = cls()
        bayes.signal_key = 'joint_wse_width'
        bayes.stretch_name = stretch_stack.stretch_name
        # get the mean and cov of the stacked wse and width
        N = len(wse_along_stats.reference)
        # create the stacked mean
        mn = np.concatenate([
            wse_along_stats.reference, width_along_stats.reference
            ])
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
            wse_cov = rivscale.reconstruct.exponential_cov(
                stretch_stack['dist_out'],
                char_length_tau=wse_along_stats.char_length_tau,
                prior_unc_alpha=wse_along_stats.prior_unc_alpha)
            # constrain the height and width std magnitudes using the h/w-model
            dw_dh = rivscale.reconstruct.get_dw_dh_from_model(
                stretch_stack, height_width, j)
            this_prior_unc_alpha_width = dw_dh * wse_along_stats.prior_unc_alpha
            #breakpoint()
            width_cov = rivscale.reconstruct.exponential_cov(
                stretch_stack['dist_out'],
                char_length_tau=width_along_stats.char_length_tau,
                prior_unc_alpha=this_prior_unc_alpha_width)
            #rivscale.reconstruct.generate_cov_matrix(
            #    stretch_stack,
            #    width_along_stats.char_length_tau,
            #    this_prior_unc_alpha_width)
            wse_width_cov = (
                rho_wse_width * np.real(scipy.linalg.sqrtm(wse_cov)) @ (
                    np.real(scipy.linalg.sqrtm(width_cov.T))))
            Ry = np.block([
                [wse_cov, wse_width_cov.T],
                [wse_width_cov, width_cov],
                #[wse_along_stats.cov[:,:,j], wse_width_cov[:,:,j].T],
                #[wse_width_cov[:,:,j], width_along_stats.cov[:,:,j]]
                ])
            #print(np.linalg.cond(Ry))
            #if np.linalg.cond(Ry) > 1e
            # call the estimator
            signal_hat, post_cov = rivscale.reconstruct.reconstruct_one_time_obs(
                meas, meas_u, Ry, mn, nodes)
            Signal_hat.append(signal_hat)
            Signal_hat_u.append(np.diag(post_cov))
            Post_cov.append(post_cov)
        bayes.signal = np.array(Signal_hat).T
        bayes.signal_u = np.array(Signal_hat_u).T
        bayes.signal_post_cov = np.moveaxis(
            np.array(Post_cov), 0, -1)
        return bayes

    def unpack_joint(self):
        bayes_wse = BayesData()
        bayes_wse.signal_key = 'wse'
        bayes_wse.stretch_name = self.stretch_name
        bayes_width = BayesData()
        bayes_width.signal_key = 'width'
        bayes_width.stretch_name = self.stretch_name
        N = int(len(self.signal[:,0])/2)
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
        return bayes_wse, bayes_width, bayes_wse_width_post_cov

class HeightWidthModel(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding stretch_average estimates (over all nodes) of wse
            and width derived from 2D multitemporal data
            """)}],
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
            wse_stretch_avg,
            width_stretch_avg,
            width_along_stats=None,
            wse_anom=True,
            sigma_n=50):
        height_width = cls()
        if width_along_stats is None:
            ptile_list = [5, 25, 32, 50, 68, 75, 95]
        else:
            ptile_list = width_along_stats.percentile_list
        # get percntiles of measured reach data
        Pm = np.nanpercentile(width_stretch_avg.mean, ptile_list)
        wse = wse_stretch_avg.mean
        if wse_anom:
            wse = wse - wse_stretch_avg.mean_reference
        Ph = np.nanpercentile(wse, ptile_list)

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
        return height_width

    def sample(self, x, x_key='width',kind='linear'):
        """
        x: sample location(s) of the 'signal_key' dimension
        returns: the sampled value(s) of other dimension
        e.g., if signal_key=='width' retun the wse at the point
        by interpolating in between the width samples and 
        """
        y_key = 'wse'
        if x_key=='wse':
            y_key='width'
        intrp = scipy.interpolate.interp1d(
            self[x_key+'_coords'],
            self[y_key+'_coords'],
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate"
            )
        return intrp(x)

