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
import pandas as pd

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
            show=False,
            title_tag=None):
        if title_tag is None:
            title_tag = 'stretch stack data'
        if wse_reference is not None:
            # plot the wse and wse_anom together
            rivscale.plot.plot_stretch_stack(
                self,
                x_key=x_key,
                y_keys=['wse','wse'],
                y_reference=[wse_reference, wse_reference],
                y_anom=[False, True],
                outdir=outdir,
                title_tag=title_tag)
        if width_reference is not None:
            # plot the width and width_anom together
            rivscale.plot.plot_stretch_stack(
                self,
                x_key=x_key,
                y_keys=['width','width'],
                y_reference=[width_reference, width_reference],
                y_anom=[False, True],
                outdir=outdir,
                title_tag=title_tag)
        if (wse_reference is None) and (width_reference is None):
            # plot the height and width together
            rivscale.plot.plot_stretch_stack(
                self,
                x_key=x_key,
                y_keys=['wse','width'],
                y_reference=[None, None],
                y_anom=[False, False],
                outdir=outdir,
                title_tag=title_tag)
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

    def filter_node_outliers(
            self,
            along_stats,
            key='wse',# or width etc
            IQR_scale=5.0,
            plot=False):
        # handle data
        arr0 = self[key]
        ref = along_stats.reference
        p = along_stats.percentile_list
        p25 = along_stats.percentiles[:,p==25].squeeze()
        p75 = along_stats.percentiles[:,p==75].squeeze()
        IQR = p75 - p25
        # first filter global
        #p25_g = np.median(p25 - ref)
        #p75_g = np.median(p75 - ref)
        #IQR_g = (p75_g - p25_g) + np.zeros_like(ref)
        IQR_g0 = np.nanmean(IQR[IQR>0])
        IQR_g = IQR_g0 + np.zeros_like(ref)
        # for some reason Pekel sometimes give negative IQR? so hande it
        IQR[IQR<0] = IQR_g0
        arr1 = rivscale.filter.scaled_spread_outlier_rejector(
            arr0, ref, IQR_g, IQR_scale)
        # now filter per-node
        arr = rivscale.filter.scaled_spread_outlier_rejector(
            arr1, ref, IQR, IQR_scale)
        self[key] = arr
        #breakpoint()
        if plot:
            outlier_mask = np.logical_and(
                np.isfinite(arr0),
                np.isnan(arr))
            dist_out = np.broadcast_to(
                self.dist_out,
                np.shape(arr0.T)).T
            plt.figure(figsize=(10,5))
            plt.plot(dist_out, arr0)
            plt.plot(
                dist_out[outlier_mask],
                arr0[outlier_mask],'x')
            plt.plot(
                self.dist_out,
                ref + IQR_scale*IQR,'k', linewidth=2)
            plt.plot(
                self.dist_out,
                ref - IQR_scale*IQR,'k', linewidth=2)
            plt.plot(
                self.dist_out,
                ref + IQR_scale*IQR_g,'g', linewidth=2)
            plt.plot(
                self.dist_out,
                ref - IQR_scale*IQR_g,'g', linewidth=2)
            plt.grid()
            plt.xlabel('dist_out')
            plt.ylabel(key)

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


    def plot(self, outdir=None, show=False):
        rivscale.plot.plot_stretch_stats(
            self,
            x_key='dist_out',
            outdir=outdir,
            show=show)

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
    def from_pekel_df(
            cls,
            df_list,
            reaches,
            name,
            in_stats,
            kernel_size=11):
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
        df_tot = pd.concat(df_list)
        ptile_list= np.sort(100-np.unique(df_tot.cycle))
        for k,reach in enumerate(reaches):
            this_df = df_list[k]
            #this_d = df_to_dict(this_df)
            this_msk = np.where(in_reaches==reach)
            this_node_id = in_nodes[this_msk]
            this_dist_out = in_dist_out[this_msk]
            this_d = df_to_dict(this_df, this_node_id, this_dist_out)
            # handle missing percentiles (because no water mask?)
            #if len(this_d['percentile_list']) < len(ptile_list):
            # zero fill the missing one(s)?
            ptiles = []
            for ptile in ptile_list:
                if ptile in this_d['percentile_list']:
                    #breakpoint()
                    ind = np.where(this_d['percentile_list']==ptile)[0][0]
                    ptiles.append(this_d['percentiles'][ind])
                else:
                    #breakpoint()
                    ptiles.append(np.zeros_like(this_dist_out))
            #breakpoint()
            this_d['percentiles'] = np.array(ptiles)
            this_d['percentile_list'] = np.array(ptile_list)
            if k ==0:
                d['node_id'] = this_node_id
                d['dist_out'] = this_dist_out
                # assumes same percentile list for every df
                d['percentile_list'] = np.array(this_d['percentile_list'])
                d['percentiles'] = np.array(this_d['percentiles'])
            else:
                d['node_id'] = np.append(d['node_id'], this_node_id)
                d['dist_out'] = np.append(d['dist_out'], this_dist_out)
                #breakpoint()
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
        ref = stats.percentiles[:,msk].squeeze()
        ref_med = scipy.ndimage.median_filter(
            ref, size=kernel_size, mode='nearest')
        # median filter the ref profile
        # TODO: maybe should do mean filter?
        if np.sum(msk)>0:
            stats.reference = ref_med
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
        other = []
        other_key = 'wse'
        if signal_key=='wse':
            df[df['slope']<-1e5] = np.nan
            other_key = 'width'
            slope = []
        time_ids = np.unique(np.array(np.floor(df['time']/60/60))).astype(int)
        for time_i in time_ids:
            times_id = np.floor(df['time']/60/60).astype(int)
            this_df = df[times_id==time_i]
            # TODO handle window over desired reach
            dist_out.append(this_df['dist_out'])
            mean.append(this_df[signal_key])
            other.append(this_df[other_key])
            std.append(this_df[signal_key+'_u'])
            time_id.append(time_i)
            if signal_key=='wse':
                slope.append(this_df['slope'])
        #breakpoint()
        Other = np.array(other).squeeze()
        stats.time_id = np.array(time_id).squeeze()
        stats.dist_out = np.array(dist_out).squeeze()
        stats.mean = np.array(mean).squeeze()
        stats.std = np.array(std).squeeze()
        # need to make reference use the same samples for  wse and width
        nan_msk = np.zeros_like(Other) + np.nan
        nan_msk[np.logical_and(np.isfinite(stats.mean), np.isfinite(Other))] = 1
        # just use the mean of the data as reference
        stats.mean_reference = np.nanmean(
            mean * nan_msk) * np.ones_like(stats.mean)
        if signal_key=='wse':
            # TODO: should we also use the nanmask for slope reference?
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
        ['reaches', odict([['dimensions', odict([['num_reaches', 0]])]])],
        ['dist_out', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['time_id', odict([['dimensions', odict([['num_times', 0]])]])],
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
            x_key='dist_out', # or 'time_id'
            outdir=None,
            show=False):
        title_tag = 'Bayes'
        # create reference object
        wse_reference = AlongStretchStats()
        width_reference = AlongStretchStats()
        # cast to a stretch_stack object and use its plotter
        stretch_stack = StretchStack()
        stretch_stack.stretch_name = self.stretch_name
        if 'joint' not in self.signal_key:
            stretch_stack.dist_out = self.dist_out
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
            title_tag=title_tag)

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
        bayes.reaches = stretch_stack.reaches
        bayes.dist_out = stretch_stack.dist_out
        bayes.time_id = stretch_stack.time_id
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
        bayes.time_id = stretch_stack.time_id
        # get the mean and cov of the stacked wse and width
        #N = len(wse_along_stats.reference)
        # create the stacked mean
        mn = np.concatenate([
            wse_along_stats.reference, width_along_stats.reference
            ])
        wse_cov = rivscale.reconstruct.exponential_cov(
            stretch_stack['dist_out'],
            char_length_tau=wse_along_stats.char_length_tau,
            prior_unc_alpha=wse_along_stats.prior_unc_alpha)
        width_cov = rivscale.reconstruct.exponential_cov(
            stretch_stack['dist_out'],
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
        return bayes_wse, bayes_width, bayes_wse_width_post_cov

class HeightWidthModel(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding stretch_average estimates (over all nodes) of wse
            and width derived from 2D multitemporal data
            """)}],
        ['width_err',{'dtype':'float', 'value':-1}],
        ['wse_err',{'dtype':'float', 'value':-1}],
        ['count',{'dtype':'int', 'value':0}],

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
            width_anom=True,
            sigma_n=50):
        height_width = cls()
        if width_along_stats is None:
            ptile_list = [5, 25, 32, 50, 68, 75, 95]
        else:
            ptile_list = width_along_stats.percentile_list
        # get percntiles of measured reach data
        #Pm = np.nanpercentile(width_stretch_avg.mean, ptile_list)
        wse = wse_stretch_avg.mean
        width = width_stretch_avg.mean
        if wse_anom:
            wse = wse - wse_stretch_avg.mean_reference 
        if width_anom:
            width = width - width_stretch_avg.mean_reference
        # check that wse and width are all valid over the same places
        good_msk = np.logical_and(np.isfinite(wse), np.isfinite(width))
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
        height_width.width_err = np.sqrt(np.nanmean(dw**2))
        height_width.wse_err = np.sqrt(np.nanmean(dh**2))
        height_width.count = np.sum(good_msk)
        # TODO: bias adjust so mean(dw)=0, mean(dh)=0, and RMSE=STD etc
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

    def plot(
            self,
            wse_data=None,
            width_data=None,
            outdir=None,
            show=False,
            title_tag=None):
        plt.figure()
        if (wse_data is not None) and (
                width_data is not None):
            label = None
            if isinstance(wse_data, np.ndarray):
                d_width = width_data.copy()
                d_wse = wse_data.copy()
            else:
                # assume it is a stretch average
                if title_tag is None:
                    label='stretch average'
                d_width = width_data.mean - width_data.mean_reference
                d_wse = wse_data.mean - wse_data.mean_reference
            plt.plot(d_width, d_wse, 'o', label=label)
        plt.plot(self.width_coords, self.wse_coords,'-k', linewidth=2, label='model fit')
        plt.xlabel('$\Delta$ width')
        plt.ylabel('$\Delta$ wse')
        plt.legend()
        plt.grid()
        if title_tag is not None:
            plt.title(title_tag)
        # TODO: write to file if commanded
        #if outdir is not None:
        if show:
            plt.show()

