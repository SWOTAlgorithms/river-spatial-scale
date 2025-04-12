'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author (s): Brent Williams

Container for the stretch-average data (time series data like the river reach
but for the mult-reach connected stretch).

Mimics the product class from RiverObs and the SWOT project "python" repo
'''
from collections import OrderedDict as odict
from rivscale.misc import textjoin
from rivscale.products.constants import (
        DIMENSIONS_ALL, DIMENSIONS_SAVG, DIMENSIONS_ALONG, DIMENSIONS_2D,
        DIMENSIONS_PCNT, DIMENSIONS_PCNT2, DIMENSIONS_COV, DIMENSIONS_POSTCOV)

from swot.product import Product, ProductTesterMixIn
from SWOTWater.products.constants import FILL_VALUES
from swot.lr.base_classes import AttrFillerMixIn

import rivscale.estimate
import rivscale.plot
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
import os.path
import numpy as np

import rivscale.products.bayes_data
import rivscale.products.along_stretch

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
        #['cycle_id', odict([['dimensions', odict([['num_times', 0]])]])],
        ['granule_id', odict([['dimensions', odict([['num_times', 0]])]])],
        ['mean', odict([['dimensions', odict([['num_times', 0]])]])],
        ['mean_reference', odict([['dimensions', odict([['num_times', 0]])]])],
        ['std', odict([['dimensions', odict([['num_times', 0]])]])],
        ['uncert', odict([['dimensions', odict([['num_times', 0]])]])],
        ['count', odict([['dimensions', odict([['num_times', 0]])]])],
        ['percentiles', odict([['dimensions', DIMENSIONS_PCNT]])],
        ['percentile_list', odict([['dimensions', odict([['num_percentiles', 0]])]])],
        ['slope', odict([['dimensions', odict([['num_times', 0]])]])],
        ['slope_reference', odict([['dimensions', odict([['num_times', 0]])]])],
    ])
    def plot(self, outdir=None, show=False, title_tag='', per_pass=False):
        rivscale.plot.plot_stretch_stats(
            self,
            x_key='time_id',
            outdir=outdir,
            title_tag=title_tag)
        if per_pass:
            # also plot the per-pass time-series wse and width
            rivscale.plot.plot_per_pass_time_series(
                self,
                outdir=outdir,
                title_tag=title_tag)
        if show:
            plt.show()

    @classmethod
    def from_reach_df(
            cls,
            df_in,
            reach,
            signal_key,
            percentiles=[5, 25, 32, 50, 68, 75, 95]):
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
        granule_id = []
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
            ####
            g_ids = []
            for cyc, pas, cont in zip(
                    this_df['cycle'], this_df['pass'], this_df['continent']):
                g_id = '{:03d}_{:03d}_{}'.format(int(cyc), int(pas), cont)
                g_ids.append(g_id)
            #breakpoint()
            ugids = np.unique(g_ids)
            ugid='000_000_00'
            if len(ugids)>0:
                ugid = ugids[0]
            granule_id.append(ugid)
            ####
        #breakpoint()
        Other = np.array(other).squeeze()
        stats.time_id = np.array(time_id).squeeze()
        stats.granule_id = np.array(granule_id).squeeze()
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
            stats.slope = np.array(slope).squeeze()
            stats.slope_reference = np.nanmean(
                slope) * np.ones_like(stats.mean)
        # create percentiles
        stats.percentile_list = np.array(percentiles)
        ptiles = np.zeros((
            len(stats.mean),
            len(percentiles),
            )) + np.nan
        for k, ptile in enumerate(percentiles):
            ptiles[:,k] = np.nanpercentile(
                stats.mean - stats.mean_reference,
                ptile, axis=0) + stats.mean_reference
        stats.percentiles = ptiles
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
        in_keys = stretch_stack.variables.keys()
        this_keys = stats.VARIABLES.keys()
        common_keys = list(set(in_keys) & set(this_keys))
        for key in common_keys:
            if key != 'dist_out':
                stats[key] = stretch_stack[key].copy()
        #stats.reaches = stretch_stack.reaches.copy()
        #stats.time_id = stretch_stack.time_id.copy()
        #stats.cycle_id = stretch_stack.cycle_id.copy()
        #stats.granule_id = stretch_stack.granule_id.copy()
        # get stats
        signal = stretch_stack[signal_key]
        signal_u = stretch_stack[signal_key+'_u']
        first_node = 0
        last_node = -1
        if along_stats is not None:
            if ('bayes' in average_method) or ('bayes' in slope_method):
                bayes = rivscale.products.bayes_data.BayesData.simple(
                    stretch_stack,
                    along_stats,
                    signal_key)# use bayes parameters in along_stats
                #ref = along_stats.reference.copy()
                #ref2 = np.broadcast_to(ref, np.shape(signal.T)).T
                #breakpoint()
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
        elif isinstance(along_stats, rivscale.products.along_stretch.AlongStretchStats):
            ref = along_stats.reference.copy()
        reference = np.broadcast_to(
            ref, np.shape(signal.T)).T
        #stats.reference_mean = np.nanmean(
        #        np.array(reference) * weight, axis=0) / mean_weight
        # always do simple mean of reference
        #stats.mean_reference = np.nanmean(np.array(reference), axis=0)
        # weighted mean of ref too
        stats.mean_reference = np.nansum(# TODO: double check if use widow_valid
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
        # TODO: handle count when doing Bayes?
        #       maybe also/just compute an estimate of the error
        #       of the stretch average estimate?
        stats.count = np.nansum(mask, axis=0)
        if 'bayes' in average_method:
            # put in the bayes uncert instead of sample std
            post_cov = bayes.signal_post_cov
            #breakpoint()
            std = []
            N = len(stretch_stack.node_id)
            One = np.ones((N,1)).squeeze()
            for k,mn in enumerate(stats.mean):
                sig2 = 1/N**2 * One @ post_cov[:,:,k] @ One.T
                std.append(np.sqrt(sig2))
            stats.uncert = np.array(std)
        else:
            stats.uncert = stats.std / np.sqrt(stats.count)
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







