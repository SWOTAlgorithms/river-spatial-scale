'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author (s): Brent Williams

Container for along-stretch statistics, reference profile, and along-river
spatial scale parameter estimates

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
        ['node_length', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['along_dist', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['cross_track', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['local_node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['p_lat', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['p_lon', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['reference', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['std', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['count', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['percentiles', odict([['dimensions', DIMENSIONS_PCNT]])],
        ['percentile_list', odict([['dimensions', odict([['num_percentiles', 0]])]])],
    ])


    def plot(self, outdir=None, show=False, title_tag=''):
        rivscale.plot.plot_stretch_stats(
            self,
            x_key='along_dist',#'dist_out',
            outdir=outdir,
            show=show,
            title_tag=title_tag)

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
        in_keys = stretch_stack.variables.keys()
        this_keys = stats.VARIABLES.keys()
        common_keys = list(set(in_keys) & set(this_keys))
        for key in common_keys:
            if key != 'cross_track':
                stats[key] = stretch_stack[key].copy()
        #breakpoint()
        #stats.reaches = stretch_stack.reaches.copy()
        #stats.dist_out = stretch_stack.dist_out.copy()
        #stats.along_dist = stretch_stack.along_dist.copy()
        #stats.node_length = stretch_stack.node_length.copy()
        #stats.node_id = stretch_stack.node_id.copy()
        #stats.local_node_id = stretch_stack.local_node_id.copy()
        
        # get the reference profile
        stats.reference = rivscale.estimate.get_med_profile(
            stretch_stack[signal_key],
            stretch_stack['dist_out'],#stretch_stack['along_dist'],
            kernel_size = kernel_size)
        # TODO: get the spatial covariance estimate from data
        if char_length_tau is not None:
            stats.char_length_tau = char_length_tau
        if prior_unc_alpha is not None:
            stats.prior_unc_alpha = prior_unc_alpha
        # get the stats
        stats.cross_track = np.nanmean(stretch_stack['cross_track'], axis=1)
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
            ptiles[:,k] = np.nanpercentile(
                stretch_stack[signal_key], ptile, axis=1)
        stats.percentiles = np.array(ptiles)
        stats.percentile_list = np.array(percentiles)
        return stats

    @classmethod
    def from_pekel_df(
            cls,
            cfg,
            df_list,
            reaches,
            name,
            in_stats):
        """
            #smooth_size=None,
            #med_kernel_size=11):
        """
        # handle optional variables
        if 'width_smooth_size' not in cfg.keys():
            cfg['width_smooth_size'] = 'None'
        if 'mid_kernel_size' not in cfg.keys():
            cfg['mid_kernel_size'] = '11'
        #
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
        # optionally smooth the percentile width estimates
        # to reduce the wedging issues
        if cfg['width_smooth_size'] is not None:
            #breakpoint()
            d['percentiles'] = rivscale.estimate.smooth_widths(
                d['percentiles'].T, size=cfg['width_smooth_size']).T
        for key in d.keys():
            if key == 'percentiles':
                stats[key] = d[key].T
            else:
                stats[key] = d[key]
        # put the 50%ile in as the reference
        msk = stats.percentile_list==50
        ref = stats.percentiles[:,msk].squeeze()
        ref_med = ref.copy()
        if cfg['med_kernel_size'] is not None:
            ref_med = scipy.ndimage.median_filter(
                ref, size=cfg['med_kernel_size'], mode='nearest')
        # median filter the ref profile
        # TODO: maybe should do mean filter?
        if np.sum(msk)>0:
            stats.reference = ref_med
        if len(stats.percentile_list)<2:
            # dont create 1d version
            return None
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
        stats.stretch_name = self.stretch_name
        stats.reaches = np.array([int(reach_id),])
        stats.signal_key = self.signal_key
        stats.percentile_list = self.percentile_list
        keys = set(self.variables.keys()) - set(
            ['reaches', 'percentiles','percentile_list'])
        for key in keys:
            stats[key] = self[key][mask]#mask[0]:mask[-1]]
        #breakpoint()
        stats['percentiles'] = self['percentiles'][mask,:]#mask[0]:mask[-1],:]
        return stats

    def to_dataframe(self,
            variables=['reference', 'count', 'mean', 'std', 'percentiles'],
            df_in=None,
            crop_to_reach=True):
        """
        Grab standard variables and output as pandas dataframe while prepending
        the signal_key to the desired variables (and handling all percentiles).
        By default we will crop to reach, as we intend to use this to produce
        extra variables in a SWORD-like database 
        """
        if crop_to_reach:
            this = self.crop_to_reach()
        else:
            this = self
        #
        dic = {}
        dic['node_id'] = this['node_id']
        for key in variables:
            if key == 'percentiles':
                for k,ptile in enumerate(this['percentile_list']):
                    out_key = f'{this.signal_key}_{ptile}_percentile'
                    dic[out_key] = this[key][:,k]
            else:
                out_key = f'{this.signal_key}_{key}'
                dic[out_key] = this[key]
        #breakpoint()
        df = pd.DataFrame(dic).sort_values(by='node_id')
        return df

