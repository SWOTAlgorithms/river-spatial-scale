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

    def plot(self, outdir=None):
        rivscale.plot.plot_stretch_stack(
            self,
            x_key='dist_out',
            y_keys=['wse','width'],
            outdir=outdir)
        rivscale.plot.plot_stretch_stack(
            self,
            x_key='time_id',
            y_keys=['wse','width'],
            outdir=outdir,
            marker='o')
        if outdir is None:
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
            stretch_data,
            signal_key,
            percentiles=[5, 25, 32, 50, 68, 75, 95],
            kernel_size=35):
        stats = cls()
        stats.signal_key = signal_key
        # copy over common items
        stats.stretch_name = stretch_data.stretch_name
        stats.reaches = stretch_data.reaches.copy()
        stats.dist_out = stretch_data.dist_out.copy()
        stats.node_id = stretch_data.node_id.copy()
        stats.local_node_id = stretch_data.local_node_id.copy()
        # get the reference profile
        stats.reference = rivscale.estimate.get_med_profile(
            stretch_data[signal_key],
            stretch_data['dist_out'],
            kernel_size = kernel_size)
        # TODO: get the spatial covariance estimate
         
        # get the stats
        stats.mean = np.nanmean(stretch_data[signal_key], axis=1)
        stats.std = np.nanstd(stretch_data[signal_key], axis=1)
        mask = np.zeros(np.shape(stretch_data[signal_key]))
        mask[np.isfinite(stretch_data[signal_key])] = 1
        stats.count = np.nansum(mask, axis=1)
        stats.percentiles_list = percentiles
        ptiles = np.zeros((
            len(stats.mean),
            len(percentiles),
            )) + np.nan
        for k, ptile in enumerate(percentiles):
            ptiles[:,k] = np.nanpercentile(stretch_data[signal_key], ptile, axis=1)
        stats.percentiles = ptiles
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
            d = {
                #'node_id':node_id,
                'percentiles':[],
                'percentile_list':[],
                #'dist_out':dist_out}
                }
            # Pekel occurrence threshold is packed in cycle field
            ptiles = np.sort(100-np.unique(df.cycle))
            for ptile in ptiles:
                this_df = df[df.cycle==100-ptile]
                #if d['dist_out'] is None:
                #    d['dist_out'] = dist_out
                #if d['node_id'] is None:
                #    d['node_id'] = node_id
                w = np.zeros(np.shape(node_id))+np.nan
                this_node_id = np.array(this_df.node_id)
                for wid, nid in zip(this_df.width, this_df.node_id):
                    w[node_id==nid] = wid
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
                #if d['dist_out'] is None:
                d['dist_out'] = this_dist_out
                #if d['percentile_list'] is None:# assumes same percentile list for every df
                d['percentile_list'] = np.array(this_d['percentile_list'])
                d['percentiles'] = np.array(this_d['percentiles'])
            else:
                d['node_id'] = np.append(d['node_id'], this_node_id)
                d['dist_out'] = np.append(d['dist_out'], this_dist_out)
                d['percentiles'] = np.append(d['percentiles'], np.array(this_d['percentiles']), axis=1)
        for key in d.keys():
            if key == 'percentiles':
                stats[key] = d[key].T
            else:
                stats[key] = d[key]
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
        ['reference_mean',{'dtype':'float', 'value':0.0}],
        ])
    DIMENSIONS = DIMENSIONS_ALL
    VARIABLES = odict([
        ['reaches', odict([['dimensions', odict([['num_reaches', 0]])]])],
        ['dist_out', odict([['dimensions', odict([['num_times', 0]])]])],
        ['time_id', odict([['dimensions', odict([['num_times', 0]])]])],
        ['cycle_id', odict([['dimensions', odict([['num_times', 0]])]])],
        ['mean', odict([['dimensions', odict([['num_times', 0]])]])],
        ['std', odict([['dimensions', odict([['num_times', 0]])]])],
        ['count', odict([['dimensions', odict([['num_times', 0]])]])],
        ['percentiles', odict([['dimensions', DIMENSIONS_PCNT]])],
        ['percentile_list', odict([['dimensions', odict([['num_percentiles', 0]])]])],
        #['slope', odict([['dimensions', odict([['num_times', 0]])]])],
        #['slope_reference', odict([['dimensions', odict([['num_times', 0]])]])],
    ])

    @classmethod
    def from_StretchStack(
            cls,
            stretch_data,
            signal_key,
            percentiles=[5, 25, 32, 50, 68, 75, 95],
            reference=None):
        stats = cls()
        # copy common things
        stats.reaches = stretch_data.reaches.copy()
        stats.time_id = stretch_data.time_id.copy()
        stats.cycle_id = stretch_data.cycle_id.copy()
        # get stats
        stats.dist_out = np.nanmean(stretch_data.dist_out, axis=0)
        if reference is None:
            reference = np.zeros_like(stretch_data[signal_key][:,0])
        elif isinstance(reference, AlongStretchStats):
            reference = reference.reference.copy()
        reference = np.broadcast_to(
            reference, np.shape(stretch_data[signal_key].T)).T
        stats.reference_mean = np.nanmean(np.array(reference), axis=0)
        stats.mean = np.nanmean(stretch_data[signal_key]-reference, axis=0)\
            + stats.reference_mean
        stats.std = np.nanstd(stretch_data[signal_key]-reference, axis=0)
        mask = np.zeros(np.shape(stretch_data[signal_key]))
        mask[np.isfinite(stretch_data[signal_key]-reference)] = 1
        stats.count = np.nansum(mask, axis=0)
        stats.percentiles_list = percentiles
        ptiles = np.zeros((
            len(stats.mean),
            len(percentiles),
            )) + np.nan
        for k, ptile in enumerate(percentiles):
            ptiles[:,k] = np.nanpercentile(
                stretch_data[signal_key]-reference, ptile, axis=0)\
                    + stats.reference_mean
        stats.percentiles = ptiles
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
        ['bayes_wse_width_post_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['signal_mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        #['signal_cov', odict([['dimensions', odict([['num_nodes', 0]])]])],
        #['wse_cov', odict([['dimensions', DIMENSIONS_COV]])],
        #['width_cov', odict([['dimensions', DIMENSIONS_COV]])],
        #['wse_width_cov', odict([['dimensions', DIMENSIONS_COV]])],
        ['signal_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        #['width_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        #['wse_width_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
    ])


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
        ['hw_params', odict([['dimensions', odict([['num_hw_params', 0]])]])],
        ['hw_params_err', odict([['dimensions', odict([['num_hw_params', 0]])]])],
    ])
    #for name, reference in VARIABLES.items():
    #    reference['dimensions'] = DIMENSIONS




