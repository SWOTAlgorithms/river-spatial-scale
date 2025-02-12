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

class RiverStretchData(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding 2D multitemporal data and Bayes reconstruction
            processing parameters and results
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
        ['bayes_wse', odict([['dimensions', DIMENSIONS_2D]])],
        ['bayes_width', odict([['dimensions', DIMENSIONS_2D]])],
        ['bayes_wse_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['bayes_width_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['bayes_wse_post_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['bayes_width_post_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['bayes_wse_width_post_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['wse_reference', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_reference', odict([['dimensions', odict([['num_nodes', 0]])]])],
        #['wse_cov', odict([['dimensions', DIMENSIONS_COV]])],
        #['width_cov', odict([['dimensions', DIMENSIONS_COV]])],
        #['wse_width_cov', odict([['dimensions', DIMENSIONS_COV]])],
        ['wse_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['width_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['wse_width_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['wse_mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['dark_frac_mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_std', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_std', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['dark_frac_std', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_count', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_count', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['dark_frac_count', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_percentiles', odict([['dimensions', DIMENSIONS_PCNT]])], 
        ['width_percentiles', odict([['dimensions', DIMENSIONS_PCNT]])],
        ['dark_frac_percentiles', odict([['dimensions', DIMENSIONS_PCNT]])],
        ['percentiles', odict([['dimensions', odict([['num_percentiles', 0]])]])],
        ['dark_prob', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['stretch_wse_mean', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_wse_median', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_wse_std', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_wse_count', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_width_mean', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_width_median', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_width_std', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_width_count', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_wse_mean', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_wse_median', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_wse_std', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_wse_count', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_width_mean', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_width_median', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_width_std', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_width_count', odict([['dimensions', odict([['num_times', 0]])]])],
        ['hw_params', odict([['dimensions', odict([['num_hw_params', 0]])]])],
        ['hw_params_err', odict([['dimensions', odict([['num_hw_params', 0]])]])],
    ])
    #for name, reference in VARIABLES.items():
    #    reference['dimensions'] = DIMENSIONS


class StretchData(Product):
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

class RiverStretchStats(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding along-river statistics and spatial-scale covariance
            estimates derived from 2D multitemporal data
            """)}],
        ])
    DIMENSIONS = DIMENSIONS_ALL
    VARIABLES = odict([
        ['wse_reference', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_reference', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['width_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['wse_width_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['wse_mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['dark_frac_mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_std', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_std', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['dark_frac_std', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_count', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_count', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['dark_frac_count', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_percentiles', odict([['dimensions', DIMENSIONS_PCNT]])],
        ['width_percentiles', odict([['dimensions', DIMENSIONS_PCNT]])],
        ['dark_frac_percentiles', odict([['dimensions', DIMENSIONS_PCNT]])],
        ['percentiles', odict([['dimensions', odict([['num_percentiles', 0]])]])],
        ['dark_prob', odict([['dimensions', odict([['num_nodes', 0]])]])],
    ])

class AlongStretchStats(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding along-river statistics and spatial-scale covariance
            estimates derived from 2D multitemporal data
            """)}],
        ['signal_key',{'dtype':'str', 'value':'wse, width, or dark_frac'}],
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

    @classmethod
    def from_StretchData(
            cls,
            stretch_data,
            signal_key,
            percentiles=[5, 25, 32, 50, 68, 75, 95],
            kernel_size=35):
        stats = cls()
        stats.signal_key = signal_key
        # copy over common items
        #breakpoint()
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
        return stats

class RiverStretchAverage(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding stretch_average estimates (over all nodes) of wse
            and width derived from 2D multitemporal data
            """)}],
        ])
    DIMENSIONS = DIMENSIONS_SAVG
    VARIABLES = odict([
        ['reaches', odict([['dimensions', odict([['num_reaches', 0]])]])],
        ['dist_out', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_wse_mean', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_wse_median', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_wse_std', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_wse_count', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_width_mean', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_width_median', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_width_std', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_width_count', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_wse_mean', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_wse_median', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_wse_std', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_wse_count', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_width_mean', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_width_median', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_width_std', odict([['dimensions', odict([['num_times', 0]])]])],
        ['stretch_bayes_width_count', odict([['dimensions', odict([['num_times', 0]])]])],
    ])

class StretchAverageStats(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding stretch_average estimates (over all nodes) of wse
            and width derived from 2D multitemporal data
            """)}],
        ['signal_key',{'dtype':'str', 'value':'wse, width, or dark_frac'}],
        ['reference_mean',{'dtype':'float', 'value':0.0}]
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
    def from_StretchData(
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




