'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author (s): Brent Williams

Container for estimates of river profiles binned by flow state (e.g., stretch-average wse bins)

Mimics the product class from RiverObs and the SWOT project "python" repo
'''
from collections import OrderedDict as odict
from rivscale.misc import textjoin
from rivscale.products.constants import DIMENSIONS_FLOW
        #DIMENSIONS_ALL, DIMENSIONS_SAVG, DIMENSIONS_ALONG, DIMENSIONS_2D,
        #DIMENSIONS_PCNT, DIMENSIONS_PCNT2, DIMENSIONS_COV, DIMENSIONS_POSTCOV)

from SWOTWater.products.product import Product, ProductTesterMixIn
from SWOTWater.products.constants import FILL_VALUES

import rivscale.estimate
import rivscale.plot
import matplotlib.pyplot as plt
import scipy.interpolate
#import pandas as pd
import os.path
import numpy as np

#import rivscale.products.stretch_average
#from scipy.stats import spearmanr

import rivscale.special

class FlowStateModel(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding node-level estimates (averaged over flow-state bins)
            of wse and width derived from 2D multitemporal data
            """)}],
        ['stretch_name',{'dtype':'str', 'value': textjoin("""
            Name given to this stretch instance (e.g., center reach
            or river name)
            """)}],
        ['oversamp_factor',{'dtype':'float', 'value':-1}],
        ['bin_width',{'dtype':'float', 'value':-1}],
        ['kind',{'dtype':'str', 'value':''}],# e.g., from wse_stretch_avg
        ])
    DIMENSIONS = DIMENSIONS_FLOW
    VARIABLES = odict([
        ['node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['along_dist', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['bin_centers', odict([['dimensions', odict([['num_states', 0]])]])],
        ['counts', odict([['dimensions', DIMENSIONS_FLOW]])],
        ['bin_profiles', odict([['dimensions', DIMENSIONS_FLOW]])],
        ['wse_profiles', odict([['dimensions', DIMENSIONS_FLOW]])],
        ['width_profiles', odict([['dimensions', DIMENSIONS_FLOW]])],
        ['bin_iqrs', odict([['dimensions', DIMENSIONS_FLOW]])],
        ['wse_iqrs', odict([['dimensions', DIMENSIONS_FLOW]])],
        ['width_iqrs', odict([['dimensions', DIMENSIONS_FLOW]])],
        
        # TODO: mean/std etc?
    ])

    @classmethod
    def from_objects(
            cls,
            stretch_stack,
            wse_along_stats,
            wse_stretch_avg=None,
            stretch_name=None,
            node_id=None,
            along_dist=None,
            bin_width = 0.5,
            oversamp_factor=2
            ):
        if wse_stretch_avg is not None:
            bin_var = np.zeros(np.shape(stretch_stack.time_id)) + np.nan
            for k, time_i in enumerate(wse_stretch_avg.time_id):
                bin_var[stretch_stack.time_id==time_i] = wse_stretch_avg.mean[k]
            #
            kind = 'wse_stretch_avg'
        else:
            bin_var = stretch_stack.wse.copy()
            kind = 'stack_wse'
        #
        if stretch_name is None:
            stretch_name = stretch_stack.stretch_name
        #
        return cls.from_arrays(
            bin_var,
            stretch_stack.wse,
            stretch_stack.width,
            node_id=stretch_stack.node_id,
            along_dist=stretch_stack.along_dist,
            kind=kind,
            stretch_name=stretch_name,
            bin_width=bin_width,
            oversamp_factor=oversamp_factor
            )

    @classmethod
    def from_arrays(
            cls,
            bin_var,
            wse,
            width,
            kind, # 'wse_stretch_avg', or 'stack_wse'
            node_id=None,
            along_dist=None,
            stretch_name=None,
            bin_width = 0.5,
            oversamp_factor=2
            ):
        # first do some sanity checks
        if len(bin_var[np.isfinite(bin_var)]) < 2:
            return None
        #
        flow_state = cls()
        flow_state.bin_width = bin_width
        if stretch_name is not None:
            flow_state.stretch_name = stretch_name
        if node_id is not None:
            flow_state.node_id = node_id
        if along_dist is not None:
            flow_state.along_dist = along_dist
        # call the binnner
        bbins, bin_count, p_list, bin_stats, width_bin_stats, wse_bin_stats = \
            rivscale.special.var_binned_node_stats(
                bin_var, width, wse,
                bin_width=bin_width,
                oversamp_factor=oversamp_factor,
                percentile_list=[25,50,75])#only do 50th %tile
        flow_state.wse_profiles = np.squeeze(wse_bin_stats[1])
        flow_state.width_profiles = np.squeeze(width_bin_stats[1])
        flow_state.bin_profiles = np.squeeze(bin_stats[1])
        flow_state.counts = bin_count
        flow_state.wse_iqrs = np.squeeze(wse_bin_stats[2] - wse_bin_stats[0])
        flow_state.width_iqrs = np.squeeze(width_bin_stats[2] - width_bin_stats[0])
        flow_state.bin_iqrs = np.squeeze(bin_stats[2] - bin_stats[2])
        # change bins into bin centers?
        #flow_state.bin_centers = 
        return flow_state
    
    def isotonic_wse(self):
        # apply an isotonic regression on wse
        self.wse_profiles = rivscale.special.isotonic_regression_stack(
            self.wse_profiles,
            self.along_dist)

    # TODO add along-river smoothing/intepolation over holes?
    # TODO: add inter-state sampling/interpolation (e.g., given
    #       a wse_stretch_avg, return a mean wse and width profile for the state)
    def plot(self, outdir=None, title_tag='', show=False):
        # TODO: refine these
        # count
        plt.figure()
        plt.plot(self.along_dist, self.counts)
        plt.ylabel('count')
        plt.xlabel('along_dist (m)')
        plt.grid()
        # wse
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.along_dist, self.wse_profiles)
        plt.ylabel('wse (m)')
        plt.grid()
        plt.subplot(2,1,2)
        plt.plot(self.along_dist, self.wse_iqrs)
        plt.ylabel('wse IQR (m)')
        plt.xlabel('along_dist (m)')
        plt.grid()
        # width
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.along_dist, self.width_profiles)
        plt.ylabel('width (m)')
        plt.grid()
        plt.subplot(2,1,2)
        plt.plot(self.along_dist, self.width_iqrs)
        plt.ylabel('width IQR (m)')
        plt.xlabel('along_dist (m)')
        plt.grid()
        #
        # width
        # do a 3d plot
        along_dist2d = np.broadcast_to(self.along_dist, np.shape(self.wse_profiles.T)).T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for k in range(len(self.wse_profiles[0,:])):
            ax.plot(along_dist2d[:,k], self.width_profiles[:,k], self.wse_profiles[:,k])
        ax.set_xlabel('along_dist (m)')
        ax.set_ylabel('width (m)')
        ax.set_zlabel('wse (m)')
        if show:
            plt.show()






