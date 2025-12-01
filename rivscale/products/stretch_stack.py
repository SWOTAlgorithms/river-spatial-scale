'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author (s): Brent Williams

Container for stretch stack, a 2D object in time and along river of ordered
node-level river measurements (e.g., from SWOT RiverSP data), with methods
to filter manipulate, and plot the data.

Mimics the product class from RiverObs and the SWOT project "python" repo
'''
from collections import OrderedDict as odict
from rivscale.misc import textjoin
from rivscale.products.constants import (
        DIMENSIONS_NTR,  DIMENSIONS_2D)

from SWOTWater.products.product import Product, ProductTesterMixIn
from SWOTWater.products.constants import FILL_VALUES

import rivscale.estimate
import rivscale.plot
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
import os.path
import numpy as np

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
    DIMENSIONS = DIMENSIONS_NTR
    VARIABLES = odict([
        ['reaches', odict([['dimensions', odict([['num_reaches', 0]])]])],
        ['dist_out', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['node_length', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['along_dist', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['local_node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['p_lat', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['p_lon', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['time_id', odict([['dimensions', odict([['num_times', 0]])]])],
        ['granule_id', odict([['dimensions', odict([['num_times', 0]])]])],
        #['swot_time', odict([['dimensions', odict([['num_times', 0]])]])],
        ['date_hour', odict([['dimensions', odict([['num_times', 0]])]])],
        ['cross_track', odict([['dimensions', DIMENSIONS_2D]])],
        ['wse', odict([['dimensions', DIMENSIONS_2D]])],
        ['width', odict([['dimensions', DIMENSIONS_2D]])],
        ['area_total', odict([['dimensions', DIMENSIONS_2D]])],
        ['wse_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['width_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['area_tot_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['node_q_b', odict([['dimensions', DIMENSIONS_2D]])],
        ['dark_frac', odict([['dimensions', DIMENSIONS_2D]])],
        ['sig0 (dB)', odict([['dimensions', DIMENSIONS_2D]])],
        ['flow_angle', odict([['dimensions', DIMENSIONS_2D]])],
        ['layovr_val', odict([['dimensions', DIMENSIONS_2D]])],
        ['n_good_pix', odict([['dimensions', DIMENSIONS_2D]])],
        ['width_correction', odict([['dimensions', DIMENSIONS_2D]])],
    ])
    # TODO: should we also keep ice flag, etc...?

    def plot(
            self,
            wse_reference=None,
            width_reference=None,
            x_key='along_dist',#'dist_out', # or 'time_id'
            outdir=None,
            show=False,
            title_tag=None,
            bits_to_plot=[],
            #bits_to_plot=[0,1,2,3,4,7,9,10,11,18,19,22]
            #bits_to_plot=[0,1,2,3,4,7,9,10,11,13,14,18,19,22,23,24,25,26,27,28]
            extra_vars=False
            ):
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
            # plot the 2D wse and wse_anom
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='wse',
                y_reference=wse_reference,
                y_anom=False,
                outdir=outdir,
                title_tag=title_tag)
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='wse',
                y_reference=wse_reference,
                y_anom=True,
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
            # plot the 2D width and width anom
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='width',
                y_reference=width_reference,
                y_anom=False,
                outdir=outdir,
                title_tag=title_tag)
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='width',
                y_reference=width_reference,
                y_anom=True,
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
            # plot the 2D height and width
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='wse',
                y_reference=None,
                y_anom=False,
                outdir=outdir,
                title_tag=title_tag)
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='width',
                y_reference=None,
                y_anom=False,
                outdir=outdir,
                title_tag=title_tag)
        if np.nansum(self['dark_frac'])>0 and extra_vars:
            # also plot the dark frac 1D and 2D
            """
            rivscale.plot.plot_stretch_stack(
                self,
                x_key=x_key,
                y_keys=['dark_frac',],
                y_reference=[None,],
                y_anom=[False,],
                outdir=outdir,
                title_tag=title_tag)
            """
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='dark_frac',
                y_reference=None,
                y_anom=False,
                outdir=outdir,
                title_tag=title_tag)
        if np.nansum(np.isfinite(self['sig0 (dB)']))>0 and extra_vars:
            # also plot the sig0 2D
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='sig0 (dB)',
                y_reference=None,
                y_anom=False,
                outdir=outdir,
                title_tag=title_tag)
        if np.nansum(np.isfinite(self['flow_angle']))>0 and extra_vars:
            # also plot the flow_dir 2D
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='flow_angle',
                y_reference=None,
                y_anom=False,
                outdir=outdir,
                title_tag=title_tag)
        if np.nansum(np.isfinite(self['layovr_val']))>0 and extra_vars:
            # also plot the flow_dir 2D
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='layovr_val',
                y_reference=None,
                y_anom=False,
                outdir=outdir,
                title_tag=title_tag)
        if np.nansum(np.isfinite(self['n_good_pix']))>0 and extra_vars:
            # also plot the flow_dir 2D
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='n_good_pix',
                y_reference=None,
                y_anom=False,
                outdir=outdir,
                title_tag=title_tag)
        # make a 2D plot for commanded qual bits?
        for bit in bits_to_plot:
            rivscale.plot.plot_2D_stretch_stack(
                self,
                y_key='node_q_b',
                y_reference=None,
                y_anom=False,
                outdir=outdir,
                title_tag=title_tag,
                bit=bit)
        if show:
            plt.show()

    def smooth_widths(self, size=11):
        self.width = rivscale.estimate.smooth_widths(self.width, size=size)
        # TODO: maybe should also update width_u?
        # TODO: maybe should take into account dark water

    def filter_dark_water(self, key='wse', dark_thresh=0.8):
        """
        filter out dark water in variable 'key' by setting to nan
        """
        self[key][self.dark_frac>dark_thresh] = np.nan

    def filter_node_outliers(
            self,
            along_stats=None,
            key='wse',# or width etc
            Delta2=False,
            use_ptiles=False,
            IQR_scale=5.0,
            plot=False,
            title_tag='',
            outdir=None):
        """
        This method filters the StretchStack object for the 'key'
        variable by replacing the values with nans. The method applies
        an Inter-Quartile Range (IQR) filter on the data for each node
        using the multitemproal stack statistics.  Two IQR filters are
        applied, one based on the statistics for each node and one based
        on the statistics over the entire stretch. There are a few
        options depending on the optional inputs:

        Inputs
            along_stats: an AlongSStretchStats object.  If input the
                         'reference' field is used instead of computing
                         it using hte median of the data over time.

            key:         'wse' or 'width' (i.e., which variable to filter)

            Delta2:      if True, estimate a stretch average for each
                         time to also subtract off to produce an the
                         anomaly-anomaly (Delta2) before computing the
                         IQR stats.

            use_ptiles:  use the 25th and 75th percentile from the
                         along_stats object instead of computing them.

            IQR_scale:   postive float value indicating how many IQRs to
                         set rejection threshold

            plot:        make plots for debugging purposes

        TODO: should probably use masked arrays everywhere instead of
              relying on nans.
        TODO: probably should set a lower limit on the IQR based on the
              known uncertainty of the 'key' variable
        """
        # handle data
        arr0 = self[key].copy()
        if along_stats is None:
            # use the median over time as ref
            ref = np.nanmedian(arr0, axis=1)
        else:
            # use ref from along_stats
            ref = along_stats.reference
        ref2 = np.broadcast_to(ref, np.shape(arr0.T)).T
        delta = arr0 - ref2
        delta_tag='$\Delta$'
        if Delta2:
            # get the anomaly-anomaly
            delta_bar = np.nanmean(delta, axis=0)
            delta_bar = np.nanmedian(delta, axis=0)
            delta_bar2 = np.broadcast_to(delta_bar, np.shape(arr0))
            ref2 = ref2 + delta_bar2
            use_ptiles=False
            delta_tag = '$\Delta^2$'
            #delta2 = arr0 - dmean2
            #arr0 = delta2
        #breakpoint()
        if use_ptiles:
            p = along_stats.percentile_list
            p25 = along_stats.percentiles[:, p==25].squeeze()
            p75 = along_stats.percentiles[:, p==75].squeeze()
        else:
            # compute the percentiles
            p25 = np.nanpercentile(arr0-ref2, 25, axis=1)
            p75 = np.nanpercentile(arr0-ref2, 75, axis=1)
        #p = along_stats.percentile_list
        #p25 = along_stats.percentiles[:,p==25].squeeze()
        #p75 = along_stats.percentiles[:,p==75].squeeze()
        IQR = p75 - p25
        IQR2 = np.broadcast_to(IQR, np.shape(arr0.T)).T
        # first filter global
        #p25_g = np.median(p25 - ref)
        #p75_g = np.median(p75 - ref)
        #IQR_g = (p75_g - p25_g) + np.zeros_like(ref)
        IQR_g0 = np.nanmean(IQR[IQR>0])
        IQR_g = IQR_g0 + np.zeros_like(ref2)
        # for some reason Pekel sometimes give negative IQR? so hande it
        IQR[IQR<0] = IQR_g0
        #breakpoint()
        arr1 = rivscale.filter.scaled_spread_outlier_rejector(
            arr0, ref2, IQR_g, IQR_scale)
        # now filter per-node
        arr = rivscale.filter.scaled_spread_outlier_rejector(
            arr1, ref2, IQR2, IQR_scale)
        self[key] = arr
        #breakpoint()
        if plot:
            outlier_mask = np.logical_and(
                np.isfinite(arr0),
                np.isnan(arr))
            x_label = 'along_dist (m)'#'dist_out (m)'
            x_data = self.along_dist#self.dist_out
            x_data2 = np.broadcast_to(
                x_data,
                np.shape(arr0.T)).T
            plt.figure(figsize=(10,10))
            plt.subplot(2,1,1)
            plt.plot(x_data2, arr0-ref2)
            plt.plot(
                x_data2[outlier_mask],
                arr0[outlier_mask]-ref2[outlier_mask],'x')
            plt.plot(
                x_data, #ref + \
                IQR_scale*IQR,'k', linewidth=2)
            plt.plot(
                x_data,
                -IQR_scale*IQR,'k', linewidth=2)
            plt.plot(
                x_data,
                IQR_scale*IQR_g,'g', linewidth=2)
            plt.plot(
                x_data,
                -IQR_scale*IQR_g,'g', linewidth=2)
            plt.grid()
            plt.xlabel(x_label)
            plt.ylabel(rivscale.plot.label_units(delta_tag+' '+key))
            yscale = IQR_scale*IQR_g[0,0]*2
            plt.ylim((-yscale, yscale))
            #
            # Also plot 2D plots
            #
            darr = arr0-ref2
            darr_out = np.zeros(np.shape(darr))
            lim = np.median(IQR_g * IQR_scale)
            darr_out[outlier_mask] = 1
            #plt.figure()
            plt.subplot(2,1,2)
            plt.imshow(darr_out.T,
                interpolation='none', aspect='auto', cmap='gray_r', clim=(0,1.1))
            plt.imshow(darr.T,
                interpolation='none', aspect='auto', cmap='jet', alpha=0.5,
                clim=(-lim,lim))
            plt.colorbar(label=rivscale.plot.label_units(delta_tag+' '+key))
            #plt.title(delta_tag+' '+key)
            plt.xlabel('node index')
            plt.ylabel('time index')
            #breakpoint()
            if title_tag != '':
                title_tag = title_tag + ' '
            subtitle = '(x and dark hue flagged as outliers)'
            title = '{} {} {} {}outliers'.format(
                self.stretch_name, delta_tag, key, title_tag)
            plt.suptitle(title+'\n'+subtitle)
            #plt.tight_layout()
            #breakpoint()
            if outdir is not None:
                # create output dir if not exist
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                #breakpoint()
                fname = title.replace(' ','_').replace(
                    '$\Delta$','delta').replace('$\Delta^2$','delta2')
                plt.savefig(os.path.join(outdir, fname), dpi=300)
                plt.close()

    def crop_to_reach(
            self,
            reach_id=None):
        stack = StretchStack()
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
        stack.stretch_name = self.stretch_name
        stack.reaches = np.array([int(reach_id),])
        # crop the along-river variables
        #breakpoint()
        keys = set(self.variables.keys()) - set(['reaches',])
        for key in keys:
            #print(key)
            dims = self.VARIABLES[key]['dimensions']
            if 'num_nodes' in dims.keys():
                stack[key] = self[key][mask]#mask[0]:mask[-1]]
            else:
                stack[key] = self[key]
        return stack

    def split_per_pass(self):
        gid = self.granule_id.copy()
        pid = np.array([g.split('_')[1] for g in gid])
        upid = np.unique(pid)
        #
        stacks = []
        for p in upid:
            this_stack = StretchStack()
            # copy the attribute
            this_stack.stretch_name = self.stretch_name
            # copy the variables
            for key in self.variables.keys():
                dims = self.VARIABLES[key]['dimensions']
                if 'num_times' in dims.keys():
                    if len(dims)>1:
                        this_stack[key] = self[key][:,pid==p]
                    else:
                        this_stack[key] = self[key][pid==p]
                else:
                    this_stack[key] = self[key]
            stacks.append(this_stack)
        return stacks

    def plot_per_pass(self, y_key='wse', outdir=None):
        title_tag = 'all passes'
        rivscale.plot.plot_2D_stretch_stack(
            self,
            y_key=y_key,
            y_reference=None,
            y_anom=False,
            outdir=outdir,
            title_tag=title_tag)
        stacks = self.split_per_pass()
        for stack in stacks:
            pid = stack.granule_id[0].split('_')[1]
            title_tag = f' pass {pid}'
            rivscale.plot.plot_2D_stretch_stack(
                stack,
                y_key=y_key,
                y_reference=None,
                y_anom=False,
                outdir=outdir,
                title_tag=title_tag)

    def bundle_adjust_per_pass_widths(self, apply_filter=True, cfg=None):
        """
        This method applies a width correction to each node in the stretch
        that is computed from the 'refrence' width (from width_along_stats)
        after splitting stretch_stack by pass
        
        Note that this does not actually modify the current object but creates
        and outputs a modified copy
        
        The optional input 'cfg' has config parameters suitable for calling
        the filter_stretch_stack and along_stats routines.
        """
        if cfg is None:
            cfg = {
            'wse_dark_thresh' : 0.8,
            'width_dark_thresh' :  0.3,
            'wse_outlier_scale' : 5.0,
            'width_outlier_scale' : 10,#2.0,
            'width_smooth_size' : None,
            'wse_ref_kernel_size' : 35,
            'width_ref_kernel_size' : None,
            'crop' : False
            }
        # first filter the stretch_stack for quality
        if apply_filter:
            stack, _ = rivscale.filter.filter_stretch_stack(
                cfg,
                self.copy(),
                wse_stats=None,
                width_stats=None,
                plot=False)
        else:
            stack = self.copy()
        # now split into separate passes
        stacks = stack.split_per_pass()
        # now estimate the correction for each pass for all nodes 
        wgt = []
        w_ref = []
        h_ref = []
        pid = []
        for this_stack in stacks:# loop over the pass-split stacks
            # run alongstats
            t_wse_stats, t_width_stats = \
                rivscale.estimate.process_along_stats(
                    cfg, this_stack)
            #
            ct = np.nanmedian(np.abs(this_stack.cross_track), axis=-1)
            # compute a weighting to estimate the bulk width adjustment
            # so that it favors the middle of the swath
            w = np.exp(-((ct - 45)/10)**2)
            wgt.append(w)
            w_ref.append(t_width_stats.reference)
            h_ref.append(t_wse_stats.reference)
            # also append the pass_id
            gid = this_stack.granule_id.copy()
            pid0 = np.array([g.split('_')[1] for g in gid])
            upid = np.unique(pid0)
            if len(upid)>1:
                print('Warning: multiple pass ids after spliting by pass???')
            pid.append(upid[0])
        # estimate the bulk/central width offset
        wgt = np.array(wgt)
        w_bulk = np.squeeze(
            np.nansum(wgt * np.array(w_ref), axis=0) / np.nansum(wgt, axis=0))
        # apply the buld offset and the correction for each pass
        w_corr = np.broadcast_to(w_bulk, np.shape(w_ref)) - w_ref
        # now expand correction to full stack
        gid = self.granule_id.copy()
        pid0 = np.array([g.split('_')[1] for g in gid])
        width_correction = np.zeros_like(self.width)
        for k,p in enumerate(pid): # loop over split-stack pid and populate 
            width_correction[:,pid0==p] = np.broadcast_to(
                w_corr[k,:], np.shape(width_correction[:,pid0==p].T)).T
        # now create a copy of the stretch_stack and adjust the width
        out_stack = self.copy()
        out_stack.width_correction = width_correction
        out_stack.width = out_stack.width + width_correction
        return out_stack

