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

#from scipy.optimize import minimize

from matplotlib.colors import LightSource
#from matplotlib import colormaps as cm
from matplotlib import cm

import scipy.ndimage

class HeightWidthModelArray(Product):
    """
    A more general container for height/width model in array form
    (e.g., a single object with parameter estimates for a while list
    of nodes as opposed to a list of height/width model objects)
    """
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
        ])
    DIMENSIONS = odict([['num_nodes',0],['num_hw_params',0]])
    VARIABLES = odict([
        ['width_coords', odict([['dimensions', odict([
            ['num_nodes', 0],
            ['num_hw_params', 0]]
            )]])],
        ['wse_coords', odict([['dimensions', odict([
            ['num_nodes', 0],
            ['num_hw_params', 0]]
            )]])],
        #
        ['along_dist', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_err', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_err', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['tot_err', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['count', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['spearman_r',odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['spearman_p_value',odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['ptile_list',odict([['dimensions', odict([['num_hw_params', 0]])]])],
    ])

    @classmethod
    def from_object(
            cls,
            obj,
            wse_reference=None,
            ptile_list = [5, 25, 32, 50, 68, 75, 95],
            snapit=True,
            neighbor_win_len=0
            ):
        """
        process a stretch-stack object to an along-river, per-node
        height/width model array
        """
        try:
            # stretch_stack object
            wse = obj.wse.copy()
            width = obj.wse.copy()
        except AttributeError as e:
            # flow_state
            wse = obj.wse_profiles.copy()
            width = obj.width_profiles.copy()
        height_width = cls.from_arrays(
            wse,
            width,
            along_dist=obj.along_dist.copy(),
            node_id=obj.node_id.copy(),
            stretch_name=obj.stretch_name,
            wse_reference=wse_reference,
            ptile_list=ptile_list,
            snapit=snapit,
            neighbor_win_len=neighbor_win_len)
        return height_width


    @classmethod
    def from_arrays(
            cls,
            wse_in,# 1D or 2D array
            width_in,# 1D or 2D array
            along_dist=None,
            node_id=None,
            stretch_name=None,
            wse_reference=None,
            ptile_list = [5, 25, 32, 50, 68, 75, 95],
            snapit=True,
            neighbor_win_len=0
            ):
        """
        neighbor_win_len > 0 enables computation over multi-node sections
        it is the length in one direction (e.g., 1 means 3 total nodes: 
        the center node, 1 upstream, and 1 downstream node)
        """
        # init the object
        height_width = cls()
        height_width.ptile_list = np.array(ptile_list)
        if stretch_name is not None:
            height_width.stretch_name = stretch_name
        if along_dist is not None:
            height_width.along_dist = along_dist
        if node_id is not None:
            height_width.node_id = node_id
        # set up to make ptiles
        wse0 = wse_in.copy()
        width0 = width_in.copy()
        
        wse = wse0
        width = width0
        # stack node neighbors
        for offset0 in range(neighbor_win_len):
            offset = offset0 + 1
            width_plus = np.roll(width0, offset, axis=0)
            width_minus = np.roll(width0, -offset, axis=0)
            width_plus[0:offset,:] = np.nan
            width_minus[-offset:,:] = np.nan
            # now stack them
            width = np.hstack((width_minus, width, width_plus))
            if wse_reference is not None:
                # do the same for wse but adjust the wse from the other nodes
                #breakpoint()
                wse_ref = np.broadcast_to(wse_reference, np.shape(wse0.T)).T
                dwse_plus = wse_ref - np.roll(wse_ref, offset, axis=0)
                dwse_minus = np.roll(wse_ref, -offset, axis=0) - wse_ref
                wse_plus = np.roll(wse0, offset, axis=0) + dwse_plus
                wse_minus = np.roll(wse0, -offset, axis=0) - dwse_minus
                wse_plus[0:offset,:] = np.nan
                wse_minus[-offset:,:] = np.nan
                wse = np.hstack((wse_minus, wse, wse_plus))
            else:
                # just use the same wse as the center node
                wse = np.hstack((wse0, wse, wse0))
        #breakpoint()
        # check that wse and width are all valid over the same places
        good_msk = np.logical_and(np.isfinite(wse), np.isfinite(width))
        bad_msk = np.logical_or(~np.isfinite(wse), ~np.isfinite(width))
        #
        wse[bad_msk] = np.nan
        width[bad_msk] = np.nan
        Ph = np.nanpercentile(wse, ptile_list, axis=-1).T
        Pl_hat = np.nanpercentile(width, ptile_list, axis=-1).T
        # now match-up the height and width percentiles
        # and make a wse a piecewise-linear function of width
        # ignore nosie on height for now (TODO: handle it)
        #breakpoint()
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
        #
        height_width.width_err = np.sqrt(np.nanmean(dw**2, axis=-1))
        height_width.wse_err = np.sqrt(np.nanmean(dh**2, axis=-1))
        ones_good = np.zeros(np.shape(wse))
        ones_good[good_msk] = 1
        height_width.count = np.sum(ones_good, axis=-1)
        #
        # need to loop over nodes
        sp_r = []
        sp_p = []
        #k = 0
        for this_wse, this_width in zip(wse, width):
            #print(f'{k}')
            msk = np.logical_and(np.isfinite(this_wse), np.isfinite(this_width))
            if np.sum(msk)>0:
                sp, p_val = spearmanr(this_wse[msk], this_width[msk])
                sp_r.append(sp)
                sp_p.append(p_val)
            else:
                sp_r.append(np.nan)
                sp_p.append(np.nan)
            #k = k + 1
        #breakpoint()
        height_width.spearman_r = np.array(sp_r)
        height_width.spearman_p_value = np.array(sp_p)
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
        return height_width
    
    def sample_row(self, x_in, row, 
            x_key='width', kind='linear', fill_value='extrapolate'):
        """
        sample a particular row/model (e.g. a particular node)
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
            self[x_key+'_coords'][row,:],
            self[y_key+'_coords'][row,:],
            kind=kind,
            bounds_error=False,
            fill_value=fill_value
            )
        y = intrp(x)
        if isinstance(x_in, np.ma.MaskedArray):
            mask = ~np.isfinite(y)
            y = np.ma.masked_array(y, mask=mask)
        #
        return y

    def sample(self, x_in, x_key='width',
            kind='linear', fill_value='extrapolate'):
        """
        x: sample location(s) of the 'signal_key' dimension
        returns: the sampled value(s) of other dimension
        e.g., if signal_key=='width' retun the wse at the point
        by interpolating in between the width samples and 
        """
        #breakpoint()
        y_key = 'wse'
        if x_key=='wse':
            y_key='width'
        x = x_in.copy()
        if len(np.shape(x))==1:
            # broadcast to every num_model
            #breakpoint()
            #if not np.isscalar(model_ids):
            shp = (len(self[x_key+'_coords'][:,0]), len(x))
            x = np.broadcast_to(x, shp)
        x_bad = ~np.isfinite(x)
        #x[x_bad] = 0
        #if isinstance(x_in, np.ma.MaskedArray):
        #    x = x.filled(np.nan)
        y = np.zeros_like(x)
        ks = np.arange(len(self[x_key+'_coords'][:,0])).astype(int)
        for k in ks:
            yy = self.sample_row(x[k,:], k, fill_value=fill_value)
            y[k,:] = yy
        return y

    def sample_deriv(self, x, x_key='width', kind='linear', delta=0.1):
        """
        estimate the derivative with finite differences
        """
        y_plus = self.sample(x + delta, x_key, kind)
        y_minus = self.sample(x - delta, x_key, kind)
        return (y_plus -y_minus) / (2 * delta)

    def snap_to_curve(self, wse_in, width_in, sig_wse_in, sig_width_in, wse0_in):
        """
        find 'closest' point on the curve to each point with given covariance
        (assuming each point is indep)
        
        Basically just do a non-linear max-liklihood estimate of the
        observations given that the 'true' value lies on the curve
        and the height/width measurements are uncorrelated.
        """
        from scipy.optimize import minimize
        def ml_error(wse_t,
                wse_m, width_m, sig_wse, sig_width, height_width, row):
            width_t = height_width.sample_row(wse_t, row, x_key='wse')
            return np.sum((wse_m - wse_t)**2 / sig_wse**2 + (
                width_m - width_t)**2 / sig_width**2)
        #
        if np.isscalar(sig_wse_in):
            sig_wse_in = np.zeros_like(wse_in[:,0]) + sig_wse_in
        if np.isscalar(sig_width_in):
            sig_width_in = np.zeros_like(width_in[0,:]) + sig_width_in
        k = 0
        WSE_est = np.zeros_like(wse_in) + np.nan
        Width_est = np.zeros_like(width_in) + np.nan
        Err = np.zeros_like(wse_in[:,0]) + np.nan
        K = len(wse_in[:,0])
        print("Snapping to curve")
        for wse, width, sig_wse, sig_width, wse0 in zip(
                wse_in, width_in, sig_wse_in, sig_width_in, wse0_in):
            print(f"{k} of {K}")
            #breakpoint()
            good = np.logical_and(np.isfinite(wse), np.isfinite(width))
            if np.sum(good)<1:
                continue
            s_wse = sig_wse
            if not np.isscalar(s_wse):
                s_wse = s_wse[good]
            s_width = sig_width
            if not np.isscalar(s_width):
                s_width = s_width[good]
            res = minimize(ml_error, wse0[good], method='nelder-mead',
                args=(wse[good], width[good], s_wse, sig_width, self, k),
                options={'xatol': 1e-8, 'disp': True})
            #
            #breakpoint()
            wse_est = res.x
            width_est = self.sample_row(wse_est, k, x_key='wse')
            err = ml_error(wse_est, wse[good], width[good], sig_wse, sig_width, self, k)
            #breakpoint()
            #
            WSE_est[k,good] = wse_est
            Width_est[k,good] = width_est
            Err[k] = err
            k = k + 1
        return WSE_est, Width_est, np.sqrt(Err)

    def plot(self, outdir=None, show=False, title_tag=None,
            surface=True, scatter=False):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # just use index
        node_index = np.arange(len(self.along_dist))
        #along_dist0 = np.broadcast_to(
        #    np.arange(len(self.along_dist)),
        #        np.shape(self.width_coords.T)).T
        #xlabel='node index'
        #if (np.sum(self.along_dist.mask)==0) and not surface:
            # use along distance
        along_dist1d = self.along_dist
        #np.abs(self.along_dist - np.min(self.along_dist))
        xlabel='along_dist (m)'
        along_dist0 = np.broadcast_to(
            along_dist1d, np.shape(self.width_coords.T)).T
        if scatter:
            ptile = np.broadcast_to(self.ptile_list, np.shape(self.width_coords))
            scat = ax.scatter(
                along_dist0,
                self.width_coords,
                self.wse_coords,
                c=ptile)
            fig.colorbar(scat, ax=ax, label='percentile')
        else:
            # plot along-river lines of percentile
            for k in range(len(self.wse_coords[0,:])):
                ax.plot(
                    along_dist0[:,k],
                    self.width_coords[:,k],
                    self.wse_coords[:,k],
                    label=f'{self.ptile_list[k]}%ile'
                    )
            ax.legend()
        if surface:
            width_bins = np.linspace(
                np.nanpercentile(self.width_coords,1),
                np.nanpercentile(self.width_coords,99),
                100)
            # TODO: enable interpolating in along_dist1d (e.g., uneven node sampling)
            
            grid_y, grid_x = np.meshgrid(width_bins, along_dist1d)
            #breakpoint()
            # only extrapolate a little bit
            grid_z_noextrap = self.sample(grid_y,x_key='width', fill_value=None)
            grid_z0 = self.sample(grid_y,x_key='width')
            # get mask near non extrapolated
            msk = np.zeros(np.shape(grid_z0))
            msk[np.isfinite(grid_z_noextrap)] = 1
            struc = np.array([[0,1,0],[1,1,1],[0,1,0]])
            msk_d = scipy.ndimage.binary_dilation(msk, structure=struc,
                iterations=2)
            # smooth out extrapolation
            grid_z_clip = grid_z0.copy()
            grid_z_clip[grid_z0>np.max(self.wse_coords)] = np.nanmax(
                self.wse_coords)
            grid_z_clip[grid_z0<np.min(self.wse_coords)] = np.nanmin(
                self.wse_coords)
            grid_z_sm = scipy.ndimage.uniform_filter(grid_z_clip, size=3)
            grid_z = grid_z_sm.copy()
            grid_z[msk==1] = grid_z0[msk==1]
            grid_z[msk_d==0] = np.nan
            grid_z_rgb = grid_z.copy()
            zmin = np.nanmin(self.wse_coords)
            grid_z_rgb[~(np.isfinite(grid_z_rgb))] = zmin
            grid_z_rgb[msk_d==0] = zmin
            # now interpolate/resample in regular grid in along_river
            ls = LightSource(270, 45)
            
            rgb = ls.shade(grid_z_rgb, cmap=cm.copper#gist_earth#gray,#Blues,#'gray'],
                #vert_exag=100,
                #blend_mode='soft',
                #dx=200,
                #dy=width_bins[1]-width_bins[0],
                #vmin = np.min(grid_z_rgb) - (np.max(grid_z_rgb) - np.min(grid_z_rgb) )/2
                )
            ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.3,
                facecolors=rgb, edgecolor='none')#, shade=False)#,linewidth=0.2)#, shade=False)#, linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('width (m)')
        ax.set_zlabel('wse (m)')
        ax.set_zlim((np.nanmin(self.wse_coords), np.nanmax(self.wse_coords)))
        title = self.stretch_name
        if title_tag is not None:
            title = title + ' ' +title_tag
        ax.set_title(title)
        fname = title + '_wse_vs_width_scatter' 
        if outdir is not None:
            # create output dir if not exist
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            fname = fname.replace(' ','_')
            plt.savefig(os.path.join(outdir, fname), dpi=300)
            plt.close()
        if show:
            plt.show()

    def crop_to_reach(
            self,
            reach_id=None):
        this = HeightWidthModelArray()
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
        this.stretch_name = self.stretch_name
        # crop the along-river variables
        for key in self.variables.keys():
            #print(key)
            dims = self.VARIABLES[key]['dimensions']
            if 'num_nodes' in dims.keys():
                this[key] = self[key][mask]#mask[0]:mask[-1]]
            else: 
                this[key] = self[key]
        #make along_dist start at 0
        this['along_dist'] = this['along_dist'] - np.min(this['along_dist'])
        return this

