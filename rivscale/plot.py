'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent Williams

'''
import glob
import os.path

import argparse

import numpy as np
import pandas as pd

import errtools.plots
import matplotlib.pyplot as plt

import scipy.ndimage
from scipy.linalg import pinv, svd, eigh, norm

from errtools.misc import (split_utc_time, field_time_to_swot_time,
    swot_time_to_field_time, get_dist_to_outlet_from_node_id)

from errtools.plots import plot_cdf

import netCDF4 as nc

import statsmodels.api

import geopandas as gpd

import rivscale.misc
import seaborn as sns

########## Sept 2024 stretch-based processing
def plot_stretch_profiles(
        stretch_data,
        x_key='dist_out',
        y_key='wse',
        plot_anom=False,
        maskem=True,
        title='',
        show=False,
        outdir=None):
    """
    function for plotting profiles
    """
    x = stretch_data[x_key]
    y = stretch_data[y_key]
    y2 = stretch_data['bayes_'+y_key]
    anom_str = ''
    if plot_anom:
        # handle anomaly plots
        ref = stretch_data['{}_reference'.format(y_key)]
        ref2 = np.broadcast_to(ref, np.shape(y.T)).T
        y = y - ref2
        y2 = y2 - ref2
        anom_str = ' anomaly'
    if x_key=='time_id':
        # convert to datetime
        x = swot_time_to_field_time(x*60*60)
        y = y.T
        y2 = y2.T
    if maskem:
        nanmask = np.ones_like(y)
        nanmask[~np.isfinite(y)] = np.nan
        y = y * nanmask
        y2 = y2 * nanmask
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x, y)
    ylabel = '{}{}'.format(y_key, anom_str)
    plt.ylabel(ylabel)
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(x, y2)
    plt.ylabel('{}{}'.format('bayes_'+y_key, anom_str))
    plt.xlabel(x_key)
    plt.suptitle(title)
    plt.grid()
    if outdir is not None:
        # create output dir if not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = '{}_profile_{}_vs_{}'.format(title,ylabel, x_key)
        plt.savefig(os.path.join(outdir, fname))
        plt.close()
    else:
        if show:
            plt.show() 

def plot_stretch_width_vs_wse(
        stretch_data,
        wse_key='wse',
        width_key='width',
        maskem=True,
        title='',
        show=False,
        outdir=None):
    wse = stretch_data[wse_key]
    width = stretch_data[width_key]
    wse_ref = stretch_data['wse_reference']
    width_ref = stretch_data['width_reference']

    anom = wse - np.broadcast_to(wse_ref, np.shape(wse.T)).T
    anom_w = width - np.broadcast_to(width_ref, np.shape(width.T)).T
    if maskem:
        nanmask = np.ones_like(anom)
        nanmask[~np.isfinite(anom)] = np.nan
        nanmask[~np.isfinite(anom_w)] = np.nan
        anom = anom * nanmask
        anom_w = anom_w * nanmask
    # also do the wse vs width
    key_wse = '{} anomaly'.format(wse_key)
    key_width = '{} anomaly'.format(width_key)
    dic = {
        key_wse:anom[np.isfinite(anom*anom_w)],
        key_width:anom_w[np.isfinite(anom*anom_w)]
    }
    df = pd.DataFrame(dic)
    fig =errtools.plots.scatterdensity(
        df, key_wse, key_width, show=False)
    ylim = (np.nanpercentile(df[key_width], 2),
        np.nanpercentile(df[key_width], 95))
    xlim = (np.nanpercentile(df[key_wse], 2),
        np.nanpercentile(df[key_wse], 95))
    if fig is not None:
        fig.set_ylim(ylim)
        fig.set_xlim(xlim)
        fig.set_title(title)
    if outdir is not None:
        # create output dir if not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = '{}_density_{}_vs{}'.format(title, width_key, wse_key)
        plt.savefig(os.path.join(outdir, fname))
        plt.close()
    else:
        if show:
            plt.show()

def plot_stretch_spacetime(stretch_data, title='', show=False, outdir=None):
    """
    make plot of time/space sampling
    """
    wse = stretch_data['wse']
    dist_out = stretch_data['dist_out']
    #time_id = stretch_data['time_id']
    time_id = swot_time_to_field_time(stretch_data['time_id']*60*60)
    reach_id = rivscale.misc.reach_id_from_node_id_int(
        stretch_data['node_id'])
    # make plot of time/space sampling
    dist_out2 = np.broadcast_to(dist_out, np.shape(wse.T)).T
    time_id2 = np.broadcast_to(time_id, np.shape(wse))
    reach_id2 = np.broadcast_to(reach_id, np.shape(wse.T)).T
    dic = {
        'time_id':time_id2[wse>0],
        'dist_out':dist_out2[wse>0],
        'reach_id':reach_id2[wse>0]
        }
    df_tmp = pd.DataFrame(dic)
    plt.figure()
    sns.scatterplot(data=df_tmp, x='time_id', y='dist_out',
        hue='reach_id', palette='tab20')#"deep")
    plt.xticks(rotation=15)
    plt.title(title)
    plt.tight_layout()
    if outdir is not None:
        # create output dir if not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = '{}_spacetime'.format(title)
        plt.savefig(os.path.join(outdir, fname))
        plt.close()
    else:
        if show:
            plt.show()

def plot_stretch(stretch_data, title='', outdir=None):
        # wse
        plot_stretch_profiles(
            stretch_data,
            x_key='dist_out',
            y_key='wse',
            plot_anom=False,
            title=title,
            outdir=outdir)
        # wse anom
        plot_stretch_profiles(
            stretch_data,
            x_key='dist_out',
            y_key='wse',
            plot_anom=True,
            title=title,
            outdir=outdir)
        # width
        plot_stretch_profiles(
            stretch_data,
            x_key='dist_out',
            y_key='width',
            plot_anom=False,
            title=title,
            outdir=outdir)
        # width anom
        plot_stretch_profiles(
            stretch_data,
            x_key='dist_out',
            y_key='width',
            plot_anom=True,
            title=title,
            outdir=outdir)
        # wse anom vs time
        plot_stretch_profiles(
            stretch_data,
            x_key='time_id',
            y_key='wse',
            plot_anom=True,
            title=title,
            outdir=outdir)
        # width anom vs time
        plot_stretch_profiles(
            stretch_data,
            x_key='time_id',
            y_key='width',
            plot_anom=True,
            title=title,
            outdir=outdir)
        # plot the width vs wse desnity plots
        plot_stretch_width_vs_wse(
            stretch_data,
            wse_key='wse',
            width_key='width',
            maskem=True,
            title=title,
            outdir=outdir)
        plot_stretch_width_vs_wse(
            stretch_data,
            wse_key='bayes_wse',
            width_key='bayes_width',
            maskem=True,
            title=title,
            outdir=outdir)
        # make space-time sampling plot
        plot_stretch_spacetime(stretch_data, title=title, outdir=outdir)


######### TODO: clean-up/delete.revise stuff below
def plot_swot_profiles(swot_df, swot_node_df, full_profile_df, swot_mean_df):
    # plot some stuff
    reaches = np.unique(np.array(swot_df['reach_id']))
    bad_cycles = []
    for reach in reaches:
        #plt.figure()
        this_swot_df = swot_node_df[swot_node_df['reach_id']==reach]
        this_swot_r_df = swot_df[swot_df['reach_id']==reach]
        this_swot_mean_df = swot_mean_df[swot_mean_df['reach_id']==reach].sort_values('node_id')
        #this_full_profile_df = full_profile_df[full_profile_df['reach']==reach]
        ind0 = np.where(full_profile_df['reach']==reach)
        #breakpoint()
        print(reach, ind0, len(ind0))
        plotit=False
        if len(ind0)>0:
            ind00 = ind0[0]
            if len(ind00)>0:
                ind = ind00[0]
                plotit=True
                mean_profile_wse = full_profile_df['mean_reach_profile'][ind]
                mean_profile_nodes = full_profile_df['node_id'][ind]
        #plt.plot(this_swot_mean_df['node_id'], this_swot_mean_df['IQR_profile'],'-')
        #plt.plot(this_swot_mean_df['node_id'], this_swot_mean_df['IQR_profile'],'-')
        #plt.title('reach: {}'.format(reach))
        plt.figure()
        lgnd = []
        for cycle in np.unique(this_swot_df['cycle']):
            this_df0 = this_swot_df[this_swot_df['cycle']==cycle].sort_values('node_id')
            this_df = this_df0[this_df0['node_q']==1]
            #this_r_df = this_swot_r_df[this_swot_r_df['cycle']==cycle]
            #wse_r = np.array(this_r_df['wse'])
            #wse_r = np.array(this_swot_mean_df['mean_profile'])
            #if len(wse_r) == 0:
            #    continue
            #wse_r = this_swot_mean_df['mean_profile']
            plt.plot(this_df['node_id'], this_df['wse'],'>')
        plt.plot(this_swot_mean_df['node_id'], this_swot_mean_df['mean_profile'],'-')
        if plotit:
            plt.plot(mean_profile_nodes, mean_profile_wse,'--')
        #mn_r_wse = np.mean(this_swot_r_df['wse'])
        #plt.plot(this_swot_mean_df['node_id'], this_swot_mean_df['mean_profile']-mn_r_wse,'-')
        #plt.gca().set_prop_cycle(None)
        #for cycle in np.unique(this_swot_df['cycle']):
        #    this_df = this_swot_mean_df[this_swot_df['cycle']==cycle].sort_values('node_id')
            #this_df = this_df0[this_df0['node_q']==1]
        #    plt.plot(this_df['node_id'], this_df['wse'],'-')
        #for cycle in np.unique(this_swot_df['cycle']):
        #    this_df = this_swot_df[this_swot_df['cycle']==cycle].sort_values('node_id')
        #    plt.plot(this_df['node_id'], this_df['wse'],'-')
        #    lgnd.append('{}'.format(cycle))
        #plt.legend(lgnd)
        #plt.gca().set_prop_cycle(None)
        #for cycle in np.unique(this_swot_df['cycle']):
        #    this_df0 = this_swot_df[this_swot_df['cycle']==cycle].sort_values('node_id')
        #    this_df = this_df0[this_df0['node_q']==2]
        #    plt.plot(this_df['node_id'], this_df['wse'],'^')
        #plt.gca().set_prop_cycle(None)
        #for cycle in np.unique(this_swot_df['cycle']):
        #    this_df0 = this_swot_df[this_swot_df['cycle']==cycle].sort_values('node_id')
        #    this_df = this_df0[this_df0['node_q']==3]
        #    plt.plot(this_df['node_id'], this_df['wse'],'x')
        ###
        #plt.gca().set_prop_cycle(None)
        #median_wse = np.median(this_swot_r_df['wse'])
        #Q1 = np.percentile(this_swot_r_df['wse'], 25)
        #Q3 = np.percentile(this_swot_r_df['wse'], 75)
        #IQR = Q3 - Q1
        #upper = Q3 + 1.5*IQR
        #lower = Q1 - 1.5*IQR
        #for cycle in np.unique(this_swot_df['cycle']):
        #    this_df = this_swot_df[this_swot_df['cycle']==cycle].sort_values('node_id')
        #    this_r_df = this_swot_r_df[this_swot_r_df['cycle']==cycle]
        #    wse = np.array(this_r_df['wse'])
        #    if len(wse) == 0:
        #        continue
        #    outlier = False
        #    if wse > upper:
        #         bad_cycles.append(cycle)
        #         outlier = True
        #    elif wse < lower:
        #         bad_cycles.append(cycle)
        #         outlier = True
        #    typ = ':'
        #    if outlier:
        #        typ = '-'
        #    plt.plot(this_df['node_id'], np.zeros_like(np.array(this_df['wse'])) + wse[0], typ)
        plt.title('reach: {}'.format(reach))
    plt.show()


def time_series_plots(pt_df, drift_df):
    lgnd = []
    pt_ids = np.unique(pt_df['pt_serial'])
    plt.figure();
    for pt_id in pt_ids:
        this_df = pt_df[pt_df['pt_serial']==pt_id].sort_values('pt_time_UTC')
        plt.plot(this_df['pt_time_UTC'], this_df['mean_node_pt_wse_m']);
        lgnd.append(pt_id)
    drift_ids = np.unique(drift_df['drift_id'])
    for drift_id in drift_ids:
        this_df = drift_df[drift_df['drift_id']==drift_id].sort_values('time_UTC')
        plt.plot(this_df['time_UTC'], this_df['mean_node_drift_wse_m'],'-x');
        lgnd.append(drift_id)
    #plt.legend(lgnd)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('wse (m)')
    plt.title('pt (solid) and drift (-x) wse vs time')


def plot_reach_average_df(df):
    reaches = np.unique(df['reach_id'])
    for reach in reaches:
        this_df = df[df['reach_id']==reach]
        reach_times = np.array(this_df['reach_times'])
        reach_wse = np.array(this_df['reach_wse'])
        reach_slope = np.array(this_df['reach_slope'])
        plt.figure()
        plt.subplot(2,1,1)
        plt.title('reach: {}'.format(reach))
        plt.plot(reach_times, reach_wse)
        plt.xlim((reach_times[0],reach_times[-1]))
        plt.ylabel('pt_reach wse (m)')
        plt.subplot(2,1,2)
        plt.plot(reach_times, reach_slope*100*1000)
        plt.ylabel('pt_reach slope (cm/km)')
        plt.xlim((reach_times[0],reach_times[-1]))
        #
        plt.figure()
        plt.scatter(reach_wse, reach_slope*100*1000)
        plt.xlabel('reach wse (m)')
        plt.ylabel('reach slope (cm/km)')

def plot_global_stats(reconst_match_df):
    #mean_wse_swot = np.mean(np.array(reconst_match_df['wse_swot']))
    #mean_wse = np.mean(np.array(reconst_match_df['wse']))
    mean_wse_err = np.mean(np.array(reconst_match_df['wse_err']))
    rel_err = np.array(reconst_match_df['wse_err']) - mean_wse_err
    reconst_match_df['wse_rel_err'] = rel_err
    cm_km = (1000*100) 
    plot_cdf(reconst_match_df['wse_err']*100,
        'reach wse error',
        'wse error (cm)',
        'wse error',
        xlim=(-15, 30))
    plot_cdf(reconst_match_df['wse_rel_err']*100,
        'reach wse error',
        'wse error (cm)',
        'wse error',
        xlim=(-15, 30))
    plot_cdf(reconst_match_df['slope_err']*cm_km,
        'reach slope error CDF',
        'slope error (cm/km)',
        'slope error',
        xlim=(-15, 30))
    plot_cdf(reconst_match_df['slope2_err']*cm_km,
        'reach slope2 error CDF',
        'slope2 error (cm/km)',
        'slope2 error',
        xlim=(-15, 30))


def plot_profile_data(full_profile_data):
    for k, reach in enumerate(full_profile_data['reach']):
        area = np.array(full_profile_data['area_stack'][k])
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(full_profile_data['wse_stack'][k])
        plt.subplot(3,1,2)
        plt.plot(np.sqrt(area))
        plt.subplot(3,1,3)
        plt.plot(np.sqrt(area), full_profile_data['wse_stack'][k],'o')
        plt.suptitle('reach: {}'.format(reach))
        # plot reach-level area vs wse
        #this_df = swot_df[swot_df['reach_id']==reach].sort_values('cycle')
        #cycles = full_profile_data['cycle'][k]
        #cycles_orig = np.array(this_df['cycle'])
        #cycles_to_drop = list(set(cycles_orig) - set(cycles))
        #for cycl in cycles_to_drop:
        #    this_df = this_df[this_df['cycle']!=cycl]
        #area_orig = np.array(this_df['width'])**2
        #plt.figure()
        #plt.subplot(3,1,1)
        #plt.plot(this_df['cycle'],this_df['wse'])
        #plt.subplot(3,1,2)
        #plt.plot(this_df['cycle'],np.sqrt(area_orig))
        #plt.subplot(3,1,3)
        #plt.plot(np.sqrt(area_orig), this_df['wse'],'o')
        #plt.xlabel('reach area')
        #plt.ylabel('reach wse')

def plot_profile_with_bayes0(full_profile_data, bayes_data):
    for k, reach in enumerate(full_profile_data['reach']):
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(full_profile_data['wse_stack'][k], 'o')
        plt.gca().set_prop_cycle(None)
        plt.plot(full_profile_data['wse_stack_interp'][k])
        plt.ylabel('wse (m)')
        #plt.figure()
        plt.subplot(2,2,3)
        plt.plot(bayes_data['wse_stack'][k])
        plt.xlabel('node_index')
        plt.ylabel('wse (m) (Bayes)')
        #plt.figure()
        plt.subplot(2,2,2)
        plt.plot(full_profile_data['cycle'][k], full_profile_data['wse_stack'][k].T, 'o')
        plt.gca().set_prop_cycle(None)
        plt.plot(full_profile_data['cycle'][k], full_profile_data['wse_stack_interp'][k].T)
        #plt.figure()
        plt.subplot(2,2,4)
        plt.plot(bayes_data['cycle'][k], bayes_data['wse_stack'][k].T)
        plt.xlabel('cycle')
        plt.suptitle('reach: {}'.format(reach))

def plot_profile_with_bayes(full_profile_data, bayes_data, prof_stack_data=None):
    for k, network in enumerate(full_profile_data['network']):
        dst = full_profile_data['dist_out'][k]/1000
        y = full_profile_data['wse_stack'][k]
        y1 = np.ones_like(y) + np.nan
        if 'pt_wse_stack'in full_profile_data:
            y1 = full_profile_data['pt_wse_stack'][k]
        y2 = bayes_data['wse_stack'][k]
        ylab_tag = ''
        if prof_stack_data is not None:
            # plot the anomaly
            y = y - prof_stack_data['med_prof_stack'][k]
            y2 = y2 - prof_stack_data['med_prof_stack'][k]
            ylab_tag = 'anomaly '
            if 'pt_wse_stack' in full_profile_data:
                y1 = y1 - prof_stack_data['med_prof_stack'][k]
        xlim = [np.min(dst), np.max(dst)]
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(dst, y)
        plt.xlim(xlim)
        plt.grid()
        if 'pt_wse_stack' in full_profile_data:    
            plt.gca().set_prop_cycle(None)
            plt.plot(dst, y1, 'o')
        plt.ylabel('wse {}(m)'.format(ylab_tag))
        plt.title('reach: {}'.format(full_profile_data['reach'][k]))
        #plt.figure()
        plt.subplot(2,1,2)
        plt.plot(dst, y2)
        if 'pt_wse_stack' in full_profile_data:
            plt.gca().set_prop_cycle(None)
            plt.plot(dst, y1, 'o')
        plt.xlabel('distance to outlet (km)')
        plt.ylabel('wse {}(m) (Bayes)'.format(ylab_tag))
        plt.xlim(xlim)
        plt.grid()
        #plt.figure()
        #plt.subplot(2,2,2)
        #plt.plot(full_profile_data['cycle'][k], full_profile_data['wse_stack'][k].T, 'o')
        #plt.gca().set_prop_cycle(None)
        #plt.plot(full_profile_data['cycle'][k], full_profile_data['wse_stack_interp'][k].T)
        #plt.figure()
        #plt.subplot(2,2,4)
        #plt.plot(bayes_data['cycle'][k], bayes_data['wse_stack'][k].T)
        #plt.xlabel('cycle')
        plt.suptitle('network: {}'.format(network))

        #plt.figure()
        #plt.plot(full_profile_data['wse_stack_interp'][k] - bayes_data['wse_stack'][k])

def plot_reach_average(reach_average_data, swot_df):
    cm_km = 100*1000
    for k, reach in enumerate(reach_average_data['reach']):
        this_df = swot_df[swot_df['reach_id']==reach].sort_values('cycle')
        # drop the cycles that are not in the reach_average_data
        cycles = reach_average_data['cycle'][k]
        cycles_orig = np.array(this_df['cycle'])
        cycles_to_drop = list(set(cycles_orig) - set(cycles))
        for cycl in cycles_to_drop:
            this_df = this_df[this_df['cycle']!=cycl]
        plt.figure()
        plt.subplot(3,2,1)
        plt.plot(reach_average_data['cycle'][k], reach_average_data['wse'][k],'.-')
        plt.plot(this_df['cycle'], this_df['wse'],'.-')
        #plt.title('reach: {}'.format(reach))
        plt.ylabel('reach average wse (m)')
        plt.subplot(3,2,3)
        plt.plot(reach_average_data['cycle'][k], reach_average_data['slope'][k]*cm_km),'.-'# cm/km
        plt.plot(this_df['cycle'], this_df['slope']*cm_km,'.-')
        plt.plot(this_df['cycle'], this_df['slope2']*cm_km,'.-')
        #plt.title('reach: {}'.format(reach))
        plt.ylabel('reach slope (cm/km)')
        plt.xlabel('cycle')
        plt.subplot(3,2,5)
        plt.plot(reach_average_data['cycle'][k], reach_average_data['width'][k],'.-')
        plt.plot(this_df['cycle'], this_df['width'],'.-')
        plt.ylabel('reach width (m)')
        plt.xlabel('cycle')
        plt.subplot(3,2,4)
        plt.plot(reach_average_data['wse'][k],reach_average_data['slope'][k]*cm_km, 'o')
        plt.plot(this_df['wse'], this_df['slope']*cm_km,'x')
        plt.plot(this_df['wse'], this_df['slope2']*cm_km,'^')
        plt.xlabel('reach average wse (m)')
        plt.ylabel('reach slope (cm/km)')
        plt.subplot(3,2,6)
        plt.plot(reach_average_data['wse'][k],reach_average_data['width'][k], 'o')
        plt.plot(this_df['wse'], this_df['width'],'x')
        plt.xlabel('reach average wse (m)')
        plt.ylabel('reach width (m)')
        plt.suptitle('reach: {}'.format(reach))


def plot_anomaly_per_cycle(anom_data):
    cycles = np.unique(np.concatenate([list(c) for c in anom_data['cycle']]))
    for cyc in cycles:
        plt.figure()
        for k,reach in enumerate(anom_data['reach']):
            #breakpoint()
            ind = np.where(np.array(anom_data['cycle'][k])==cyc)
            #breakpoint()
            if len(ind)==0:
                continue
            else:
                ind = ind[0]
            if len(ind)>0:
                ind =ind[0]
            #breakpoint()
            plt.plot(np.array(anom_data['dist_out'][k]),np.array(anom_data['wse_stack'][k][:,ind]))
        plt.show()


def plot_mean_profile(full_profile_data, prof_stack_data):
    for k, network in enumerate(full_profile_data['network']):
        dst = full_profile_data['dist_out'][k]/1000
        y1 = full_profile_data['wse_stack'][k]
        #y1 = np.ones_like(y) + np.nan
        #if 'pt_wse_stack'in full_profile_data:
        #    y1 = full_profile_data['pt_wse_stack'][k]
        #y2 = bayes_data['wse_stack'][k]
        y2 = full_profile_data['wse_stack_linfit'][k]
        y3 = y1 - y2
        mean_linfit = np.nanmean(y2, axis=1)
        y4 = prof_stack_data['med_prof_stack'][k][:,0] - mean_linfit
        #y4 = full_profile_data['wse_stack_detrend'][k]
        #breakpoint()
        ylab_tag = ''
        #if prof_stack_data is not None:
        #    # plot the anomaly
        #    y = y - prof_stack_data['med_prof_stack'][k]
        #    y2 = y2 - prof_stack_data['med_prof_stack'][k]
        #    ylab_tag = 'anomaly '
        #    if 'pt_wse_stack' in full_profile_data:
        #        y1 = y1 - prof_stack_data['med_prof_stack'][k]
        xlim = [np.min(dst), np.max(dst)]

        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(dst, y1)
        plt.xlim(xlim)
        plt.grid()
        #if 'pt_wse_stack' in full_profile_data:
        #    plt.gca().set_prop_cycle(None)
        #    plt.plot(dst, y1, 'o')
        plt.ylabel('wse {}(m)'.format(ylab_tag))
        #plt.title('reach: {}'.format(full_profile_data['reach'][k]))
        #plt.figure()
        plt.subplot(2,2,2)
        plt.plot(dst, y2)
        #if 'pt_wse_stack' in full_profile_data:
        #    plt.gca().set_prop_cycle(None)
        #    plt.plot(dst, y1, 'o')
        #plt.xlabel('distance to outlet (km)')
        plt.ylabel('wse_linfit {}(m)'.format(ylab_tag))
        plt.xlim(xlim)
        plt.grid()
        #
        ylim = (-1,1)
        plt.subplot(2,2,3)
        plt.plot(dst, y3)
        #if 'pt_wse_stack' in full_profile_data:
        #    plt.gca().set_prop_cycle(None)
        #    plt.plot(dst, y1, 'o')
        plt.xlabel('distance to outlet (km)')
        plt.ylabel('wse - wse_linfit {}(m) '.format(ylab_tag))
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid()
        #
        plt.subplot(2,2,4)
        plt.plot(dst, y4)
        #if 'pt_wse_stack' in full_profile_data:
        #    plt.gca().set_prop_cycle(None)
        #    plt.plot(dst, y1, 'o')
        plt.xlabel('distance to outlet (km)')
        plt.ylabel('mean_wse - mean_linfit {}(m)'.format(ylab_tag))
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid()
        #plt.figure()
        #plt.subplot(2,2,2)
        #plt.plot(full_profile_data['cycle'][k], full_profile_data['wse_stack'][k].T, 'o')
        #plt.gca().set_prop_cycle(None)
        #plt.plot(full_profile_data['cycle'][k], full_profile_data['wse_stack_interp'][k].T)
        #plt.figure()
        #plt.subplot(2,2,4)
        #plt.plot(bayes_data['cycle'][k], bayes_data['wse_stack'][k].T)
        #plt.xlabel('cycle')
        plt.suptitle('network {}\n reach: {}'.format(network,full_profile_data['reach'][k]))
        #plt.suptitle('network: {}'.format(network))
        plt.tight_layout()



### TODO: refactor the plots in this main function...
def main():
    parser = create_parser()
    args = parser.parse_args()
    fles = glob.glob(os.path.join(args.dataframe_dir,'*.csv'))
    # read SWORD
    sword_df, sword_node_df, d_up, d_down = read_SWORD(args.sword_file)

    # read the SWOT data
    swot_df, swot_node_df = read_rivertiles(args.rivertile_dir)
    # drop reaches not in the desired tile (from SWORD tile)
    bad_reaches = list(set(swot_node_df['reach_id'])-set(sword_df['reach_id']))
    for rch in bad_reaches:
        swot_node_df = swot_node_df[swot_node_df['reach_id']!=rch]
    #breakpoint()
    network_list = get_connected_networks(sword_df, d_up, d_down)
    # hack to filter out unneeded things for the yellowstone case
    network_list = [network_list[3]]
    full_profile_data0 = make_swot_data_stack(swot_node_df, sword_node_df)
    full_profile_data1 = network_stack(full_profile_data0, network_list)
    #breakpoint()
    #full_profile_data = filter_stack(full_profile_data1)
    #full_profile_data = exclude_bad_stack_cycles(full_profile_data)
    #full_profile_data = modify_uncert(full_profile_data)
    full_profile_data = modify_uncert(full_profile_data1)
    #breakpoint()
    full_profile_data = interp_stack(full_profile_data, left=np.nan, right=np.nan)
    #breakpoint()

    #full_profile_data0 = filter_stack(full_profile_data)#, plotem=True)
    #full_profile_data = exclude_bad_stack_cycles(full_profile_data0)

    #breakpoint()
    #full_profile_data = interp_stack(full_profile_data, left=np.nan, right=np.nan)
    stats = stack_mean_and_covariance(full_profile_data)
    bayes_data = reconstruct_stack(stats, full_profile_data)
    #plot_profile_with_bayes(full_profile_data, bayes_data)
    
    #breakpoint()
    #reach_average_data = compute_reach(bayes_data)
    #breakpoint()
    #plot_profile_data(full_profile_data)
    #plot_profile_with_bayes(full_profile_data, bayes_data)
    #plot_reach_average(reach_average_data, swot_df)
    #breakpoint()
    #plt.show()

    #swot_df, swot_node_df = filter_bad_swot(swot_df, swot_node_df)
    #swot_node_df = flag_swot_node_outliers(swot_node_df)

    pt_df, drift_df = load_field_data(fles, args.sword_file)
    #breakpoint()
    # make Bayes with simple model params
    stats3 = replace_signal_stats(stats, full_profile_data, sword_node_df)
    bayes_data3 = reconstruct_stack(stats3, full_profile_data)
    #plot_profile_with_bayes(full_profile_data, bayes_data3)
    
    #reach_average_data3 = compute_reach(bayes_data3)
    #plot_reach_average(reach_average_data3, swot_df) 
    #breakpoint()
    #plt.show()
    #breakpoint()

    full_profile_data2 = detrend(full_profile_data, sword_node_df)
    anom_data0 = compute_anomaly(full_profile_data2, bayes_data=None, sword_node_df=None)
    anom_data = compute_anomaly(full_profile_data2, bayes_data=bayes_data3, sword_node_df=None)
    # do bayes, but with med_prof_stack as the mean
    stats2 = {}# estimated cov with full mean profile
    stats4 = {}# exp cov with full mean profile
    stats5 = {}# combo cov with full mean profile
    alpha = 0.2
    for key in stats:
        stats2[key] = []
        stats4[key] = []
        stats5[key] = []
        for k, network in enumerate(stats['network']):
            stats2[key].append(stats[key][k])
            stats4[key].append(stats3[key][k])
            stats5[key].append(stats[key][k])
    for k, network in enumerate(stats['network']):
        stats2['mean'][k] = anom_data['med_prof_stack'][k][:,0]
        stats4['mean'][k] = anom_data['med_prof_stack'][k][:,0]
        stats5['mean'][k] = anom_data['med_prof_stack'][k][:,0]
        stats5['Ry'][k] = (alpha*stats['Ry'][k] + (1-alpha)*stats3['Ry'][k])
    bayes_data2 = reconstruct_stack(stats2, full_profile_data)
    bayes_data4 = reconstruct_stack(stats4, full_profile_data)
    bayes_data5 = reconstruct_stack(stats5, full_profile_data)
    #plot_profile_with_bayes(full_profile_data, bayes_data3, anom_data)
    plot_profile_with_bayes(full_profile_data, bayes_data2)
    plot_profile_with_bayes(full_profile_data, bayes_data2, anom_data)
    #
    plot_profile_with_bayes(full_profile_data, bayes_data4)
    plot_profile_with_bayes(full_profile_data, bayes_data4, anom_data)
    #
    plot_profile_with_bayes(full_profile_data, bayes_data5)
    plot_profile_with_bayes(full_profile_data, bayes_data5, anom_data)
    #plt.show()
    #plt.plot()
    # look at spectra
    # first interpolate over holes, while handling ends
    tmp = full_profile_data['wse_stack_interp'][0]
    msk = ~np.isfinite(tmp)
    tmp0 = full_profile_data['wse_stack'][0]
    msk0 = ~np.isfinite(tmp0)
    tmp[msk] = full_profile_data['wse_stack_linfit'][0][msk]
    tmp2 = tmp - full_profile_data['wse_stack_linfit'][0]
    jnk = tmp0-full_profile_data['wse_stack_linfit'][0]
    n_p = np.nanstd(jnk[0:100,:])
    #n_p = np.nanmedian(full_profile_data['wse_u_stack'][0][~msk0])
    noise = np.random.randn(len(tmp[:,0]), len(tmp[0,:]))*n_p
    # add fake noise to the interpolated sections so it doesnt shape spectral noise floor
    tmp2[msk0] = tmp2[msk0] + noise[msk0]
    #tmp20 = full_profile_data['wse_stack'][0] - full_profile_data['wse_stack_linfit'][0]
    #dist = full_profile_data['dist_out'][0]
    plt.figure()
    plt.plot(tmp2)

    plt.figure()
    avg = 0
    for k in range(len(tmp[0,:])):
        f, pxx = scipy.signal.periodogram(tmp2[:,k])
        #f=f0[1:]
        #f = np.linspace(0.01, 10, 1000)
        #msk = np.where(np.isfinite(tmp20[:,k]))
        #wse = tmp20[:,k]
        #breakpoint()
        #pxx = scipy.signal.lombscargle(wse[msk], dist[msk], f)
        plt.loglog(f, pxx)
        #plt.semilogy(f, pxx)
        avg = avg + pxx
        #plt.show()
    ylim = (1e-6, 1000)
    plt.grid()
    plt.ylabel('power spectral density')
    plt.xlabel('wave-number/spatial-frequency (1/node)')
    plt.ylim(ylim)
    avg = avg / len(tmp[0,:])
    plt.figure()
    plt.loglog(f, avg)
    #n_p = np.sqrt(np.nanmedian(full_profile_data['wse_u_stack'][0]))
    noise = np.random.randn(len(tmp[:,0]))*n_p
    noise2 = np.random.randn(len(tmp[:,0]))*(n_p**2)
    fn, pxxn = scipy.signal.periodogram(noise)
    fn, pxxn2 = scipy.signal.periodogram(noise2)
    pxxn_c = np.zeros_like(pxxn) + np.mean(pxxn)
    plt.loglog(fn, pxxn_c,'--')
    plt.grid()
    plt.ylabel('power spectral density')
    plt.xlabel('wave-number/spatial-frequency (1/node)')
    #plt.loglog(fn, pxxn_c/np.sqrt(len(tmp[0,:])),'--')
    #plt.loglog(fn, pxxn,'--')
    #plt.loglog(fn, pxxn2,'--')
    # look at lomb-scargle periodogram without interpolating first
    #scipy.signal.lombscargle()

    #breakpoint()
    # check out the exponetial that fits this the best...?
    dist = full_profile_data['dist_out'][0]
    dist2 = dist - dist[int(len(dist)/2)]
    char_length_tau = 20000
    prior_unc_alpha=2
    cov = np.exp(-np.abs(dist2) / char_length_tau)
    cov = cov / np.max(cov) * prior_unc_alpha**2
    ft = np.fft.fft(cov)
    plt.loglog(fn, np.abs(ft[0:len(fn)]))
    plt.loglog(fn, np.abs(ft[0:len(fn)])+np.mean(pxxn))
    plt.legend(['mean psd', 'noise floor', 'exp. cov', 'exp. cov + noise'])
    plt.ylim(ylim)
    #plt.show()

    # plot the PTs at the same time/place as SWOT
    nodes = np.unique(pt_df['node_id'])
    good_nodes = [71241000100011, 71241000100101, 71241000110411, 71241000110861, 71241000120341, 71241000120211]
    bad_nodes = list(set(nodes) - set(good_nodes))
    # filter out bad pts
    for node in bad_nodes:
        pt_df = pt_df[pt_df['node_id'] != node]
    #breakpoint()
    pt_swot_match = find_pt_swot_matches(pt_df, full_profile_data)
    stuff_pt_stack(pt_swot_match, full_profile_data)
    plot_profile_with_bayes(full_profile_data, bayes_data4, anom_data)
    
    plot_mean_profile(full_profile_data, anom_data0)
    
    R = stats3['Ry'][0]
    x0 = int(len(R[:,0])/2)
    dst = full_profile_data['dist_out'][0]/1000
    x = dst - dst[x0]
    plt.figure()
    plt.plot(x, R[:, x0])
    plt.grid()
    plt.ylabel('covariance')
    plt.xlabel('distance (km)')
    plt.figure()
    plt.imshow(R)
    plt.colorbar()
    # now plot the posterior cov
    plt.figure()
    plt.imshow(bayes_data4['post_cov'][0][0])
    plt.colorbar()
    plt.title('posterior covariance')
    plt.figure()
    for c,cyc in enumerate(bayes_data['cycle'][0]):
        err = np.diag(bayes_data4['post_cov'][0][c])
        plt.plot(dst, np.sqrt(err))
    plt.xlabel('dist out (km)')
    plt.ylabel('sqrt(post. cov.) (m)')
    plt.grid()
    plt.show()
    #
    breakpoint()
    # now plot some hysteresis stuff
    plt.figure()
    lgnd = []
    nodes = np.unique(pt_df['node_id'])
    good_nodes = [71241000100011, 71241000100101, 71241000110411, 71241000110861, 71241000120341, 71241000120211]
    for node in good_nodes:
        this_pt_df = pt_df[pt_df['node_id']==node]
        plt.plot(this_pt_df['pt_time_UTC'], this_pt_df['mean_node_pt_wse_m'])
        lgnd.append(node)
    plt.xlabel('time UTC')
    plt.ylabel('wse (m)')
    plt.legend(lgnd)
    # 
    cm_km = (1000*100)
    tmp = bayes_data4['wse_stack'][0]-anom_data['med_prof_stack'][0]
    tmp_prime = np.diff(tmp, axis=0) * 100 # in cm
    dst_diff = np.diff(dst)
    #breakpoint()
    for k in range(len(tmp_prime[0,:])):
        tmp_prime[:,k] = tmp_prime[:,k]/dst_diff # in cm/km
    tmp_prime_smooth = scipy.ndimage.uniform_filter1d(tmp_prime, 20, axis=0)
    plt.figure()
    plt.plot(dst[0:-1], tmp_prime)
    plt.xlabel('dist out (km)')
    plt.ylabel('node-level slope anomaly (cm/km)')
    plt.grid()
    plt.figure()
    plt.plot(tmp_prime, tmp[0:-1,:])
    plt.xlabel('wse_anomaly (m)')
    plt.ylabel('node-level slope anomaly (cm/km)')
    plt.grid()
    #
    plt.figure()
    plt.plot(dst[0:-1], tmp_prime_smooth)
    plt.xlabel('dist out (km)')
    plt.ylabel('smoothed node-level slope anomaly (cm/km)')
    plt.grid()
    plt.figure()
    plt.plot(tmp_prime_smooth, tmp[0:-1,:])
    plt.xlabel('wse_anomaly (m)')
    plt.ylabel('smoothed node-level slope anomaly (cm/km)')
    plt.grid()
    #
    plt.figure()
    for node in good_nodes:
        this_pt_df = pt_df[pt_df['node_id']==node]
        mean = anom_data['med_prof_stack'][0]
        nid = anom_data['node_id'][0]
        id0 = np.where(nid == node)
        #mean[id0][0]
        #breakpoint()
        plt.plot(this_pt_df['pt_time_UTC'], np.array(this_pt_df['mean_node_pt_wse_m'])-mean[id0][0][0])
        lgnd.append(node)
    plt.xlabel('time UTC')
    plt.ylabel('wse anomaly (m)')
    plt.legend(lgnd)
    #
    # make a hysteresis plot from field data
    this_node = 71241000110411
    that_node = 71241000100011
    this_pt_df = pt_df[pt_df['node_id']==this_node]
    that_pt_df = pt_df[pt_df['node_id']==that_node]
    this_wse = np.array(this_pt_df['mean_node_pt_wse_m'])
    that_wse = np.array(that_pt_df['mean_node_pt_wse_m'])
    nid = anom_data['node_id'][0]
    this_nid = np.where(nid == this_node)
    that_nid = np.where(nid == that_node)
    #breakpoint()
    this_time = field_time_to_swot_time(this_pt_df['pt_time_UTC_str'])
    that_time = field_time_to_swot_time(that_pt_df['pt_time_UTC_str'])
    # resample to same time
    that_wse2 = np.interp(this_time, that_time, that_wse)
    this_wse_anom = this_wse - anom_data['med_prof_stack'][0][this_nid[0]][0][0]
    that_wse2_anom = that_wse2 - anom_data['med_prof_stack'][0][that_nid[0]][0][0]
    d_wse_anom = this_wse_anom - that_wse2_anom
    plt.figure()
    plt.plot(this_pt_df['pt_time_UTC'], d_wse_anom)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('pressure transducer slope anomaly')
    plt.figure()
    plt.plot(this_pt_df['pt_time_UTC'], this_wse_anom)
    plt.plot(this_pt_df['pt_time_UTC'], that_wse2_anom)
    plt.grid()
    plt.ylabel('pressure transducer wse anomaly')
    plt.xlabel('time')
    plt.figure()
    plt.plot(d_wse_anom, this_wse_anom)
    plt.grid()
    plt.xlabel('pressure transducer wse anomaly')
    plt.ylabel('pressure transducer slope anomaly')
    plt.show()
    # 
    breakpoint()
    # TODO: plot full_extent anomaly (vs dist_out?) smashiing all consecutive reaches together
    #plot_anomaly_per_cycle(anom_data)
    breakpoint()

    # get the mean profiles
    #swot_mean_df = compute_mean_swot_profile(
    #    swot_node_df,
    #    np.unique(swot_node_df['reach_id']),
    #    sword_node_df)
    ##estimate_offsets()
    ## get the covariance
    #full_profile_data = estimate_swot_mean_and_covariances(swot_node_df, sword_node_df, plotem=False)
    #breakpoint()
    ##
    #plot_swot_profiles(swot_df, swot_node_df, full_profile_data, swot_mean_df)
    ##swot_df, swot_node_df = filter_bad_swot(swot_df, swot_node_df)
    ##
    #sword_df, sword_node_df = read_SWORD(args.sword_file)
    #pt_df, drift_df = load_field_data(fles, args.sword_file)

    # filter out bad drifts
    good_drift_df = flag_drifts(drift_df, pt_df, sword_node_df)
    match_df = find_pt_drift_matches(pt_df, drift_df)

    # fix the jumps in the PT data
    #fix_pt_jumps(pt_df, drift_df, good_drift_df, match_df)
   
    # get the pt reach average dataframe
    reach_average_df = get_reach_average_df(pt_df, np.unique(match_df['reach_id'])) 
    plot_reach_average_df(reach_average_df)

    # estimate mean and covariances for each reach
    good_drifts = estimate_mean_and_covariances(drift_df, good_drift_df, sword_node_df, match_df)

    breakpoint()
    good_drifts['mean'] = good_drifts['mean_reach_profile']
    bayes_data2 = reconstruct_stack(good_drifts, full_profile_data)
    reach_average_data2 = compute_reach(bayes_data2)
    plot_profile_with_bayes(full_profile_data, bayes_data2)
    plot_reach_average(reach_average_data2, swot_df)
    plt.show()
    #swot_df2 = bayes_to_swot_df(reach_average_data, swot_df)
    # reconstruct profiles from PTs
    #reconst_match_df = reconstruct_profiles_from_pts(pt_df, good_drifts, swot_df2, sword_df, reach_average_df)
    reconst_match_df = reconstruct_profiles_from_pts(pt_df, good_drifts, swot_df, sword_df, reach_average_df)
    plt.show()
    # plot global stats
    #plot_global_stats(reconst_match_df)
 
if __name__ == "__main__":
    main()

