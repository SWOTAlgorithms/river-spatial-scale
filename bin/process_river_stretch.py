#!/usr/bin/env python
'''
Copyright (c) 2024-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Brent Williams

This code processes the river streaches in an input csv file.  Note you first need
to create the swot_node_df data either bby calling the hydrocron.py script (to get
data from podaac), or by creating one from the off-lone rerun river-tiles.

 TODO: need to update this to be more generic with input arguments etc instead of hard-coding them
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rivscale
import rivscale.io
import rivscale.estimate 
import rivscale.plot
import rivscale.data
import rivscale.reconstruct
import rivscale.filter
import rivscale.products

import scipy.signal

import rivscale.filter

from errtools.misc import swot_time_to_field_time
import errtools.plots
import os.path

def process_stretch(stretch_reaches, swot_node_df, sword_node_df, d_up, d_down):
    # make the stretch multitemporal stack object
    stretch_data = rivscale.data.make_stretch_stack(
        stretch_reaches, swot_node_df, sword_node_df, d_up, d_down)
    # populate witdh_u
    stretch_data['width_u'] = (stretch_data['area_tot_u'] * stretch_data['width']) / (
        stretch_data['area_total'])
    # filter out bad data (call it twice to get them all)
    stretch_data = rivscale.filter.filter_bad_stretch_data(stretch_data)
    stretch_data = rivscale.filter.filter_bad_stretch_data(stretch_data)
    # drop times/cycles with too little good quality data
    stretch_data = rivscale.filter.drop_stretch_nans(stretch_data)
    # compute statistics
    stretch_data = rivscale.estimate.get_stretch_stats(stretch_data, signal_key='wse')
    stretch_data = rivscale.estimate.get_stretch_stats(stretch_data, signal_key='width')
    # compute the reference profiles
    stretch_data['wse_reference'] = rivscale.estimate.get_med_profile(
        stretch_data['wse'], stretch_data['dist_out'])
    stretch_data['width_reference'] = rivscale.estimate.get_med_profile(
        stretch_data['width'], stretch_data['dist_out'])
    # set up the bayes estimator signal covariance
    char_length_tau = 100000 # TODO: estimate these from the data
    prior_unc_alpha = 1.5
    stretch_data['wse_cov'] = rivscale.reconstruct.exponential_cov(
            stretch_data['dist_out'], char_length_tau=char_length_tau,
            prior_unc_alpha=prior_unc_alpha)
    char_length_tau = 100000 # TODO: estimate these from the data
    prior_unc_alpha = 200
    stretch_data['width_cov'] = rivscale.reconstruct.exponential_cov(
            stretch_data['dist_out'], char_length_tau=char_length_tau,
            prior_unc_alpha=prior_unc_alpha)
    # now do the bayes reconstruction
    stretch_data = rivscale.reconstruct.reconstruct_stretch(
        stretch_data, signal_key='wse', uncert_key='wse_u')
    stretch_data = rivscale.reconstruct.reconstruct_stretch(
        stretch_data, signal_key='width', uncert_key='width_u')
    return stretch_data

def main():
    #df = pd.read_csv('swot_data_Ocmulgee_River.csv')
    #df = pd.read_csv('swot_data_ocmulgee.csv')
    #df = pd.read_csv('swot_data_all.csv')
    df = pd.read_csv('calval_nodes_delivery_merged_240826_v5.csv')
    #df = df[df['river_name']=='Ocmulgee River']
    # drop elements with no_data times
    df = df[df['time_str']!='no_data']
    df['time_str'] = pd.to_datetime(df['time_str'])
    df['date'] = [ dt.date() for dt in df['time_str']]
    
    # drop bad data
    df = rivscale.filter.filter_node_qual(df, height=True, area=False)
    #
    df['cycle'] = df['cycle_id']
    df['local_node_id'] = rivscale.misc.node_id_to_local_node_id(df['node_id'])
    df['wse_u'] = df['wse_r_u']
    #
    swot_node_df = df
    sword_dir = '/u/swot-fn-r0/swot/sim_proc_inputs/river_database/20230802/v16/netcdf'
    #sword_file = '/Users/bawillia/Desktop/data/SWORD/v16/netcdf/na_sword_v16.nc'
    #sword_df, sword_node_df, d_up, d_down = rivscale.io.read_SWORD(sword_file)
    #breakpoint()
    #sword_df = sword_df[sword_df['river_name'] == 'Ocmulgee River']
    #network_list = rivscale.data.get_connected_networks(sword_df, d_up, d_down)
    #network_list = [np.unique(np.sort(swot_node_df['reach_id']))]
    #df_stretches = pd.read_csv('Flow_wave_reaches_fixed.csv')
    df_stretches = pd.read_csv('calval_stretches.csv')
    stretch_list = []
    for key in df_stretches.keys():
        sword_name = 'na_sword_v16.nc'
        if key == 'Waimak':
            sword_name = 'oc_sword_v16.nc'
        if key == 'Garonne':
            sword_name = 'eu_sword_v16.nc'
        sword_file = os.path.join(sword_dir,sword_name)
        sword_df, sword_node_df, d_up, d_down = rivscale.io.read_SWORD(sword_file)
        #stretch_reaches = np.array(
        #    df_stretches[np.isfinite(df_stretches[key])][key]).astype(int)
        stretch_reaches = np.array(
            df_stretches[df_stretches[key]>0][key]).astype(int)
        print("processing stretch:", key, ", Reaches:",stretch_reaches)
        # process the stretch
        stretch_data = process_stretch(stretch_reaches, swot_node_df, sword_node_df, d_up, d_down)
    
        # write out the data to ncfile
        stretch_data.to_ncfile('{}_stretch.nc'.format(key))
        #breakpoint()
    breakpoint()

if __name__ == "__main__":
    main()
