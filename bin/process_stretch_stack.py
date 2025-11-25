#!/usr/bin/env python
'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

This code processes the river streaches in an input csv file.  Note you first need
to create the swot_node_df data either bby calling the hydrocron.py script (to get
data from podaac), or by creating one from the off-lone rerun river-tiles.

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
import rivscale.products.stretch_stack
import rivscale.products.along_stretch
import rivscale.products.stretch_average
import rivscale.products.height_width
import rivscale.products.flow_state
import rivscale.products.height_width_array
import rivscale.misc

import scipy.signal

import rivscale.filter

import os.path
import configparser
import argparse
import glob
import scipy.ndimage

import warnings

import time
import rivscale.misc

import rivscale.processors.est_priors

import logging
LOGGER = logging.getLogger('estimate_priors')

EXAMPLE=''

def run_processor(processor_name, products, cfg, outfiles):
    print(f'    processing {processor_name}')
    warn_str = f'        already processed {processor_name}, using existing files'
    if processor_name=='width_correction':
        corr_stack = products['stretch_stack_corr']
        if products['stretch_stack_corr'] is not None:
            print(warn_str)
        else:
            corr_stack = rivscale.estimate.process_width_correction(
                cfg, products['stretch_stack'])
            if corr_stack is not None:
                corr_stack.to_ncfile(outfiles['stretch_stack_corr'])
            # update products
            products['stretch_stack_corr'] = corr_stack
        if cfg['use_as_working_stack']:
            products['stretch_stack'] = corr_stack.copy()
    elif processor_name=='dark_stats':
        if products['dark_stats'] is not None:
            print(warn_str)
        else:
            dark_stats = rivscale.estimate.process_dark_stats(
                cfg, products['stretch_stack'])
            # write output
            if dark_stats is not None:
                dark_stats.to_ncfile(outfiles['dark_stats'])
            # update products
            products['dark_stats'] = dark_stats
    elif processor_name=='filter_stack':
        filt_stack = products['stretch_stack_filt']
        if products['stretch_stack_filt'] is not None:
            print(warn_str)
        else:
            filt_stack = rivscale.estimate.process_filter_stack(
                cfg, products['stretch_stack'])
            if filt_stack is None:
                return -1 # return bad exist status
            filt_stack.to_ncfile(outfiles['stretch_stack_filt'])
            products['stretch_stack_filt'] = filt_stack
        if cfg['use_as_working_stack']:
            products['stretch_stack'] = filt_stack.copy()
    if processor_name=='smooth_widths':
        if products['stretch_stack_smoothwidth'] is not None:
            print(warn_str)
        else:
            corr_stack = rivscale.estimate.process_smooth_widths(
                cfg, products['stretch_stack'])
            if corr_stack is not None:
                corr_stack.to_ncfile(outfiles['stretch_stack_smoothwidth'])
            # update products
            products['stretch_stack_smoothwidth'] = corr_stack
    elif processor_name=='along_stats':
        if products['width_stats'] is not None:
            print(warn_str)
        else:
            wse_stats, width_stats = rivscale.estimate.process_along_stats(
                cfg, products['stretch_stack'])
            # write outputs
            if wse_stats is not None:
                wse_stats.to_ncfile(outfiles['wse_stats'])
            if width_stats is not None:
                width_stats.to_ncfile(outfiles['width_stats'])
            #
            products['wse_stats'] = wse_stats
            products['width_stats'] = width_stats
            #
            if (wse_stats is None):
                print(" wse_stats not generated, skipping rest of processing")
                # TODO: should we check width too? but only of not using Pekel?
                #continue
                return -1
    elif processor_name=='stretch_average':
        if products['width_avg'] is not None:
            print(warn_str)
        else:
            wse_avg, width_avg = rivscale.estimate.process_stretch_average(
                cfg,
                products['stretch_stack'],
                products['wse_stats'],
                products['width_stats'])
            # write outputs
            if wse_avg is not None:
                wse_avg.to_ncfile(outfiles['wse_avg'])
            if width_avg is not None:
                width_avg.to_ncfile(outfiles['width_avg'])
            #
            products['wse_avg'] = wse_avg
            products['width_avg'] = width_avg
    elif processor_name=='height_width':
        if products['height_width'] is not None:
            print(warn_str)
        else:
            height_width = rivscale.products.height_width.HeightWidthModel.from_objects(
                cfg,
                products['wse_avg'],
                products['width_avg'],
                products['width_stats'])# TODO: handle Pekel
            if height_width is not None:
                height_width.to_ncfile(outfiles['height_width'])
            products['height_width'] = height_width
    elif processor_name=='flow_state':
        if products['flow_state'] is not None:
            print(warn_str)
        else:
            flow_state = rivscale.estimate.process_flow_state(
                cfg,
                products['stretch_stack'],
                products['wse_stats'],
                products['wse_avg'],
                )
            if flow_state is not None:
                flow_state.to_ncfile(outfiles['flow_state'])
            products['flow_state'] = flow_state
    elif processor_name=='height_width_array':
        if products['height_width_array'] is not None:
            print(warn_str)
        else:
            hw_array = rivscale.estimate.process_height_width_array(
                cfg,
                products['stretch_stack'],
                products['wse_stats'],
                products['flow_state'],
                )
            if hw_array is not None:
                hw_array.to_ncfile(outfiles['height_width_array'])
            products['height_width_array'] = hw_array
    elif processor_name=='reconstruct':
        if products['bayes'] is not None:
            print(warn_str)
        else:
            bayes = rivscale.reconstruct.process_bayes_reconstruction(
                cfg,
                products['stretch_stack'],
                products['wse_stats'],
                products['width_stats'],# TODO: handle Pekel
                products['height_width'])
            if bayes is not None:
                bayes.to_ncfile(outfiles['bayes'])
            products['bayes'] = bayes
    return 0
    
def process_one_stretch(cfg_run, cfg_param, force, stretch_dir0, pekel_dir0, outdir0, key, stretch_stack_flavor):
    """
    this function contains everything that is needed for processing one
    particular stretch
    """
    stretch_dir = os.path.join(
        stretch_dir0, key, 'stretch_stack_{}'.format(
            stretch_stack_flavor))
    pekel_dir = None
    if pekel_dir0 is not None:
        pekel_dir = os.path.join(
           pekel_dir0, key, 'pekel_{}'.format(
                cfg_run['main']['pekel_flavor']))
    outdir = os.path.join(outdir0, key, cfg_run['estimate']['flavor'])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    #this_start = time.time()
    #stretch_reaches = np.array(
    #    df_stretches[df_stretches[key]>0][key]).astype(int)
    #print("processing {} of {}, stretch: {}".format(
    #    i, N, key), ", Reaches:", stretch_reaches)
    # check if already run
    ####
    # setup input/output files
    ####
    infile_stretch = os.path.join(
        stretch_dir, '{}_stretch_stack.nc'.format(key))
    infile_pekel = None
    if pekel_dir  is not None:
        infile_pekel = os.path.join(
            pekel_dir, '{}_pekel_stats.nc'.format(key))
    # get a list of output product names
    product_list = [
        'stretch_stack_corr',
        'dark_stats',
        'stretch_stack_filt',
        'stretch_stack_smoothwidth',
        'wse_stats',
        'width_stats',
        'wse_avg',
        'width_avg',
        'height_width',
        'flow_state',
        'height_width_array',
        'bayes',
        ]
    # create the output files
    outfiles = {}
    for prod_key in product_list:
        outfiles[prod_key] = os.path.join(
            outdir, '{}_{}.nc'.format(key, prod_key))
    ####
    # read the stretch data
    ####
    products = {}
    stretch_stack = \
        rivscale.products.stretch_stack.StretchStack.from_ncfile(
            infile_stretch)
    # populate witdh_u
    # TODO: fix the uncertainty itself instead of fudging it here  
    node_len = stretch_stack['area_total'] / stretch_stack['width']
    stretch_stack['width_u'] = stretch_stack['area_tot_u'] / node_len
    # make measurement uncert at least as much as signal uncert we assume
    stretch_stack['width_u'] = stretch_stack['width_u'] + 10 # + 500#2*prior_unc_alpha_width
    products['stretch_stack'] = stretch_stack
    # exit on missing or short stretch
    # TODO: should probably also check by reach type (either here
    #       or when creating hte stretch_stack)
    if not(os.path.exists(infile_stretch)):
        print("  The input stretch has not been created")
        #continue
        return
    # TODO: make this a param config
    min_nodes_to_process = 50
    if len(stretch_stack.node_id) < min_nodes_to_process:
        print("  This is a short stretch, dont process it?")
        #continue
        return
    #
    ####
    # now process in the order listed in the product_list
    ####
    # init the output products to None
    for prod_key in product_list:
        products[prod_key] = None
    # now go through and try to load or process the products
    for prod_key in product_list:
        # try to load the file for this product
        this_prod = rivscale.io.load_product(outfiles[prod_key],prod_key, force)
        products[prod_key] = this_prod
        #
    
    # get processor list from section heading names
    processor_list = list(cfg_param.keys())
    # exclude the stretch-stack, stretch_avg, and pekel etc
    # assume we start processing with dark_stats
    processor_list = processor_list[processor_list.index('dark_stats'):]
    for processor_name in processor_list:
        status = run_processor(
            processor_name, products, cfg_param[processor_name], outfiles)
        if status < 0:
            # return without processing rest
            # TODO: log
            return
    #this_stop = time.time()
    #print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))





def main():
    parser = argparse.ArgumentParser(
        description='Process river stretch, reach, or multireach',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('config', help='config file')
    parser.add_argument('--force', default=False, action='store_true',
        help='force rerun and overwriting of output files')
    parser.add_argument(
        '-l', '--log-level', type=str, default="debug",
        help="logging level, one of: debug info warning error")
    args = parser.parse_args()
    # read in the config file
    #cfg = configparser.ConfigParser()
    #cfg.read(args.config)
    cfg_run = rivscale.misc.CfgParser()
    cfg_run.read(args.config)
    cfg_param = rivscale.misc.CfgParser()
    cfg_param.read(cfg_run['estimate']['param_config'])
    # handle non-strings for stretch_subset
    #cfg['main']['stretch_subset'] = '{}'.format(cfg['main']['stretch_subset'])
    stretch_list0 = rivscale.misc.get_stretch_list_from_subset_cfg(
        cfg_run, None)
    # make the output dir if needed
    stretch_stack_in_path = cfg_run['main']['out_path']# default to main output
    if 'stretch_stack_in_path' in cfg_run['estimate'].keys():
        stretch_stack_in_path = cfg_run['estimate']['stretch_stack_in_path']
    stretch_dir0 = os.path.join(
        stretch_stack_in_path, cfg_run['main']['orbit'])
    stretch_stack_flavor = cfg_run['stretch_stack']['flavor']# default to main output
    if 'stretch_stack_flavor' in cfg_run['estimate'].keys():
        stretch_stack_flavor = cfg_run['estimate']['stretch_stack_flavor']
    pekel_dir0 = None
    if 'pekel_in_path' in cfg_run['estimate'].keys():
        pekel_dir0 = os.path.join(
            cfg_run['estimate']['pekel_in_path'],cfg_run['main']['orbit'])
    out_path = cfg_run['main']['out_path']
    if 'out_path' in cfg_run['estimate'].keys():
        out_path = cfg_run['estimate']['out_path']
    outdir0 = os.path.join(out_path, cfg_run['main']['orbit'])
    #outdir = os.path.join(outdir0, cfg['main']['flavor'])
    stretch_files = []
    for stretch in stretch_list0:
        # get all reaches in basins smaller than stretch
        this_files = glob.glob(os.path.join(
            stretch_dir0,'{}*'.format(stretch),
            'stretch_stack_*', '{}*_stretch_stack.nc'.format(stretch)))
        stretch_files = stretch_files + this_files
    stretch_list = []
    for fle in stretch_files:
        head, tail = os.path.split(fle)
        stretch_list.append(tail.split('_')[0])
    df_stretches = pd.read_csv(
        cfg_run['main']['stretch_definition_file'],
        usecols=stretch_list)
    #if not os.path.exists(outdir):
    #    os.makedirs(outdir)
    # go through each stretch and process it
    N = len(df_stretches.keys())
    #breakpoint()
    if len(df_stretches.keys())==0:
        print('no files to process')
    #stretch_list = []
    for i,key in enumerate(df_stretches.keys()):
        this_start = time.time()
        stretch_reaches = np.array(
            df_stretches[df_stretches[key]>0][key]).astype(int)
        print("processing {} of {}, stretch: {}".format(
            i, N, key), ", Reaches:", stretch_reaches)
        #process_one_stretch(cfg_run, cfg_param, args.force,
        #    stretch_dir0, pekel_dir0, outdir0, key, stretch_stack_flavor)
        # create config
        this_cfg = rivscale.misc.CfgParser()
        #this_cfg.copy_sects(['main', 'estimate'], cfg_run)
        #this_cfg.abspaths()
        #this_cfg.write_sects(os.path.join(cfg_run['main']['out_path'],cfg_run['main']['out_path'],))
        try:
            stretch_flavor = cfg_run['estimate']['stretch_stack_flavor']
        except KeyError as e:
            # use the one from the stretch_stack section
            stretch_flavor = cfg_run['stretch_stack']['flavor']
        try:
            reach_flavor = cfg_run['estimate']['reach_flavor']
        except KeyError as e:
            # use the one from the reach_avg section
            reach_flavor = cfg_run['reach_avg']['flavor']
        try:
            pekel_flavor = cfg_run['estimate']['pekel_flavor']
        except KeyError as e:
            pekel_flavor = cfg_run['pekel']['flavor']
        #
        stretch_file = os.path.join(
            outdir0,
            f'{stretch}',
            f'stretch_stack_{stretch_flavor}',
            f'{key}_stretch_stack.nc')
        reach_path = os.path.join(
            outdir0,
            f'{stretch}',
            f'reach_avg_{reach_flavor}')
        this_outpath = os.path.join(
            outdir0,
            f'{stretch}',
            '{}'.format(cfg_run['estimate']['flavor']))
        pekel_file = os.path.join(
            outdir0,
            f'{stretch}',
            f'pekel_{pekel_flavor}',
            f'{key}_width_along_stats.nc')
        param_config_file = cfg_run['estimate']['param_config']
        cfg_str = '[main]\n'+ \
            'stretch_definition_file = {}\n'.format(
                cfg_run['main']['stretch_definition_file']) + \
            f'stretch_name = {key}\n' + \
            f'stretch_stack_file = {stretch_file}\n' + \
            f'reach_avg_path = {reach_path}\n' + \
            f'pekel_along_stats_file = {pekel_file}\n' + \
            f'out_path = {this_outpath}\n' + \
            f'param_config = {param_config_file}'
        this_cfg.read_string(cfg_str)
        this_cfg.abspaths()
        #this_cdf.write_sects(['main',])
        #breakpoint()
        # replace the stretch_subset with the single one
        
        #this_cfg['main']['stretch_subset'] = f'{key}'
        #this_cdf.write_sects(['main', 'estimate'])
        # create output dir if no exist and write the config
        if not os.path.exists(this_outpath):
            os.makedirs(this_outpath)
        this_cfg.write_sects(os.path.join(this_outpath,'run.cfg'),['main'])
        # set up logger
        both_logfile = os.path.join(this_outpath, 'log.txt')
        level = {'debug': logging.DEBUG, 'info': logging.INFO,
             'warning': logging.WARNING, 'error': logging.ERROR}[args.log_level]
        frmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        #if args.stdout:
        #    logging.basicConfig(level=level, format=format)
        #else:
        print('Logging all output to:', both_logfile)
        logging.basicConfig(
            filename=both_logfile ,level=level, format=frmt, filemode='w')
        # call worker
        worker = rivscale.processors.est_priors.Worker(this_cfg, args.force)
        success = worker.run()
        # TODO: logs gets clobbered if we dont reprocess
        #       should check and not clobber log if we dont rerun
        #       Also, should probably check to not clobber config
        #breakpoint()
        if not success:
            continue
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

