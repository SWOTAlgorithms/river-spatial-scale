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
        ####
        # create a one-stretch config
        ####
        this_cfg = rivscale.estimate.make_single_stretch_config(cfg_run, key)
        # initialize the worker
        worker = rivscale.processors.est_priors.Worker(this_cfg, args.force)
        # check if already run
        if worker.need_to_run():
            # write config
            this_outpath = this_cfg['main']['out_path']
            if not os.path.exists(this_outpath):
                os.makedirs(this_outpath)
            this_cfg.write_sects(os.path.join(this_outpath,'run.cfg'),['main'])
            # set up logger
            both_logfile = os.path.join(this_outpath, 'log.txt')
            level = {'debug': logging.DEBUG, 'info': logging.INFO,
                'warning': logging.WARNING, 'error': logging.ERROR}[args.log_level]
            frmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            print('Logging all output to:', both_logfile)
            logging.basicConfig(
                filename=both_logfile ,level=level, format=frmt, filemode='w')
            # run if needed
            success = worker.run()
        
            #if not success:
            #    continue
        else:
            print('    already processed, not reruning')
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

