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

import rivscale.estimator
import shutil


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
        '-l', '--log-level', type=str, default="info",#default="debug",
        help="logging level, one of: debug info warning error")
    #
    args = parser.parse_args()
    # read in the config file
    cfg_run = rivscale.misc.CfgParser()
    cfg_run.read(args.config)
    ####
    # get the list of stretches to process
    ###
    stretch_list0 = rivscale.misc.get_stretch_list_from_subset_cfg(
        cfg_run, None)
    # handle basin-level stretch_list
    isbasin = False
    for stretch in stretch_list0:
        if len(stretch)< 11:
            isbasin = True
    if isbasin:
        stretch_list = []
        this_df = pd.read_csv(
            cfg_run['main']['stretch_definition_file'])
        reach_ids = this_df.keys()
        for stretch in stretch_list0:
            msk = [False for rid in reach_ids] # start with all False
            for k, rid in enumerate(reach_ids):
                if rid.startswith(stretch):
                    msk[k] = True
            these_stretches = list(reach_ids[msk])
            stretch_list = stretch_list + these_stretches
    else:
        stretch_list = stretch_list0
    print(f'preparing to process stretches: {stretch_list}')
    # get the stretch_definition rows for the stretch_list
    df_stretches = pd.read_csv(
        cfg_run['main']['stretch_definition_file'],
        usecols=stretch_list)
    # check if there are no stretches to process
    N = len(df_stretches.keys())
    if len(df_stretches.keys())==0:
        print('no files to process')
    # go through each stretch and process
    # TODO: handle multi-processing
    for i,key in enumerate(df_stretches.keys()):
        this_start = time.time()
        stretch_reaches = np.array(
            df_stretches[df_stretches[key]>0][key]).astype(int)
        print("processing {} of {}, stretch: {}".format(
            i, N, key), ", Reaches:", stretch_reaches)
        ####
        # create a one-stretch config
        ####
        this_cfg = rivscale.estimate.make_single_stretch_config(cfg_run, key)
        this_outpath = this_cfg['main']['out_path']
        if not os.path.exists(this_outpath):
            os.makedirs(this_outpath)
        log_file_tmp = os.path.join(this_outpath, 'log_tmp.txt')
        log_file = os.path.join(this_outpath, 'log.txt')
        # initialize the worker
        worker = rivscale.estimator.Estimator(this_cfg,
            stretch_name='{}'.format(key),
            log_level=args.log_level,
            log_file=log_file_tmp,
            force=args.force)
        # check if already run
        if worker.need_to_run():
            # write config if we are processing it
            this_cfg.write_sects(os.path.join(this_outpath,'run.cfg'),['main'])
            print('Logging all output to:', log_file)
            try:
                success = worker.run()
                # only squash the final log file if we rerun it
                shutil.move(log_file_tmp, log_file)
            except Exception as e:
                print(f'problem processing {e}')
        else:
            print('    already processed, not reruning')
        this_stop = time.time()
        print('  execution time: {:2.2f} seconds'.format(this_stop - this_start))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

