#!/usr/bin/env python3
'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

'''

#import pandas as pd
import numpy as np
#import rivscale.plot
import matplotlib.pyplot as plt
import argparse
import configparser
import os.path
import glob
import pandas as pd
import rivscale.reconstruct
import rivscale.plot

import rivscale.products.stretch_stack
import rivscale.products.along_stretch
import rivscale.products.stretch_average
import rivscale.products.bayes_data
import rivscale.products.height_width
import rivscale.products.height_width_array
import rivscale.products.flow_state

import rivscale.processor
import rivscale.misc
import glob

import warnings

EXAMPLE = ''

def plot_single_stretch(files, outdir=None, cfg=None, width_correction=None):
    if len(files)<1:
        return
    dic = {}
    # plot each individual file
    for f in files:
        base, fle = os.path.split(f)
        if not os.path.exists(f):
            continue
        if 'stretch_stack' in fle:
            if 'stretch_stack_smoothwidth' in fle:
                dic['stretch_stack_smoothwidth'] = \
                    rivscale.products.stretch_stack.StretchStack.from_ncfile(f)
            elif 'stretch_stack_filt' in fle:
                dic['stretch_stack_filt'] = \
                    rivscale.products.stretch_stack.StretchStack.from_ncfile(f)
            elif 'stretch_stack_corr' in fle:
                dic['stretch_stack_corr'] = \
                    rivscale.products.stretch_stack.StretchStack.from_ncfile(f)
            else:
                dic['stretch_stack'] = \
                    rivscale.products.stretch_stack.StretchStack.from_ncfile(f)
        if 'wse_stats' in fle:
            dic['wse_stats'] = \
                    rivscale.products.along_stretch.AlongStretchStats.from_ncfile(f)
        if 'width_stats' in fle:
            dic['width_stats'] = \
                    rivscale.products.along_stretch.AlongStretchStats.from_ncfile(f)
        if 'dark_stats' in fle:
            dic['dark_stats'] = \
                    rivscale.products.along_stretch.AlongStretchStats.from_ncfile(f)
        if 'pekel_stats' in fle:
            dic['pekel_stats'] = \
                    rivscale.products.along_stretch.AlongStretchStats.from_ncfile(f)
        if ('wse_stretch_average' in fle) or ('wse_avg' in fle):
            dic['wse_stretch_average'] = \
                    rivscale.products.stretch_average.StretchAverageStats.from_ncfile(f)
        if ('width_stretch_average' in fle) or ('width_avg' in fle):
            dic['width_stretch_average'] = \
                    rivscale.products.stretch_average.StretchAverageStats.from_ncfile(f)
        if 'wse_reach_average' in fle:
            dic['wse_reach_average'] = \
                    rivscale.products.stretch_average.StretchAverageStats.from_ncfile(f)
        if 'width_reach_average' in fle:
            if 'height_width' in fle:
                if 'height_width_reach_average_corr' in fle:
                    dic['height_width_reach_average_corr'] = \
                        rivscale.products.height_width.HeightWidthModel.from_ncfile(f)
                else:
                    dic['height_width_reach_average'] = \
                        rivscale.products.height_width.HeightWidthModel.from_ncfile(f)
            else:
                if 'width_reach_average_corr' in fle:
                    dic['width_reach_average_corr'] = \
                        rivscale.products.stretch_average.StretchAverageStats.from_ncfile(f)
                else:
                    dic['width_reach_average'] = \
                        rivscale.products.stretch_average.StretchAverageStats.from_ncfile(f)
        if 'flow_state' in fle:
            dic['flow_state'] = \
                    rivscale.products.flow_state.FlowStateModel.from_ncfile(f)
        if 'height_width' in fle:
            if 'array' in fle:
                # plot the height_width_array object
                dic['height_width_array'] = \
                        rivscale.products.height_width_array.HeightWidthModelArray.from_ncfile(f)
            else:
                if 'reach_average' in fle:
                    pass
                    #dic['height_width_reach_average'] = \
                    #    rivscale.products.HeightWidthModel.from_ncfile(f)
                else:
                    dic['height_width'] = \
                        rivscale.products.height_width.HeightWidthModel.from_ncfile(f)
        if 'bayes' in fle:
            dic['bayes'] = \
                    rivscale.products.bayes_data.BayesData.from_ncfile(f)
    rivscale.plot.plot_products(dic, width_correction=width_correction, outdir=outdir)

def get_stretch_list(stretch_list_in, dir_in, kind='stretch_stack'):
    stretch_files = []
    for stretch in stretch_list_in:
        # get all reaches in basins smaller than stretch
        kind2 = kind
        if kind=='reach_avg':
            kind2='reach_average'
        glob_str = os.path.join(
            dir_in,'{}*'.format(stretch),
            '{}_*'.format(kind), '{}*_{}.nc'.format(stretch, kind2))
        print(glob_str)
        this_files = glob.glob(glob_str)
        stretch_files = stretch_files + this_files
    stretch_list = []
    for fle in stretch_files:
        head, tail = os.path.split(fle)
        stretch_list.append(tail.split('_')[0])
    return stretch_list

def setup_from_cfg(cfg):
    stretch_list = rivscale.misc.get_stretch_list_from_subset_cfg(
        cfg, None)
    df_stretches = pd.read_csv(
        cfg['main']['stretch_definition_file'],
        usecols=stretch_list)
    return df_stretches

def main():
    parser = argparse.ArgumentParser(
        description='Plot river stretch, reach, or multireach',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('-c','--config', default=None,
        help='processing.cfg or river_avg.cfg')
    parser.add_argument('--infile', nargs='+', help='input file(s)')
    #parser.add_argument('-t','--filetype', type=str, default='StretchData',
    #    help='StreachData, AlongStretchStats')
    #parser.add_argument('-o','--outdir', default=None,
    #    help='output directory to save plots')
    parser.add_argument('-s','--stretch_name', nargs='+', default=None,
        help='stretch_name(s) to plot')
    parser.add_argument('--width_correction', default=None,
        help='csv input file with width vs cross-track correction to apply')
    parser.add_argument('--kind', default='all',
        help='all, reach, stretch, estimate, or reconstruct')
    args = parser.parse_args()
    #breakpoint()
    if args.infile is not None:
        # just plot the specific file, and show it (not saving it)
        plot_single_stretch(args.infile, outdir=None, cfg=None,
            width_correction=None)
        plt.show()
        return
    #cfg = configparser.ConfigParser()
    #cfg.read(args.config)
    cfg = rivscale.misc.CfgParser()
    cfg.read(args.config)
    df_stretches = setup_from_cfg(cfg)
    if len(df_stretches.keys())==0:
        print('no files to process')
    all_stretches = list(df_stretches.keys())
    #breakpoint()
    warnings.filterwarnings("ignore")
    if args.stretch_name is not None:
        all_stretches = list(args.stretch_name)
    for i,key in enumerate(all_stretches):
        print('plotting stretch: {}'.format(key))
        kinds = [args.kind,]
        if args.kind=='all':
            # get all the minor section headings
            cfg_run = args.config
            if isinstance(cfg_run,str):
                infile = cfg_run
                cfg_run = rivscale.misc.CfgParser()
                cfg_run.read(infile)
            
            kinds = [k for k in cfg_run.keys()]
            kinds = list(set(kinds) - set(['DEFAULT', 'main',]))
            #
            myout_path = cfg_run['main']['out_path']
        for kind in kinds:
            print('kind:', kind)
            plotdir = None
            # use the processor class to get all the output files
            if not(kind in ['estimate', 'reconstruct']):
                # use the processor class to get all the output files
                this_cfg = rivscale.estimate.make_single_stretch_config(
                    args.config, key, section='estimate')
                if kind=='reach_avg':
                    thepath = this_cfg['main']['reach_avg_path']
                    files = glob.glob(f'{thepath}/*.nc')
                if kind=='pekel':
                    pekel_file = this_cfg['main']['pekel_along_stats_file']
                    thepath,_ = os.path.split(pekel_file)
                    files = [pekel_file,]
                if kind=='stretch_stack':
                    stretch_file = this_cfg['main']['stretch_stack_file']
                    thepath,_ = os.path.split(stretch_file)
                    files = [stretch_file,]
                plotdir = os.path.join(thepath,'plots')
                #
                #breakpoint()
                #
                plot_single_stretch(files, outdir=plotdir, cfg=None,
                    width_correction=None)
                #plt.show()
                #breakpoint()
            else:
                # use the processor class to get all the output files
                this_cfg = rivscale.estimate.make_single_stretch_config(
                    args.config, key, section=kind)
                this_outpath = this_cfg['main']['out_path']
                # initialize the worker
                worker = rivscale.processor.Processor(this_cfg,
                    kind=kind,
                    stretch_name='{}'.format(key),
                    log_level='info',
                    log_file=None,
                    force=False)
                #plotdir = None
                if args.stretch_name is None:
                    plotdir = os.path.join(this_outpath, 'plots')
                # now plot the rest
                try:
                    # TODO: handle --force
                    #breakpoint()
                    worker.plot(plotdir)
                    #plot_single_stretch(files, outdir=plotdir, cfg=cfg,
                    #    width_correction=args.width_correction)
                except Exception as e:
                    print('problem plotting')
        #
        if plotdir is None:
            plt.show()
        plt.close('all')

if __name__ == "__main__":
    main()

