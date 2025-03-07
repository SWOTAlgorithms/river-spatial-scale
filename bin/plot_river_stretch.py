#!/usr/bin/env python
'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

'''

#import pandas as pd
import numpy as np
#import rivscale.plot
import rivscale.products
import matplotlib.pyplot as plt
import argparse
import configparser
import os.path
import glob
import pandas as pd
import rivscale.reconstruct

EXAMPLE = ''

def plot_single_stretch(files, outdir=None, cfg=None):
    dic = {}
    # plot each individual file
    for f in files:
        base, fle = os.path.split(f)
        if 'stretch_stack' in fle:
            dic['stretch_stack'] = \
                    rivscale.products.StretchStack.from_ncfile(f)
        if 'wse_stats' in fle:
            dic['wse_stats'] = \
                    rivscale.products.AlongStretchStats.from_ncfile(f)
        if 'width_stats' in fle:
            dic['width_stats'] = \
                    rivscale.products.AlongStretchStats.from_ncfile(f)
        if 'pekel_stats' in fle:
            dic['pekel_stats'] = \
                    rivscale.products.AlongStretchStats.from_ncfile(f)
        if 'wse_stretch_average' in fle:
            dic['wse_stretch_average'] = \
                    rivscale.products.StretchAverageStats.from_ncfile(f)
        if 'width_stretch_average' in fle:
            dic['width_stretch_average'] = \
                    rivscale.products.StretchAverageStats.from_ncfile(f)
        if 'wse_reach_average' in fle:
            dic['wse_reach_average'] = \
                    rivscale.products.StretchAverageStats.from_ncfile(f)
        if 'width_reach_average' in fle:
            dic['width_reach_average'] = \
                    rivscale.products.StretchAverageStats.from_ncfile(f)
        if 'height_width' in fle:
            if 'reach_average' in fle:
                dic['height_width_reach_average'] = \
                    rivscale.products.HeightWidthModel.from_ncfile(f)
            else:
                dic['height_width'] = \
                    rivscale.products.HeightWidthModel.from_ncfile(f)
        if 'bayes' in fle:
            dic['bayes'] = \
                    rivscale.products.BayesData.from_ncfile(f)
    for key in dic.keys():
        # plot each individual plot
        if key=='height_width':
            something_plotted=False
            # plot the stretch averages if they exist
            if ('wse_stretch_average' in dic.keys()) and (
                    'width_stretch_average' in dic.keys()):
                wse_data = dic['wse_stretch_average']
                width_data = dic['width_stretch_average']
                dic[key].plot(
                    wse_data=wse_data,
                    width_data=width_data,
                    outdir=outdir,
                    show=False,
                    title_tag='stretch average data')
                something_plotted=True
            if (('stretch_stack' in dic.keys()) and (
                    'wse_stats' in dic.keys()) and (
                        'width_stats')):
                # plot the noisy node data
                stretch_stack = dic['stretch_stack']
                wse_stats = dic['wse_stats']
                width_stats = dic['width_stats']
                wse = stretch_stack['wse']
                width = stretch_stack['width']
                ref2 = np.broadcast_to(
                    wse_stats.reference, np.shape(wse.T)).T
                ref2_w = np.broadcast_to(
                    width_stats.reference, np.shape(width.T)).T
                dic[key].plot(
                    wse_data=wse - ref2,
                    width_data=width - ref2_w,
                    outdir=outdir,
                    show=False, 
                    title_tag='node measurements')
                something_plotted=True
            if 'bayes' in dic.keys():
                # plot bayes node data
                bayes = dic['bayes']
                wse_bayes, width_bayes, postcov = bayes.unpack_joint()
                wse = wse_bayes['signal']
                width = width_bayes['signal']
                ref2 = np.broadcast_to(
                    wse_bayes.signal_mean, np.shape(wse.T)).T
                ref2_w = np.broadcast_to(
                    width_bayes.signal_mean, np.shape(width.T)).T
                d_wse = wse - ref2
                d_width = width - ref2_w
                dic[key].plot(
                    wse_data=d_wse,
                    width_data=d_width,
                    outdir=outdir,
                    show=False,
                    title_tag='Bayes node estimates')
                something_plotted=True
            if not something_plotted:
                # plot just the h/w fit
                dic[key].plot(
                    outdir=outdir,
                    show=False)
        elif key=='height_width_reach_average':
            wse_data = None
            width_data = None
            title_tag = None
            if ('wse_reach_average' in dic.keys()) and (
                    'width_reach_average' in dic.keys()):
                wse_data = dic['wse_reach_average']
                width_data = dic['width_reach_average']
                title_tag='reach average data'
            dic[key].plot(
                wse_data=wse_data,
                width_data=width_data,
                outdir=outdir,
                show=False,
                title_tag=title_tag)

        else:
            # single object plot
            #if 'reach' not in key:
            #    # dont plot the time series of reach data
            dic[key].plot(outdir=outdir, show=False)
    # optionally plot the outliers?
    if cfg is not None:
        try:
            stretch_stack = rivscale.reconstruct.reconstruct_filter_data(
                cfg['reconstruct'],
                dic['stretch_stack'].copy(),
                dic['wse_stats'],
                dic['pekel_stats'],
                plot=True)
        except:
            print('could not plot outliers')
    #if outdir is None:
    #    plt.show()

def setup_from_cfg(cfg):
    """

    """
    stretch_list0 = [
        '{}'.format(t) for t in '{}'.format(
            cfg['main']['stretch_subset']).split()]
    # make the output dir if needed
    stretch_dir0 = os.path.join(
        cfg['main']['stretch_stack_in_path'],cfg['main']['orbit'])
    pekel_dir0 = os.path.join(
        cfg['main']['pekel_in_path'],cfg['main']['orbit'])
    outdir0 = os.path.join(cfg['main']['out_path'],cfg['main']['orbit'])
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
        cfg['main']['stretch_file'],
        usecols=stretch_list)
    return df_stretches, stretch_dir0, pekel_dir0, outdir0

def main():
    parser = argparse.ArgumentParser(
        description='Plot river stretch, reach, or multireach',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('-c','--config', default=None,
        help='output directory to save plots')
    parser.add_argument('--infile', nargs='+', help='input file(s)')
    #parser.add_argument('-t','--filetype', type=str, default='StretchData',
    #    help='StreachData, AlongStretchStats')
    #parser.add_argument('-o','--outdir', default=None,
    #    help='output directory to save plots')
    parser.add_argument('-s','--stretch_name', nargs='+', default=None,
        help='stretch_name(s) to plot')
    args = parser.parse_args()
    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    df_stretches, stretch_dir0, pekel_dir0, outdir0 = setup_from_cfg(cfg)
    if len(df_stretches.keys())==0:
        print('no files to process')
    all_stretches = list(df_stretches.keys())
    #breakpoint()
    if args.stretch_name is not None:
        all_stretches = list(args.stretch_name)
    for i,key in enumerate(all_stretches):
        stretch_dir = os.path.join(
            stretch_dir0, key, 'stretch_stack_{}'.format(
                cfg['main']['stretch_stack_flavor']))
        pekel_dir = os.path.join(
            pekel_dir0, key, 'pekel_{}'.format(
                cfg['main']['pekel_flavor']))
        outdir = os.path.join(outdir0, key, cfg['main']['flavor'])

        file_stretch = os.path.join(
            stretch_dir, '{}_stretch_stack.nc'.format(key))
        file_pekel = os.path.join(
            pekel_dir, '{}_pekel_stats.nc'.format(key))
        file_wse_stats = os.path.join(
            outdir, '{}_wse_stats.nc'.format(key))
        file_width_stats = os.path.join(
            outdir, '{}_width_stats.nc'.format(key))
        file_wse_avg = os.path.join(
            outdir, '{}_wse_stretch_average.nc'.format(key))
        file_width_avg = os.path.join(
            outdir, '{}_width_stretch_average.nc'.format(key))
        file_height_width = os.path.join(
            outdir, '{}_height_width.nc'.format(key))
        file_bayes = os.path.join(
            outdir, '{}_bayes.nc'.format(key))
        files = []
        if os.path.exists(file_stretch):
            files.append(file_stretch)
        if os.path.exists(file_pekel):
            files.append(file_pekel)
        if os.path.exists(file_wse_stats):
            files.append(file_wse_stats)
        if os.path.exists(file_width_stats):
            files.append(file_width_stats)
        if os.path.exists(file_wse_avg):
            files.append(file_wse_avg)
        if os.path.exists(file_width_avg):
            files.append(file_width_avg)
        if os.path.exists(file_height_width):
            files.append(file_height_width)
        if os.path.exists(file_bayes):
            files.append(file_bayes)
        #breakpoint()
        # plot the stretch
        #if os.path.exists(file_stretch):
        #    plot_single_stretch([file_stretch], outdir=None)
        # TODO: try to plot the reach one guessing the name?
        # plot the pekel
        #if os.path.exists(file_pekel):
        #    plot_single_stretch([file_pekel,], outdir=None)
        # now plot the rest
        plot_single_stretch(files, outdir=None, cfg=cfg)
        #if outdir is None:
        plt.show()

if __name__ == "__main__":
    main()

