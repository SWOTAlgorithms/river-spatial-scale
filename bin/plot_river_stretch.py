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
import os.path
EXAMPLE = ''

def main():
    parser = argparse.ArgumentParser(
        description='Plot river stretch, reach, or multireach',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('infile', nargs='+', help='input file(s)')
    #parser.add_argument('-t','--filetype', type=str, default='StretchData',
    #    help='StreachData, AlongStretchStats')
    parser.add_argument('-o','--outdir', default=None,
        help='output directory to save plots')
    args = parser.parse_args()
    #breakpoint()
    """
    filetype = []
    # guess type from file names
    # TODO: check that all are form the same stretch?
    for fle in args.infile:
        head, tail = os.path.split(fle)
        #parts = tail.split('.nc')[0].split('_')
        #kind = ''
        #for part in parts[1:]:
        #    kind = kind+'_'+part
        if 'stretch_stack' in fle:
            filetype.append('StretchStack')
        elif 'stats' in fle:
            filetype.append('AlongStretchStats')
        elif 'stretch_average' in fle:
            filetype.append('StretchAverageStats')
        elif 'height_width' in fle:
            filetype.append('HeightWidthModel')
    """
    dic = {}
    # plot each individual file
    for f in args.infile:
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
                    outdir=args.outdir,
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
                    outdir=args.outdir,
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
                    outdir=args.outdir,
                    show=False,
                    title_tag='Bayes node estimates')
                something_plotted=True
            if not something_plotted:
                # plot just the h/w fit
                dic[key].plot(
                    outdir=args.outdir,
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
                outdir=args.outdir,
                show=False,
                title_tag=title_tag)

        else:
            # single object plot
            if 'reach' not in key:
                # dont plot the time series of reach data
                dic[key].plot(outdir=args.outdir, show=False)
    
    if args.outdir is None:
        plt.show()

if __name__ == "__main__":
    main()

