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
        if 'height_width' in fle:
            dic['height_width'] = \
                    rivscale.products.HeightWidthModel.from_ncfile(f)
    for key in dic.keys():
        # plot each individual plot
        if key=='height_width':
            wse_stretch_avg = None
            width_stretch_avg = None
            # plot also the strectch averages if they exist
            if ('wse_stretch_average' in dic.keys()) and (
                    'width_stretch_average' in dic.keys()):
                wse_stretch_avg = dic['wse_stretch_average']
                width_stretch_avg = dic['width_stretch_average']
            dic[key].plot(
                wse_stretch_avg=wse_stretch_avg,
                width_stretch_avg=width_stretch_avg,
                outdir=args.outdir,
                show=False)
        else:
            # single object plot
            dic[key].plot(outdir=args.outdir, show=False)
    
    if args.outdir is None:
        plt.show()

if __name__ == "__main__":
    main()

