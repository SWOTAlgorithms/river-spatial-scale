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
    
    dic['stretch_stack'].filter_node_outliers(
        dic['wse_stats'], key='wse',plot=True)
    plt.title('wse outliers (x) using multitemporal wse stats')
    dic['stretch_stack'].filter_node_outliers(
        dic['width_stats'], key='width',plot=True)
    plt.title('width outliers (x) using multitemporal width stats')
    if 'pekel_stats' in dic.keys():
        dic['stretch_stack'].filter_node_outliers(
            dic['pekel_stats'], key='width',plot=True)
        plt.title('width outliers (x) using pekel width stats')
    if args.outdir is None:
        plt.show()

if __name__ == "__main__":
    main()

