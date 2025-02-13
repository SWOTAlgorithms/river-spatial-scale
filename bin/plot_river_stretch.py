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
    parser.add_argument('infile', help='*_streach.nc file')
    parser.add_argument('-t','--filetype', type=str, default='StretchData',
        help='StreachData, AlongStretchStats')
    parser.add_argument('-o','--outdir', default=None, help='output directory to save plots')
    args = parser.parse_args()
    #breakpoint()
    #
    if args.filetype=='StretchData':
        data = rivscale.products.StretchData.from_ncfile(
            args.infile)
    if args.filetype=='AlongStretchStats':
        data = rivscale.products.AlongStretchStats.from_ncfile(
            args.infile)
    #name = os.path.split(args.infile)[1].split('_')[0]
    #breakpoint()
    #rivscale.plot.plot_stretch(stretch_data, title=name, outdir=args.outdir)
    data.plot(outdir=args.outdir)
    if args.outdir is None:
        plt.show()

if __name__ == "__main__":
    main()

