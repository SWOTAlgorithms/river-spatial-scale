#!/usr/bin/env python
'''
Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

This code plots the stretch data overlaid on google-map images
'''

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles

import os.path
import rivscale.products.along_stretch

import geopandas as gpd
import numpy as np
import argparse

EXAMPLE = ''

"""
def plot_map(along_file, crop_reach=False, clabel='width IQR (m)', figsize=(12,10)):
    # read in the along-stats
    along_stats = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
        along_file)
    # crop to center reach if commanded
    if crop_reach:
        along_stats = along_stats.crop_to_reach()
    title = along_stats.stretch_name
    M = len(along_stats.reaches)
    if M>1:
        title = title + '\n reaches: ['
        #title = title+' {}'.format(along_stats.reaches)
        cnt=0
        max_cnt=3
        for k,reach in enumerate(along_stats.reaches):
            if cnt==max_cnt:
                cnt=0
                title = title + "\n"
            if k == M-1:
                title = title+'{}]'.format(reach)
            else:
                title = title+'{}, '.format(reach)
            cnt = cnt + 1
         
    p25 = along_stats.percentiles[:,along_stats.percentile_list == 25].squeeze()
    p75 = along_stats.percentiles[:,along_stats.percentile_list == 75].squeeze()
    IQR = p75-p25
    
    # define the bounding box
    buff = 0.01 #
    bbox = [
        np.nanmin(along_stats.p_lon) - buff,
        np.nanmax(along_stats.p_lon) + buff,
        np.nanmin(along_stats.p_lat) - buff,
        np.nanmax(along_stats.p_lat) + buff]
"""
def plot_map(lat, lon, c=None, s=None, title='', clabel=None,
        alpha=0.7, zoom=14, figsize=(12,10), cmap='jet'):
    # define the bounding box
    buff = 0.01 #
    bbox = [
        np.nanmin(lon) - buff,
        np.nanmax(lon) + buff,
        np.nanmin(lat) - buff,
        np.nanmax(lat) + buff]
    # plot the google image
    tiler = GoogleTiles(style="satellite")
    mercator = tiler.crs
    fig = plt.figure(figsize=figsize)
    #ax = fig.add_subplot(1, 1, 1, projection=mercator)
    ax = plt.axes(projection=mercator)
    ax.set_extent(bbox, crs=ccrs.Geodetic())#ccrs.PlateCarree())
    ax.add_image(tiler, zoom)
    #ax.coastlines('10m')
    # plot the node scatterplot data
    scat = ax.scatter(lon, lat, c=c, s=s,
        alpha=alpha, cmap=cmap, transform=ccrs.Geodetic())
    
    fig.colorbar(scat, label=clabel)
    ax.add_artist(ScaleBar(1, location='lower right'))
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    """
    parser = argparse.ArgumentParser(
        description='Plot river stretch, reach, or multireach',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('-c','--config', default=None,
        help='processing.cfg or river_avg.cfg')
    parser.add_argument('--infile', nargs='+', help='input file(s)')
    parser.add_argument('-s','--stretch_name', nargs='+', default=None,
        help='stretch_name(s) to plot')
    args = parser.parse_args()
    cfg = rivscale.misc.CfgParser()
    cfg.read(args.config)
    df_stretches, stretch_dir0, pekel_dir0, outdir0 = setup_from_cfg(cfg)
    if len(df_stretches.keys())==0:
        print('no files to process')
    all_stretches = list(df_stretches.keys())
    if args.stretch_name is not None:
        all_stretches = list(args.stretch_name)
    
    """
    parser = argparse.ArgumentParser(
        description='Plot along-river stretch on map over google-image',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('infile', default=None,
        help='processing config')
    parser.add_argument('--crop_reach',default=False, action='store_true')
    args = parser.parse_args()
   
    # read in the along-stats
    along_stats = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
        args.infile)
    # crop to center reach if commanded
    if args.crop_reach:
        along_stats = along_stats.crop_to_reach()
    title = along_stats.stretch_name
    M = len(along_stats.reaches)
    if M>1:
        title = title + '\n reaches: ['
        #title = title+' {}'.format(along_stats.reaches)
        cnt=0
        max_cnt=3
        for k,reach in enumerate(along_stats.reaches):
            if cnt==max_cnt:
                cnt=0
                title = title + "\n"
            if k == M-1:
                title = title+'{}]'.format(reach)
            else:
                title = title+'{}, '.format(reach)
            cnt = cnt + 1
         
    p25 = along_stats.percentiles[:,along_stats.percentile_list == 25].squeeze()
    p75 = along_stats.percentiles[:,along_stats.percentile_list == 75].squeeze()
    IQR = p75-p25
    clabel='width IQR (m)'
    zoom = 14
    if M >3:
        zoom = 12
    if M > 5:
        zoom = 10
    # now plot it
    plot_map(
        along_stats.p_lat, along_stats.p_lon,
        c=IQR, clabel=clabel,
        s=along_stats.reference,
        title=title, zoom=zoom)

if __name__ == '__main__':
    main()



