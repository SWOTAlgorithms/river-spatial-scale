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

from rivscale.misc import swot_time_to_field_time
from rivscale.plot import label_units

import glob


EXAMPLE = ''


def make_movie_from_images(image_file_list, video_file):
    import cv2
    video = None
    for image_file in image_file_list:
        if video is None:
            frame = cv2.imread(image_file)
            # setting the frame width, height width
            # the width, height of first image
            height, width, layers = frame.shape
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_file, fourcc, 1.5, (width, height))

        # Appending the images to the video one by one
        video.write(cv2.imread(image_file))
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated

def plot_map(lat, lon, c=None, s=None, title='', clabel=None,
        clim=None, alpha=0.7, zoom=14, figsize=(12,10), cmap='jet',
        outdir=None, show=False):
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
        alpha=alpha, cmap=cmap, clim=clim,
        transform=ccrs.Geodetic())
    
    fig.colorbar(scat, label=clabel)
    ax.add_artist(ScaleBar(1, location='lower right'))
    plt.title(title)
    plt.tight_layout()
    if outdir is not None:
        # create output dir if not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = '{}.png'.format(
            title.replace(',','').replace(' ', '_').replace(':','').replace(
                '-','_').replace('$\Delta$','delta'))
        plt.savefig(os.path.join(outdir, fname), dpi=300)
        plt.close()
    else:
        if show:
            plt.show()

def plot_multitemporal(stretch_stack, wse_along_stats, width_along_stats,
        clim = (-3,3), outdir='plots_to_delete'):
    title0 = wse_along_stats.stretch_name
    M = len(wse_along_stats.reaches)
    #title = title + 'pass: {}, date: {}'.format()
    # determine zoom
    zoom = 14
    if M >3:
        zoom = 12
    if M > 5:
        zoom = 10
    # set the variables
    #breakpoint()
    lat = stretch_stack.p_lat
    lon = stretch_stack.p_lon
    s = width_along_stats.reference
    clabel = '$\Delta$ wse'
    #clim = (-3,3)
    #outdir='plots_to_delete'
    for k in range(len(stretch_stack.time_id)):
        c = stretch_stack.wse[:,k] - wse_along_stats.reference
        cyc, pas, _ = stretch_stack.granule_id[k].split('_')
        time_id = stretch_stack.time_id[k]*60.0*60.0
        #breakpoint()
        title = title0+ ' '+ clabel + ', date: {}, pass: {}'.format(
            swot_time_to_field_time([time_id,])[0].date(), pas)
        # now plot it
        plot_map(
            lat, lon,
            c=c, clabel=label_units(clabel), clim=clim,
            s=s,
            title=title, zoom=zoom,
            outdir=outdir)
    # make the movie
    image_file_list = np.sort(glob.glob(os.path.join(outdir,'*.png')))
    pth, fle = os.path.split(image_file_list[0])
    mp4_file = fle.split('_date')[0]+'.mp4'
    video_file = os.path.join(outdir,mp4_file)
    make_movie_from_images(image_file_list, video_file)


def plot_along_stats(along_stats, width_along_stats=None,
        clim=None, outdir=None, stat='reference'):
    title0 = along_stats.stretch_name
    M = len(along_stats.reaches)
    #title = title + 'pass: {}, date: {}'.format()
    # determine zoom
    zoom = 14
    if M >3:
        zoom = 12
    if M > 5:
        zoom = 10
    # set the variables
    lat = along_stats.p_lat
    lon = along_stats.p_lon
    if width_along_stats is not None:
        s = width_along_stats.reference
    else:
        s = 100 
    clabel = along_stats.signal_key#'$\Delta$ wse'
    #clim = (-3,3)
    #outdir='along_stats_map_plots'
    
    #c = along_stats.reference
    if isinstance(stat, float):
        # assume it is a particular percentile
        # find index of closest
        ind = np.argmin(np.abs(along_stats.percentile_list - stat))
        c = along_stats.percentiles[:,ind].squeeze()
        this_stat = along_stats.percentile_list[ind]
        clabel = clabel + ' ({} %ile)'.format(this_stat)
        #breakpoint()
    else:
        c = along_stats[stat]
        clabel = clabel + ' ({})'.format(stat)
    #cyc, pas, _ = stretch_stack.granule_id[k].split('_')
    title = title0+ ' '+ clabel
    #breakpoint()
    plot_map(
            lat, lon,
            c=c, clabel=label_units(clabel), clim=clim,
            s=s,
            title=title, zoom=zoom,
            outdir=outdir)
   
    #plt.show()
    #breakpoint()

def main():
    """
    plot multitemporal objects over satellite maps to produce QGIS-like figures
    """
    parser = argparse.ArgumentParser(
        description='Plot along-river stretch on map over google-image',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('along_stats_file', nargs='+', default=None,
        help='along-river multitemproal stats file(s)')
    parser.add_argument('--stretch_stack_file', default=None,
        help='in input this will make a movie of plots for each time step')
    """
    parser.add_argument('--wse_along_stats_file', default=None,
        help='processing config')
    parser.add_argument('--width_along_stats_file', default=None,
        help='processing config')
    parser.add_argument('--dark_along_stats_file', default=None,
        help='processing config')
    """
    parser.add_argument('--crop_reach',default=False, action='store_true')
    #parser.add_argument('--movie',default=False, action='store_true',
    #        help='option to make movie (e.g., a plot for every time step) this option requires the stretch stack option be set')
    args = parser.parse_args()
    
    # read in the files
    stretch_stack = None
    wse_along_stats = None
    width_along_stats = None
    #dark_along_stats = None
    #breakpoint()
    along_stats = []
    for fle in args.along_stats_file:
        this_stats = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(fle)
        along_stats.append(this_stats)
    if args.stretch_stack_file is not None:
        stretch_stack = rivscale.products.stretch_stack.StretchStack.from_ncfile(
            args.stretch_stack_file)
    """
    if args.wse_along_stats_file is not None:
        wse_along_stats = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
            args.wse_along_stats_file)
    
    if args.width_along_stats_file is not None:
        width_along_stats = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
            args.width_along_stats_file)
    if args.dark_along_stats_file is not None:
        dark_along_stats = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
            args.dark_along_stats_file)
    """
    #
    #breakpoint()
    # crop to center reach if commanded
    if args.crop_reach:
        #wse_along_stats = wse_along_stats.crop_to_reach()
        #width_along_stats = width_along_stats.crop_to_reach()
        for k,this_stats in enumerate(along_stats):
            along_stats[k] = this_stats.crop_to_reach()
        if stretch_stack is not None:
            stretch_stack = stretch_stack.crop_to_reach()
    #breakpoint()
    outdir='along_stats_map_plots'
    # plot each along-stats
    for this_stats in along_stats:
        stat = 'reference'
        if this_stats.signal_key == 'dark_frac':
            # plot the ~80 #ile
            stat = 80.0
        plot_along_stats(this_stats, stat=stat, outdir=outdir)
    if outdir is not None:
        plt.show()
    # plot the movie if commanded
    if stretch_stack is not None:

        # make sure that the wse and width along stats are also input
        for k,this_stats in enumerate(along_stats):
            if this_stats.signal_key=='wse':
                wse_along_stats = this_stats
            if this_stats.signal_key=='width':
                width_along_stats = this_stats
        plot_multitemporal(stretch_stack, wse_along_stats, width_along_stats)
    #

if __name__ == '__main__':
    main()



