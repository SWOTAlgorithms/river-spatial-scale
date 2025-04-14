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

from errtools.misc import swot_time_to_field_time
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
    parser.add_argument('stretch_stack_file', default=None,
        help='processing config')
    parser.add_argument('wse_along_stats_file', default=None,
        help='processing config')
    parser.add_argument('width_along_stats_file', default=None,
        help='processing config')
    parser.add_argument('--crop_reach',default=False, action='store_true')
    args = parser.parse_args()
   
    # read in the files
    stretch_stack = rivscale.products.stretch_stack.StretchStack.from_ncfile(
        args.stretch_stack_file)
    wse_along_stats = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
        args.wse_along_stats_file)
    width_along_stats = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
        args.width_along_stats_file)
    #breakpoint()
    # crop to center reach if commanded
    if args.crop_reach:
        wse_along_stats = wse_along_stats.crop_to_reach()
        width_along_stats = width_along_stats.crop_to_reach()
        stretch_stack = stretch_stack.crop_to_reach()
    #breakpoint()
    title0 = wse_along_stats.stretch_name
    M = len(wse_along_stats.reaches)
    #title = title + 'pass: {}, date: {}'.format()
    """
    # print all reaches in title?
    if M>1:
        title = title + '\n reaches: ['
        #title = title+' {}'.format(along_stats.reaches)
        cnt=0
        max_cnt=3
        for k,reach in enumerate(wse_along_stats.reaches):
            if cnt==max_cnt:
                cnt=0
                title = title + "\n"
            if k == M-1:
                title = title+'{}]'.format(reach)
            else:
                title = title+'{}, '.format(reach)
            cnt = cnt + 1
    
    # get teh IQR     
    p25 = wse_along_stats.percentiles[:,wse_along_stats.percentile_list == 25].squeeze()
    p75 = wse_along_stats.percentiles[:,wse_along_stats.percentile_list == 75].squeeze()
    IQR = p75-p25
    clabel='width IQR (m)'
    """
    # determine zoom
    zoom = 14
    if M >3:
        zoom = 12
    if M > 5:
        zoom = 10
    # set the variables
    lat = stretch_stack.p_lat
    lon = stretch_stack.p_lon
    s = width_along_stats.reference
    clabel = '$\Delta$ wse'
    clim = (-3,3)
    outdir='plots_to_delete'
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

if __name__ == '__main__':
    main()



