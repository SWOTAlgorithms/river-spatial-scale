#!/usr/bin/env python
'''
Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

This code plots the stretch data overlaid on google-map images
'''

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles

import os.path
import rivscale.products.along_stretch

import geopandas as gpd
import numpy as np
import argparse
import glob

EXAMPLE = ''

def main():
    parser = argparse.ArgumentParser(
        description='Plot along-river stretch on map over google-image',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('along_stretch_file', default=None,
        help='filename of AlongStretchStats netcdf file')
    parser.add_argument('--sword_node_file', default=None,
        help='SWORD gpkg file')
    parser.add_argument('--bbox', default=[-124, -122, 44, 45],
        help='bbox for SWORD so we dont rty to load in everything')
    parser.add_argument('--crop_reach',default=False, action='store_true')
    args = parser.parse_args()
    # read in the along-stats
    #sword_node_file = '/u/swot-fn-r0/swot/sim_proc_inputs/river_database/20230802/v16/gpkg/na_sword_nodes_v16.gpkg'
    #reach_id = '78220000151'
    #basedir = '/u/franka-z/bawillia/working/flow_waves/PxC0/PNW/outputs/both_orbits/{}/v1'.format(reach_id)
    #fle = os.path.join(basedir, '{}_wse_stats.nc'.format(reach_id))
    #glob_str = os.path.join(args['along_stretch_dir'], '*_width_stats.nc')

    along_stats = rivscale.products.along_stretch.AlongStretchStats.from_ncfile(
            args.along_stretch_file)
    if args.crop_reach:
        along_stats = along_stats.crop_to_reach()
    #breakpoint()
    p25 = along_stats.percentiles[:,along_stats.percentile_list == 25].squeeze()
    p75 = along_stats.percentiles[:,along_stats.percentile_list == 75].squeeze()
    iqr = p75-p25
    #breakpoint()
    # define the bounding box
    buff = 0.01 #
    bbox = [
        np.nanmin(along_stats.p_lon) - buff,
        np.nanmax(along_stats.p_lon) + buff,
        np.nanmin(along_stats.p_lat) - buff,
        np.nanmax(along_stats.p_lat) + buff]
    """
    #breakpoint()
    if args.sword_node_file is not None
        # read in SWORD as a geopandas
        sword = gpd.read_file(args.sword_node_file, args.bbox)
        this_sword  = sword[sword['node_id'].isin(np.array(along_stats.node_id))]
        
        #bbox = [np.min(this_sword.x) - buff, np.max(this_sword.x) + buff,
        #    np.min(this_sword.y) - buff, np.max(this_sword.y) + buff]
    # make new array from along_stats data ordered like sword
    #along_stats.percentiles
    #breakpoint()
    reference = np.array(this_sword['width']) * np.nan
    IQR = np.array(this_sword['width']) * np.nan
    for node_id, ref, q in zip(along_stats.node_id, along_stats.reference, iqr):
        reference[this_sword['node_id'] == node_id] = ref
        IQR[this_sword['node_id'] == node_id] = q
    """
    #breakpoint()
    # plot it
    tiler = GoogleTiles(style="satellite")
    mercator = tiler.crs
    #that_sword = this_sword.to_crs(mercator)
    #breakpoint()
    fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1, projection=mercator)
    ax = plt.axes(projection=mercator)
    ax.set_extent(bbox, crs=ccrs.Geodetic())#ccrs.PlateCarree())

    zoom = 14
    ax.add_image(tiler, zoom)
    #ax.coastlines('10m')
    #
    #scat = ax.scatter(that_sword.geometry.x, that_sword.geometry.y,
    #        c = IQR, s=reference, alpha=0.7,cmap='jet')
    scat = ax.scatter(along_stats.p_lon, along_stats.p_lat,
        c = iqr, s=along_stats.reference,
        alpha=0.7,cmap='jet', transform=ccrs.Geodetic())
    
    fig.colorbar(scat, label='width IQR (m)')
    plt.show()


if __name__ == '__main__':
    main()



