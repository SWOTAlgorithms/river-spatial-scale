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
    #dic['wse_stretch_average'] = None
    #dic['width_stretch_average'] = None
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
    # get the originals
    wse = dic['stretch_stack'].wse.copy()
    width = dic['stretch_stack'].width.copy()
    #
    dark_frac = dic['stretch_stack'].dark_frac.copy()
    # plot dark frac 2D
    kwargs = {'cmap':'jet', 'aspect':'auto', 'interpolation':'none'}
    plt.figure()
    plt.imshow(dark_frac.T, **kwargs)
    plt.title('dark_frac')
    plt.xlabel('node')
    plt.ylabel('time')
    plt.colorbar()
    # wse
    wse_stretch_stack = dic['stretch_stack'].copy()
    wse2_stretch_stack = dic['stretch_stack'].copy() 
    wse_stretch_stack.filter_node_outliers(
        dic['wse_stats'],
        key='wse',
        Delta2=False,
        use_ptiles=False,
        plot=True)
    plt.suptitle('wse outliers (x) using multitemporal wse stats')
    wse2_stretch_stack.filter_node_outliers(
        dic['wse_stats'],
        key='wse',
        Delta2=True,
        use_ptiles=False,
        plot=True)
    plt.suptitle('wse outliers (x) using multitemporal wse stats')
    # width
    width_stretch_stack = dic['stretch_stack'].copy()
    width2_stretch_stack = dic['stretch_stack'].copy()
    width_stretch_stack.filter_node_outliers(
        dic['width_stats'],
        key='width',
        Delta2=False,
        use_ptiles=False,
        plot=True)
    plt.suptitle('width outliers (x) using multitemporal width stats')
    width2_stretch_stack.filter_node_outliers(
        dic['width_stats'],
        key='width',
        Delta2=True,
        use_ptiles=False,
        plot=True)
    plt.suptitle('width outliers (x) using multitemporal width stats')
    # pekel
    if 'pekel_stats' in dic.keys():
        pekel_stretch_stack = dic['stretch_stack'].copy()
        pekel_stretch_stack.filter_node_outliers(
            dic['pekel_stats'], key='width',plot=True)
        plt.suptitle('width outliers (x) using pekel width stats')

    if args.outdir is None:
        plt.show()
    breakpoint()
    return 
    """
    # plot 2d
    #dic['stretch_stack'].filter_dark_water('width',0.2)
    #dic['stretch_stack'].smooth_widths()
    #wse = dic['stretch_stack'].wse.copy()
    #width = dic['stretch_stack'].width.copy()
    #dark_frac = dic['stretch_stack'].dark_frac.copy()
    ref = dic['wse_stats'].reference
    ref2 = np.broadcast_to(ref, np.shape(wse.T)).T
    ref_w = dic['width_stats'].reference
    ref2_w = np.broadcast_to(ref_w, np.shape(width.T)).T
    # now filter
    #dic['stretch_stack'].filter_dark_water('width',0.2)
    #dic['stretch_stack'].smooth_widths()
    wse_f = dic['stretch_stack'].wse
    width_f = dic['stretch_stack'].width
    #dic['stretch_stack'].filter
    kwargs = {'cmap':'jet', 'aspect':'auto', 'interpolation':'none'}
    #
    plt.figure()
    plt.imshow(wse.T - ref2.T, **kwargs)
    plt.title('$\Delta$ wse')
    plt.xlabel('node')
    plt.ylabel('time')
    plt.colorbar()
    plt.figure()
    plt.imshow(width.T - ref2_w.T, **kwargs)
    plt.title('$\Delta$ width')
    plt.xlabel('node')
    plt.ylabel('time')
    plt.colorbar()
    plt.figure()
    plt.imshow(dark_frac.T, **kwargs)
    plt.title('dark_frac')
    plt.xlabel('node')
    plt.ylabel('time')
    plt.colorbar()
    #
    plt.figure()
    plt.imshow(wse_f.T - ref2.T, **kwargs)
    plt.title('$\Delta$ wse (filtered)')
    plt.xlabel('node')
    plt.ylabel('time')
    plt.colorbar()
    plt.figure()
    plt.imshow(width_f.T - ref2_w.T, **kwargs)
    plt.title('$\Delta$ width (filterd/smoothed)')
    plt.xlabel('node')
    plt.ylabel('time')
    plt.colorbar()
    """
    # plot dark frac 2D
    kwargs = {'cmap':'jet', 'aspect':'auto', 'interpolation':'none'}
    plt.figure()
    plt.imshow(dark_frac.T, **kwargs)
    plt.title('dark_frac')
    plt.xlabel('node')
    plt.ylabel('time')
    plt.colorbar()
    #
    ptile_list = dic['pekel_stats'].percentile_list.copy()
    ptiles = dic['pekel_stats'].percentiles.copy()
    ptile_ref = dic['pekel_stats'].reference.copy()
    ptile_ref2 = np.broadcast_to(ptile_ref, np.shape(ptiles.T)).T
    dptiles = ptiles - ptile_ref2
    plt.plot()
    plt.figure()
    plt.imshow(dptiles.T,
        aspect='auto', interpolation='none', cmap='jet')
    plt.colorbar()
    #
    # compute 
    width = dic['stretch_stack'].width
    #ref_w = dic['width_stats'].reference
    ref_w = dic['pekel_stats'].reference
    ref2_w = np.broadcast_to(ref_w, np.shape(width.T)).T
    dwidth = width - ref2_w

    wse = dic['stretch_stack'].wse.copy()
    ref = dic['wse_stats'].reference
    ref2 = np.broadcast_to(ref, np.shape(wse.T)).T
    dwse = wse - ref2
    #
    med_dptiles = np.median(dptiles, axis=0)
    med_dwidth = np.nanmedian(width - ref2_w, axis=0)
    # create an interpolator to get ptile given width
    import scipy.interpolate
    intrp = scipy.interpolate.interp1d(
            med_dptiles,
            np.arange(len(ptile_list)),
            kind='nearest',
            bounds_error=False, fill_value='extrapolate')
    pcnt_id = intrp(med_dwidth)
    pcnt_id[pcnt_id>len(ptile_list)] = pcnt_id[-1]
    pcnt_id[pcnt_id<0] = pcnt_id[0]
    dwidth_fit  = dptiles[:,pcnt_id.astype(int)]
    plt.figure()
    plt.imshow(dwidth_fit.T, **kwargs)
    plt.colorbar()
    plt.title('$\Delta$ width best pekel thresh')
    # plot
    plt.figure()
    plt.plot(dwidth, dwidth_fit,'o')
    xx = np.arange(-200,200)
    plt.plot(xx,xx,'-k')

    plt.figure()
    plt.plot(dwidth, dwse,'o')
    
    plt.figure()
    plt.plot(dwidth_fit, dwse,'o')

    plt.figure()
    plt.imshow(dwidth.T - dwidth_fit.T, **kwargs)
    plt.colorbar()

    plt.figure()
    plt.plot(dwidth.T)

    plt.figure()
    plt.plot(dwidth_fit.T)

    plt.figure()
    gid = width_stretch_stack.granule_id
    gid[gid==''] = '000_000_00'
    plt.plot(dwidth[160:165,:].T,'-o');
    plt.xticks(range(len(gid)), gid, rotation='vertical')
    plt.tight_layout()
    #
    # plot height width colored by pass
    #
    """
    gid0 = []
    for tid in dic['wse_stretch_average'].time_id:
        gid0.append(gid[width_stretch_stack.time_id==tid])
    gid0 = np.array(gid0).squeeze()
    pid0 = np.array([g.split('_')[1] for g in gid0]) 
    upid0 = np.unique(pid0)
    """
    wse_avg = dic['wse_stretch_average'].mean
    width_avg = dic['width_stretch_average'].mean
    gid0 = dic['width_stretch_average'].granule_id
    pid0 = np.array([g.split('_')[1] for g in gid0]) 
    upid0 = np.unique(pid0)
    plt.figure()
    for p in upid0:
        this_wse = wse_avg[pid0==p].flatten()
        this_width = width_avg[pid0==p].flatten()
        plt.plot(this_width, this_wse, 'o', label='pass {}'.format(p))
    plt.legend()

    """
    this_width = np.zeros(np.shape(width))
    for k,p in enumerate(ptile_list):
        if k < len(ptile_list):
            p_next = ptile_list[k+1]
        else:
            p_next = ptile_list[-1]
        if k==0:
            p_last = ptile_list[0]
        
        msk_left = pcnt > p - (p_last - p)/2
        msk_right = pcnt <= p + (p_next - p)/2
        if k==len(ptile_list):
            msk_right = np.isfinite(pcnt)
        if k==0:
            msk_left = np.isfinite(pcnt)
        msk = np.logical_and(msk_lft, msk_right)
        # 
        this_width[,]
    """
    # maybe should mask same as outliers first ?
    #good = np.isfinite(med_dwidth)
    #for k,arr in enumerate(dptiles[:,k]):
    #    dptiles[k] = 

    #breakpoint()
    #
    if args.outdir is None:
        plt.show()
    breakpoint()

if __name__ == "__main__":
    main()

