'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent Williams

'''
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.ndimage
import netCDF4 as nc
import rivscale.misc
import rivscale.reconstruct
from rivscale.misc import swot_time_to_field_time

def label_units(label):
    if 'slope' in label:
        label = label + ' (m/m)'
    elif ('wse' in label) or ('width' in label) or ('dist' in label):
        label = label + ' (m)'
    elif ('flow' in label):
        label = label + ' (deg.)'
    elif ('layovr' in label):
        label = label + ' (m)'
    return label

def plot_stretch_average_hw(stretch_data):
    p_est = stretch_data['hw_params']
    wse = stretch_data.stretch_wse_mean
    width = stretch_data.stretch_width_mean
    ref = stretch_data.wse_reference
    ref_w = stretch_data.width_reference
    plt.figure()
    plt.scatter(wse, width)
    x = np.linspace(np.nanmin(wse),np.nanmax(wse))
    y = rivscale.reconstruct.piecewise_linear(p_est, x)
    plt.plot(x,y,'k', linewidth=3)
    plt.grid()

def plot_spectra(
        stretch_data,
        y_key='wse',
        char_length_tau=10000,
        prior_unc_alpha=2,
        title='',
        show=False,
        outdir=None):
    # zero-fill in the nomanom
    # plot along-river spectra of the anomaly fields
    dist_out = stretch_data['dist_out']
    y = stretch_data[y_key]
    ref = stretch_data['{}_reference'.format(y_key)]
    ref2 = np.broadcast_to(ref, np.shape(y.T)).T
    anom = y - ref2
    # now get anomanom
    med_anom = np.nanmedian(anom,axis=0)
    med_anom_prof_stack = np.tile(med_anom.T, (len(y[:,0]),1))
    anomanom = anom - med_anom_prof_stack
    tmp = anomanom.copy()
    tmp[~np.isfinite(tmp)] = 0
    figsize=(7,6)
    plt.figure(figsize=figsize)
    plt.subplot(2,1,1)
    avg = 0     
    for k in range(len(tmp[0,:])): 
        f, pxx = scipy.signal.periodogram(tmp[:,k])
        plt.loglog(f, pxx)
        #plt.semilogy(f, pxx)
        avg = avg + pxx
        #plt.show()
    ylim = (1e-6, 1000)
    plt.grid()
    plt.ylabel('power spectral density')
    #plt.xlabel('wave-number/spatial-frequency (1/node)')
    plt.ylim(ylim)
    avg = avg / len(tmp[0,:])
    plt.title(title)
    #plt.figure()
    plt.subplot(2,1,2)
    plt.loglog(f, avg)
    #char_length_tau = 100000
    #prior_unc_alpha = 1.5
    i0 = int(np.floor(len(dist_out)/2))
    t = dist_out - dist_out[i0]
    exp_cov = np.exp(-np.abs(t) / char_length_tau)
    exp_cov = exp_cov * np.max(exp_cov) * prior_unc_alpha ** 2
    #f2, pxx_exp = scipy.signal.periodogram(exp_cov)
    ft = np.abs(np.fft.fft(exp_cov))
    plt.loglog(f, ft[0:i0+1])
    # also plot an estimte of the noise floor
    #breakpoint()
    #noise = np.sqrt(0.01) * np.random.randn(len(exp_cov))
    #ft_noise = np.abs(np.fft.fft(noise))
    #plt.loglog(f, ft_noise[0:i0+1])
    noise_floor = 0.05 * np.ones_like(f)
    plt.loglog(f, noise_floor)
    plt.loglog(f, noise_floor + ft[0:i0+1])
    plt.ylim(ylim)
    plt.grid()
    plt.legend(['average power spectrum','exp cov', 'noise floor', 'exp cov with noise floor'])
    plt.ylabel('power spectral density')
    plt.xlabel('wave-number/spatial-frequency (1/node)')
    #plt.title(title)
    if outdir is not None:
        # create output dir if not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = '{}_spectrum_of_{}'.format(title, y_key)
        plt.savefig(os.path.join(outdir, fname), dpi=300)
        plt.close()
    else:
        if show:
            plt.show()

def plot_stretch_stats(
        stats,
        x_key='along_dist',#'dist_out',
        outdir=None,
        show=False,
        figsize=(10,5),
        plot_slope=False,
        title_tag=''):
    nplots = 2
    if 'slope' in (stats.variables.keys()):
        plot_slope = True
        figsize = (figsize[0], figsize[1] + 2)
        nplots = 3
    title = stats.stretch_name
    if title_tag != '':
        title = title+' '+title_tag
    x = stats[x_key]
    x_label = x_key
    marker=None
    if x_key=='time_id':
        # convert to datetime
        #breakpoint()
        x = swot_time_to_field_time(x*60.0*60.0)
        x_label = 'time'
        marker = '.'
    y_key = stats.signal_key
    y_mean = stats.mean
    file_tag = 'profile_stats'
    try:
        y_ref = stats.reference
    except AttributeError:
        # it must be a stretch_average object
        y_ref = stats.mean_reference + np.zeros_like(y_mean)
        file_tag = 'stretch_avg'
    y_std = stats.std
    y_ptiles = stats.percentiles
    if len(y_ptiles)>0:
        # plot the ptiles
        ptiles = ['{}-%ile'.format(t) for t in stats['percentile_list']]
        x2D = np.broadcast_to(x, np.shape(y_ptiles.T)).T
    else:
        ptiles = []
    #figsize=(10,5)
    plt.figure(figsize=figsize)
    plt.subplot(nplots,1,1)
    if len(y_ptiles)>0:
        plt.plot(x2D, y_ptiles, marker=marker)
    plt.plot(x, y_ref, 'k', linewidth=2)
    ncols = np.ceil((len(ptiles)+1)/3)
    #if ncols==0:
    #    ncols=i
    y_label = y_key
    plt.legend(ptiles+['ref',], ncol=ncols)
    plt.grid()
    plt.xlabel(label_units(x_label))
    plt.ylabel(label_units(y_label))
    title = title+' {} statistics'.format(y_key)
    plt.suptitle(title)
    plt.subplot(nplots,1,2)
    plt.plot(x, y_mean, marker=marker)
    plt.plot(x, y_mean+y_std,'--')
    plt.plot(x, y_mean-y_std,'--')
    plt.plot(x, y_ref, 'k', linewidth=2)
    plt.legend(['mean', 'mean + std', 'mean - std', 'ref'])
    plt.grid()
    plt.xlabel(label_units(x_label))
    plt.ylabel(label_units(y_label))
    if plot_slope:
        plt.subplot(nplots,1,3)
        plt.plot(x, stats.slope, label='data', marker=marker)
        plt.plot(x, stats.slope_reference, label='reference', marker=marker)
        plt.legend()
        plt.grid()
        plt.xlabel(label_units(x_label))
        plt.ylabel(label_units(y_key+'_slope'))
    plt.tight_layout()
    if outdir is not None:
        # create output dir if not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = '{}_{}_vs_{}'.format(
            title.replace(' ', '_').replace('$\Delta$','delta'),
            file_tag, x_label)
        plt.savefig(os.path.join(outdir, fname), dpi=300)
        plt.close()
    else:
        if show:
            plt.show()

def plot_the_per_pass(gid, x, y, y_u, marker):
    pid = np.array([g.split('_')[1] for g in gid])
    upid = np.unique(pid)
    for p in upid:
        this_y = y[pid==p].flatten()
        this_x = x[pid==p].flatten()
        this_x = swot_time_to_field_time(this_x*60.0*60.0)
        this_y_u = y_u[pid==p].flatten()
        plt.errorbar(this_x, this_y, yerr=this_y_u,
            marker=marker, label='pass {}'.format(p))
        #plt.plot(this_x, this_y, marker=marker, label='pass {}'.format(p))
    plt.grid()
    plt.legend()

def plot_per_pass_time_series(
        stats,
        stats2=None,
        outdir=None,
        show=False,
        figsize=(10,5),
        title_tag='',
        err_type='node_std'):
    x_key='time_id'
    nplots = 1
    title = stats.stretch_name
    if title_tag != '':
        title = title+' '+title_tag
    #title = title + ' (per pass)'
    x = stats[x_key]
    x_label = x_key
    #x = swot_time_to_field_time(x*60.0*60.0)
    x_label = 'time'
    marker = '.'
    #file_tag = 'stretch_avg'
    y_key = stats.signal_key
    y_mean = stats.mean
    if err_type=='node_std':
        y_u = stats.std
    elif err_type=='stretch_std':
        y_u = stats.std / np.sqrt(stats.count)
    else:
        y_u = stats.uncert
    y_ref = stats.mean_reference + np.zeros_like(y_mean)
    gid = stats.granule_id.copy()
    if stats2 is not None:
        nplots=2
    plt.figure(figsize=figsize)
    plt.subplot(nplots,1,1)
    plot_the_per_pass(gid, x, y_mean, y_u, marker)
    plt.xlabel(label_units(x_label))
    plt.ylabel(label_units(y_key))
    if stats2 is not None:
        y_key2 = stats2.signal_key
        y_mean2 = stats2.mean
        if err_type=='node_std':
            y_u2 = stats2.std
        elif err_type=='stretch_std':
            y_u2 = stats2.std / np.sqrt(stats2.count)
        else:
            y_u2 = stats2.uncert
        #y_u2 = stats2.std / np.sqrt(stats.count)
        y_ref2 = stats2.mean_reference + np.zeros_like(y_mean2)
        gid2 = stats2.granule_id.copy()
        plt.subplot(nplots,1,2)
        plot_the_per_pass(gid2, x, y_mean2, y_u2, marker)
        plt.xlabel(label_units(x_label))
        plt.ylabel(label_units(y_key2))
    if stats2 is None:
        title = title+' {} (per pass)'.format(y_key)
    else:
        title = title+' {} and {} (per pass)'.format(y_key, y_key2)
    plt.suptitle(title)
    plt.tight_layout()
    if outdir is not None:
        # create output dir if not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = '{}_vs_time'.format(
            title.replace(' ', '_').replace('$\Delta$','delta').replace(
                '(','').replace(')',''))
        plt.savefig(os.path.join(outdir, fname), dpi=300)
        plt.close()
    else:
        if show:
            plt.show()

def plot_2D_stretch_stack(
        stretch_stack,
        y_key='wse',
        y_reference=None,
        y_anom=False,
        outdir=None,
        figsize=(10,5),
        show = False,
        title_tag='',
        bit=None):
    cmap = 'jet'
    if y_key=='flow_angle':
        cmap='hsv'
    clim = None
    clabel = y_key
    y = stretch_stack[y_key].copy()
    if y_key=='node_q_b':
        #y = np.logical_and(y,2**bit)
        out, mask = rivscale.misc.decode_node_q_b(np.array(y).astype('uint32'))
        #breakpoint()
        bit_name = ''
        for key in mask.keys():
            if (mask[key] == 2**bit):
                bit_name = key
        y = out[bit_name]
        clabel = clabel + ' bit {:d} {}'.format(bit, bit_name)
        clim = (0,1)
        cmap = matplotlib.colors.ListedColormap(['deepskyblue', 'crimson'])
    ref_2D = np.zeros_like(y)
    if y_reference is not None:
        ref = y_reference.reference
        ref_2D = np.broadcast_to(ref, np.shape(y.T)).T
        if y_anom:
            clabel = '$\Delta$ '+ y_key
            y = y - ref_2D
    title = '{} {} image'.format(
        stretch_stack.stretch_name,
        clabel)
    if title_tag != '':
        title = title+ ' ' + title_tag
    if clim is None:
        clim = (np.nanpercentile(y.flatten(),5),
            np.nanpercentile(y.flatten(),95),)
    if y_key=='dark_frac':
        clim = (0,1)
    kwargs = {'cmap':cmap, 'interpolation':'none', 'aspect':'auto'}
    plt.figure(figsize=figsize)
    plt.imshow(y.T, clim=clim, **kwargs)
    plt.colorbar(label=label_units(clabel))
    plt.ylabel('time index')
    plt.xlabel('node index')
    plt.title(title)
    if outdir is not None:
        # create output dir if not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = '{}'.format(
                title.replace(' ','_').replace('$\Delta$','delta'))
        plt.savefig(os.path.join(outdir, fname), dpi=300)
        plt.close()
    else:
        if show:
            plt.show()

def plot_stretch_stack(
        stretch_stack,
        x_key='along_dist',#,'dist_out',
        y_keys=['wse','width'],
        y_reference=[None, None],
        y_anom=[False, False],
        outdir=None,
        figsize=(10,5),
        show = False,
        marker='-',
        title_tag=''):
    title = stretch_stack.stretch_name
    if title_tag != '':
        title =title + ' ' + title_tag
    ylabel = ''
    x = stretch_stack[x_key]
    y = stretch_stack[y_keys[0]]
    y2 = y.copy()
    if len(y_keys)==2:
        y2 = stretch_stack[y_keys[1]]
    y_labels = y_keys.copy()
    ref = np.zeros_like(x) + np.nan
    ref2 = np.zeros_like(x) + np.nan
    ref_2D = np.zeros_like(y)
    ref2_2D = np.zeros_like(y) 
    if y_reference[0] is not None:
        ref = y_reference[0].reference
        ref_2D = np.broadcast_to(ref, np.shape(y.T)).T
        if y_anom[0]:
            y_labels[0] = '$\Delta$ '+ y_keys[0]
    if len(y_reference)==2:
        if y_reference[1] is not None:
            ref2 = y_reference[1].reference
            ref2_2D = np.broadcast_to(ref2, np.shape(y.T)).T
        if y_anom[1]:
            y_labels[1] = '$\Delta$ '+y_keys[1]
    if y_anom[0]:
        y = y - ref_2D
    if len(y_anom)==2:
        if y_anom[1]:
            y2 = y2 - ref2_2D
    x_label = x_key
    if x_key=='time_id':
        # convert to datetime
        x = swot_time_to_field_time(x*60*60)
        y = y.T
        y2 = y2.T
        x_label = 'time'
    plt.figure(figsize=figsize)
    if len(y_keys)==2:
        plt.subplot(2,1,1)
        ylabel = '{}_{}'.format(y_keys[0], y_labels[1])
    else:
        ylabel = '{}'.format(y_keys[0])
    plt.plot(x, y, marker, markersize=1)
    if np.sum(np.isfinite(ref)):
        plt.plot(x, ref, 'k', linewidth=2, label='reference')
        plt.legend()
    plt.ylabel(label_units(y_labels[0]))
    plt.grid()
    # TODO: enable plotting reference
    # make first plot
    if len(y_keys)==2:
        plt.subplot(2,1,2)
        # make second plot
        plt.plot(x, y2, marker, markersize=1)
        plt.ylabel(label_units(y_labels[1]))
        plt.suptitle(title)
        plt.grid()
    else:
        plt.title(title)
    plt.xlabel(label_units(x_label))
    if outdir is not None:
        # create output dir if not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = '{}_{}_vs_{}'.format(
                title.replace(' ','_').replace('$\Delta$','delta'),
                ylabel.replace(' ','_').replace('$\Delta$','delta'), x_label)
        plt.savefig(os.path.join(outdir, fname), dpi=300)
        plt.close()
    else:
        if show:
            plt.show()

