'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent Williams

'''
import numpy as np
import scipy.ndimage
from configparser import ConfigParser
import SWOTRiver.products.rivertile
import textwrap
import os.path
import pandas as pd
import datetime

MISSING_VALUE_FLT = -999999999999

def textjoin(text):
    """Dedent join and strip text"""
    text = textwrap.dedent(text)
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

def decode_bitflag(flag_meanings, flag_masks, qual=None):
    out = {}
    mask = {}
    # TODO: maybe should check if flag_masks are powers of 2...
    for flag_mask, flag_meaning in zip(flag_masks, flag_meanings):
        if qual is not None:
            out[flag_meaning] = (np.bitwise_and(qual, flag_mask) / flag_mask).astype('uint32')
        mask[flag_meaning] = flag_mask
    return out, mask

def decode_node_q_b(node_qual, key='node_q_b'):
    tmp = SWOTRiver.products.rivertile.RiverTileNodes()
    flag_meanings = tmp.VARIABLES[key]['flag_meanings'].split()
    flag_masks = tmp.VARIABLES[key]['flag_masks']
    return decode_bitflag(flag_meanings, flag_masks, node_qual)


def reach_id_from_node_id_int(node_id_arr):
    return (node_id_arr/10000).astype(int)*10 + node_id_arr - (
        node_id_arr/10).astype(int)*10

def node_id_to_local_node_id(node_ids):
    """
    maps node_id to local node_id without the reach_id prefix
    """
    local_node_ids = [int(str(node_id)[-4:-1]) for
        node_id in node_ids]
    return np.array(local_node_ids)

def get_stretch_list_from_subset_cfg(cfg, sword_df):
    # get the list of stretches (or multireaches)
    stretch_list0 = [
        '{}'.format(t) for t in '{}'.format(
            cfg['main']['stretch_subset']).split()]
    stretch_list = []
    for stretch in stretch_list0:
        if stretch.isnumeric():
            if sword_df is None:
                # just append the potentially-partial reach
                stretch_list.append('{}'.format(stretch))
            else:
                # get all reaches in basins smaller than stretch
                st = '{}'.format(stretch)
                tmp = [
                    '{}'.format(r).startswith(st) for r in sword_df['reach_id']]
                reaches = np.array(sword_df['reach_id'][tmp])
                for r in reaches:
                    stretch_list.append('{}'.format(r))
        else:
            # check if it is a file, if so read it
            if stretch.endswith('.csv') and os.path.isfile(stretch):
                tmp_df = pd.read_csv(stretch)
                s_list = []
                if 'stretch_name' in tmp_df.keys():
                    s_list = tmp_df['stretch_name']
                elif 'reach_id' in tmp_df.keys():
                    s_list = tmp_df['reach_id']
                for s in s_list:
                    stretch_list.append('{}'.format(s))
            else:
                # assume it is a non-numeric stretch name
                stretch_list.append('{}'.format(stretch))
    return stretch_list

def swot_time_to_field_time(swot_times, swot_filenames=None):
    """
    convert swot total-second time to utc datetime
    copied from reproc repo    
    """
    utc_time = []
    time_start = datetime.datetime(2000, 1, 1, 0,
                                   0, 0,
                                   tzinfo=datetime.timezone.utc)
    for index, this_time in enumerate(swot_times):
        if this_time == MISSING_VALUE_FLT:
            if swot_filenames is None:
                # stuff in Jan 1, 2023
                this_time = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc).second
            else:
                # grab date time from nearby node on same cycle
                print('One of the features in', swot_filenames[index],
                    'is missing a time value. This is off-nominal. Filling '
                    'with nearby time...')
                indices = np.logical_and(swot_filenames == swot_filenames[index],  
                                     swot_times != MISSING_VALUE_FLT)
                close_times = swot_times[indices]
                closest_index = close_times.index[0] + np.abs(
                    index - close_times.index).argmin()
                this_time = close_times[closest_index]
        time_start.timestamp()
        time = time_start + datetime.timedelta(seconds=this_time)
        utc_time.append(datetime.datetime.utcfromtimestamp(time.timestamp()))
    return utc_time

def compute_anomaly(full_profile_data, bayes_data=None, sword_node_df=None):
    full_profile_data2 = full_profile_data.copy()
    if 'wse_stack_detrend' not in full_profile_data2.keys():
        full_profile_data2 = detrend(full_profile_data, sword_node_df)
    anomaly = []
    anomaly_filtered = []
    medprof_stack = []
    for k, network in enumerate(full_profile_data2['network']):
        if bayes_data is not None:
            # if Bayes_data use those WSE's
            wse_stack = np.array(bayes_data['wse_stack'][k])
        else:
            wse_stack = np.array(full_profile_data2['wse_stack'][k])
        msk = np.ones_like(full_profile_data2['wse_stack_detrend'][k])
        msk[np.isnan(full_profile_data2['wse_stack'][k])] = np.nan
        med = np.nanmedian(msk*full_profile_data2['wse_stack_detrend'][k],axis=1)
        med[np.isnan(med)] = 0
        med_fit = np.median(full_profile_data2['wse_stack_linfit'][k],axis=1)
        med_prof = med_fit + scipy.ndimage.uniform_filter1d(med, 20)
        med_prof_stack = np.tile(med_prof, (len(wse_stack[0,:]),1)).T
        anom = wse_stack - med_prof_stack
        anom1 = wse_stack - med_prof_stack
        # fill the nans with the median
        for j in range(len(anom1[0,:])):
            anom2 = anom1[:,j]
            med0 = np.nanmedian(anom2)
            anom2[np.isnan(anom2)] = med0
            anom1[:,j] = anom2
        anom1_filt = anom1.copy()
        for j in range(len(anom1[0,:])):
            anom1_filt[:,j] = scipy.ndimage.uniform_filter1d(anom1[:,j], 21)
        anomaly_filtered.append(anom1_filt)
        anomaly.append(anom1)
        medprof_stack.append(med_prof_stack)
    full_profile_data2['anomaly_filtered'] = anomaly_filtered
    full_profile_data2['anomaly'] = anomaly
    full_profile_data2['med_prof_stack'] = medprof_stack
    return full_profile_data2

class CfgParser(ConfigParser):
    '''A wrapper to ConfigParser, with automatic data types'''
    def __init__(self, *args, **kwargs):
        super(ConfigParser, self).__init__(*args, **kwargs)
        self.getting = False

    def read(self, filename, *args, **kwargs):
        """Load a file, adding 'main' section if necessary"""
        string = open(filename, 'r').readlines()
        if string[0][0] != '[' and string[0][-1] != ']':
            string = ['[main]\n'] + string
        string = ''.join(string)
        super(ConfigParser, self).read_string(string)


    def get(self, *args, **kwargs):
        '''Get the data as specific type, if possible'''
        # Deal with recursion; the various get*** methods call the 'get'
        # function and then convert the value. If one of those is calling, we
        # should call the super 'get'. Another option would be to implement
        # the converters directly (though the boolean one has some actual
        # smarts).
        # print('in get', self.getting, args, kwargs)
        if self.getting:
            return super(ConfigParser, self).get(*args, **kwargs)
        else:
            self.getting = True
        # Try to get with various types, this order should be sufficient.
        try:
            value = super(ConfigParser, self).getint(*args, **kwargs)
            self.getting = False
            return value
        except ValueError:
            try:
                value = super(ConfigParser, self).getfloat(*args, **kwargs)
                self.getting = False
                return value
            except ValueError:
                try:
                    value = super(ConfigParser, self).getboolean(
                        *args, **kwargs)
                    self.getting = False
                    return value
                except ValueError:
                    value = super(ConfigParser, self).get(*args, **kwargs)
                    # Cast 'None' strings to NoneTypes
                    if value.lower() == 'none':
                        value = None
                    self.getting = False
                    return value

