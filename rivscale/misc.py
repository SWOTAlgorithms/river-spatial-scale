
import numpy as np
import scipy.ndimage


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
