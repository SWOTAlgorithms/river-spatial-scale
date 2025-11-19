#!/usr/bin/env python
'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.


This code follows the template from the podaac hydrocron notebook for ingesting
SWOT River node and reach data

see

https://podaac.github.io/tutorials/notebooks/datasets/Hydrocron_SWOT_timeseries_examples.html

'''

import dask
import dask.dataframe as dd
from dask.distributed import Client
import hvplot.dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
import requests

import datetime
from io import StringIO

import os

def query_fts(query_url, params, kind='node_ids'):
    """Query Feature Translation Service (FTS) for reach identifers using the query_url parameter.

    Parameters
    ----------
    query_url: str - URL to use to query FTS
    params: dict - Dictionary of parameters to pass to query

    Returns
    -------
    dict of results: hits, page_size, page_number, reach_ids
    """

    reaches = requests.get(query_url, params=params)
    reaches_json = reaches.json()
    #breakpoint()
    hits = reaches_json['hits']
    if 'search on' in reaches_json.keys():
        page_size = reaches_json['search on']['page_size']
        page_number = reaches_json['search on']['page_number']
    else:
        page_size = 0
        page_number = 0

    if 'node' in kind:
       out_key = "node_ids"
       this_key = 'node_id'
    else:
       out_key = "reach_ids"
       this_key = 'reach_id' 
    return {
        "hits": hits,
        "page_size": page_size,
        "page_number": page_number,
        out_key: [ item[this_key] for item in reaches_json['results'] ]
        }

def empty_feature_dic(feature, kind='node_ids'):
    if 'node' in kind:
        d = {
            "reach_id": np.int64(0),
            "node_id": np.int64(feature),
            "river_name": "no_data",
            "time": 0.0,
            "time_str": datetime.datetime(1900, 1, 1).strftime("%Y-%m-%dT%H:%M:%S"),
            "crid":'PGC0',
            "pass_id": 0,
            "cycle_id": 0,
            "continent_id": 'NA',
            "node_q": 4,
            "node_q_b": 3,
            "xovr_cal_q": 20,
            "dark_frac": -1.0,
            "ice_clim_f": 0,
            "wse": -999999999999.0,
            "wse_r_u": -999999999999.0,
            "area_total": -999999999999.0,
            "area_tot_u": -999999999999.0,
            "area_detct": -999999999999.0,
            "area_det_u": -999999999999.0,
            "area_wse": -999999999999.0,
            "width": -999999999999.0,
            "p_dist_out": -999999999999.0,
            "p_length": -999999999999.0,
            "xtrk_dist": -999999999999.0,
            "rdr_sig0": -999999999999.0,
            "node_dist": -999999999999.0,
            "flow_angle": -999999999999.0,
            "n_good_pix": 0,
            "lat": -999999999999.0,
            "lon": -999999999999.0,
            #"p_lat": -999999999999.0,
            #"p_lon": -999999999999.0,
            "time_units": "sec",
            "dark_frac_units": np.int64(1),
            "wse_units": "m",
            "wse_r_u_units": "m",
            "area_total_units": "m^2",
            "area_tot_u_units": "m^2",
            "area_detct_units": "m^2",
            "area_det_u_units": "m^2",
            "area_wse_units": "m^2",
            "width_units": "m",
            "p_dist_out_units": "m",
            "p_length_units": "m",
            "xtrk_dist_units": "m",
            "rdr_sig0_units":np.int64(1),
            "node_dist_units": "m",
            "flow_angle_units": "degrees",
            "n_good_pix_units": np.int64(1),
            "lat_units": "degrees",
            "lon_units": "degrees",
            }
    else:
        d = {
            "reach_id": np.int64(feature),
            "river_name": "no_data",
            "time": 0.0,
            "time_str": datetime.datetime(1900, 1, 1).strftime("%Y-%m-%dT%H:%M:%S"),
            "crid":'PGC0',
            "pass_id": 0,
            "cycle_id": 0,
            "continent_id": 'NA',
            "reach_q": 4,
            "reach_q_b": 3,
            "xovr_cal_q": 20,
            "dark_frac": -1.0,
            "ice_clim_f": 0,
            "wse": -999999999999.0,
            "wse_r_u": -999999999999.0,
            "slope": -999999999999.0,
            "slope_r_u": -999999999999.0,
            "area_total": -999999999999.0,
            "area_tot_u": -999999999999.0,
            "area_detct": -999999999999.0,
            "area_det_u": -999999999999.0,
            "area_wse": -999999999999.0,
            "width": -999999999999.0,
            "width_u": -999999999999.0,
            "p_dist_out": -999999999999.0,
            "p_length": -999999999999.0,
            "xtrk_dist": -999999999999.0,
            #"rdr_sig0": -999999999999.0,
            "node_dist": -999999999999.0,
            "n_good_nod": 0,
            #"lat": -999999999999.0,
            #"lon": -999999999999.0,
            "p_lat": -999999999999.0,
            "p_lon": -999999999999.0,
            "time_units": "sec",
            "dark_frac_units": np.int64(1),
            "wse_units": "m",
            "wse_r_u_units": "m",
            "slope_units": "m/m",
            "slope_r_u_units": "m/m",
            "area_total_units": "m^2",
            "area_tot_u_units": "m^2",
            "area_detct_units": "m^2",
            "area_det_u_units": "m^2",
            "area_wse_units": "m^2",
            "width_units": "m",
            "width_u_units": "m",
            "p_dist_out_units": "m",
            "p_length_units": "m",
            "xtrk_dist_units": "m",
            #"rdr_sig0_units":np.int64(1),
            "node_dist_units": "m",
            "flow_angle_units": "degrees",
            "n_good_pix_units": np.int64(1),
            "p_lat_units": "degrees",
            "p_lon_units": "degrees",
            }
    return d

@dask.delayed
def query_hydrocron(
        query_url, feature_id, start_time, end_time, fields, empty_df,
        kind='node_ids', collection_name='SWOT_L2_HR_RiverSP_D'):
    """Query Hydrocron for reach-level time series data.

    Parameters
    ----------
    query_url: str - URL to use to query FTS
    node_id: str - String SWORD reach identifier
    start_time: str - String time to start query
    end_time: str - String time to end query
    fields: list - List of fields to return in query response
    empty_df: pandas.DataFrame that contains empty query results

    Returns
    -------
    pandas.DataFrame that contains query results
    """
    key = "Reach"
    if 'node' in kind:
        key ="Node" 
    params = {
        "feature": key,
        "feature_id": feature_id,
        "output": "csv",
        "start_time": start_time,
        "end_time": end_time,
        "fields": fields,
        "collection_name": collection_name
    }
    results = requests.get(query_url, params=params)
    na_values = ["", 
             "#N/A", 
             "#N/A N/A", 
             "#NA", 
             "-1.#IND", 
             "-1.#QNAN", 
             "-NaN", 
             "-nan", 
             "1.#IND", 
             "1.#QNAN", 
             "<NA>", 
             "N/A", 
#              "NA", 
             "NULL", 
             "NaN", 
             "n/a", 
             "nan", 
             "null"]
    if "results" in results.json().keys():
        results_csv = results.json()["results"]["csv"]
        df = pd.read_csv(StringIO(results_csv), na_values=na_values, keep_default_na=False)
    else:
        df = empty_df

    return df

#def setup_queries(query_url):
def query_main(
        BASIN_IDENTIFIER,
        start_time = "2023-07-28T00:00:00Z",
        end_time = "2024-07-24T00:00:00Z",
        kind = 'node_ids',
        collection_name='SWOT_L2_HR_RiverSP_D'
        ):
    """
    kind = 'node_ids' or 'reach_ids'
    """
    #client = Client(n_workers=4)    # Set to n workers for number of CPUs or parallel processes or set to 1 worker to run things on a single thread
    #client

    # Assign URLs to Variables for the APIs we use, FTS and Hydrocron
    #FTS_URL = "https://fts.podaac.earthdata.nasa.gov/v2"
    #HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v2/timeseries" 
    # get the Version C data (e.g., PGC0)
    FTS_URL = "https://fts.podaac.earthdata.nasa.gov/v1"
    HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
    # BASIN or RIVER to query FTS for
    #BASIN_IDENTIFIER = 74261#742610#732547#"732520" # to search via basin ID, find within SWORD database
    #RIVER_NAME = "Ocmulgee River" #"Rhine"# to search via river name

    # Search by basin code
    query_url = f"{FTS_URL}/rivers/node/{BASIN_IDENTIFIER}"
    if 'reach' in kind:
        query_url = f"{FTS_URL}/rivers/reach/{BASIN_IDENTIFIER}"
    print(f"Searching by basin ...{query_url}")

    # Search by river name
    # query_url = f"{FTS_URL}/rivers/{RIVER_NAME}" #if searching via river name instead
    # print(f"Searching by river name ...{query_url}")
    page_size = 100    # Set FTS to retrieve 100 results at a time
    page_number = 1    # Set FTS to retrieve the first page of results
    hits = 1           # Set hits to intial value to start while loop
    feature_ids = []
    while (page_size * page_number) != 0 and len(feature_ids) < hits:
        params = { "page_size": page_size, "page_number": page_number }
        results = query_fts(query_url, params, kind=kind)

        hits = results['hits']
        page_size = results['page_size']
        page_number = results['page_number'] + 1
        feature_ids.extend(results[kind])

        print("page_size: ", page_size, ", page_number: ", page_number - 1, ", hits: ", hits, ", # feature_ids: ", len(feature_ids))

    print("Total number of features: ", len(feature_ids))
    feature_ids = list(set(feature_ids))    # Remove duplicates
    print("Total number of non-duplicate features: ", len(feature_ids))

    # Create queries that return Pandas.DataFrame objects
    fields = "reach_id,node_id,river_name,time,time_str,crid,pass_id,cycle_id,continent_id,node_q,node_q_b,xovr_cal_q,dark_frac,ice_clim_f,wse,wse_r_u,area_total,area_tot_u,area_detct,area_det_u,area_wse,width,p_dist_out,p_length,xtrk_dist,rdr_sig0,node_dist,flow_angle,n_good_pix,lat,lon"
    if 'reach' in kind:
        fields = "reach_id,river_name,time,time_str,crid,pass_id,cycle_id,continent_id,reach_q,reach_q_b,xovr_cal_q,dark_frac,ice_clim_f,wse,wse_r_u,slope,slope_r_u,area_total,area_tot_u,area_detct,area_det_u,area_wse,width,width_u,p_dist_out,p_length,xtrk_dist,node_dist,n_good_nod,p_lat,p_lon"
    results = []
    for feature in feature_ids:
        # Create an empty dataframe for cases where no data is returned for a reach identifier
        empty_df = pd.DataFrame(empty_feature_dic(feature,kind=kind), index=[0])
        results.append(query_hydrocron(
            HYDROCRON_URL,
            feature,
            start_time,
            end_time,
            fields,
            empty_df,
            kind=kind,
            collection_name=collection_name
            ))
    # Load DataFrame results into dask.dataframe
    ddf = dd.from_delayed(results)
    #ddf.head(n=20, npartitions=len(node_ids))
    #breakpoint()
    try:
        df = ddf.compute()
    except:
        print("##### Problem with ddf.compute() for basin: {}".format(BASIN_IDENTIFIER))
        return -1
    return df

    # print(f"Searching by river name ...{query_url}")
    df = setup_queries(query_url)
    #client.close()
    # remove made up data (the empty dataframe inits above)
    if 'node_q' in df.keys():
        df = df[df['node_q'] < 4]
    elif 'reach_q' in df.keys():
        df = df[df['reach_q'] < 4]
    # remove any data where the time variable is no_data
    df = df[df['time_str'] != "no_data"]
    df = df[df['time'] != 0.0]
    # convert time_str to time_str
    df['time_str'] = pd.to_datetime(df['time_str'])
    return df

def basin_loop(
        basin_list,
        start_time="2023-03-01T00:00:00Z",
        end_time="2028-09-11T00:00:00Z",
        n_workers=4,
        out_csv_name=None,
        kind='Node',
        collection_name='SWOT_L2_HR_RiverSP_D'
        #collection_name='SWOT_L2_HR_RiverSP_2.0' # Version C
        ):
    this_kind = 'reach_ids'
    if 'node' in kind.lower():
        this_kind = 'node_ids'
    # Set to n workers for number of CPUs or parallel processes
    # or set to 1 worker to run things on a single thread
    client = Client(n_workers=n_workers)
    df = None
    #breakpoint()
    for i,basin_id in enumerate(basin_list):
        print("******* processing basin {} of {}".format(i,len(basin_list)))
        this_df = query_main(
            basin_id, start_time=start_time, end_time=end_time, kind=this_kind,
            collection_name=collection_name)
        while isinstance(this_df, int):
            # connection failed try again
            this_df = query_main(basin_id,
                start_time=start_time, end_time=end_time, kind=this_kind,
                collection_name=collection_name)
        if df is None:
            df = this_df
        else:
            df = pd.concat([df, this_df], ignore_index=True)
        # write out each time
        if out_csv_name is not None:
            out_csv_name = out_csv_name.split('.csv')[0]
            tmp_name = '{}_tmp.csv'.format(out_csv_name)
            df.to_csv(tmp_name, index=False)
    if out_csv_name is not None:
        df.to_csv('{}.csv'.format(out_csv_name), index=False)
        # remove the tmp one
        if os.path.isfile(tmp_name):
            os.remove(tmp_name)
    client.close()
    return df


