#!/usr/bin/env python
'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.


This code follows the template from the podaac hydrocron notebook

TODO: make this more general with input reaches/basins to process.
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

def query_fts(query_url, params):
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

    hits = reaches_json['hits']
    if 'search on' in reaches_json.keys():
        page_size = reaches_json['search on']['page_size']
        page_number = reaches_json['search on']['page_number']
    else:
        page_size = 0
        page_number = 0

    return {
        "hits": hits,
        "page_size": page_size,
        "page_number": page_number,
        "node_ids": [ item['node_id'] for item in reaches_json['results'] ]
    }

@dask.delayed
def query_hydrocron(query_url, node_id, start_time, end_time, fields, empty_df):
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

    params = {
        "feature": "Node",
        "feature_id": node_id,
        "output": "csv",
        "start_time": start_time,
        "end_time": end_time,
        "fields": fields
    }
    results = requests.get(query_url, params=params)
    if "results" in results.json().keys():
        results_csv = results.json()["results"]["csv"]
        df = pd.read_csv(StringIO(results_csv))
    else:
        df = empty_df

    return df

#def setup_queries(query_url):
def query_main(BASIN_IDENTIFIER, start_time = "2023-07-28T00:00:00Z", end_time = "2024-07-24T00:00:00Z"):
    #client = Client(n_workers=4)    # Set to n workers for number of CPUs or parallel processes or set to 1 worker to run things on a single thread
    #client

    # Assign URLs to Variables for the APIs we use, FTS and Hydrocron
    FTS_URL = "https://fts.podaac.earthdata.nasa.gov/v1"
    HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"

    # BASIN or RIVER to query FTS for
    #BASIN_IDENTIFIER = 74261#742610#732547#"732520" # to search via basin ID, find within SWORD database
    #RIVER_NAME = "Ocmulgee River" #"Rhine"# to search via river name

    # Search by basin code
    query_url = f"{FTS_URL}/rivers/node/{BASIN_IDENTIFIER}"
    print(f"Searching by basin ...{query_url}")

    # Search by river name
    # query_url = f"{FTS_URL}/rivers/{RIVER_NAME}" #if searching via river name instead
    # print(f"Searching by river name ...{query_url}")
    page_size = 100    # Set FTS to retrieve 100 results at a time
    page_number = 1    # Set FTS to retrieve the first page of results
    hits = 1           # Set hits to intial value to start while loop
    node_ids = []
    while (page_size * page_number) != 0 and len(node_ids) < hits:
        params = { "page_size": page_size, "page_number": page_number }
        results = query_fts(query_url, params)

        hits = results['hits']
        page_size = results['page_size']
        page_number = results['page_number'] + 1
        node_ids.extend(results['node_ids'])

        print("page_size: ", page_size, ", page_number: ", page_number - 1, ", hits: ", hits, ", # node_ids: ", len(node_ids))

    print("Total number of nodes: ", len(node_ids))
    node_ids = list(set(node_ids))    # Remove duplicates
    print("Total number of non-duplicate nodes: ", len(node_ids))

    # Create queries that return Pandas.DataFrame objects
    #start_time = "2023-07-28T00:00:00Z"
    #end_time = "2024-07-24T00:00:00Z"
    fields = "reach_id,node_id,river_name,time,time_str,crid,pass_id,cycle_id,node_q,node_q_b,xovr_cal_q,dark_frac,ice_clim_f,wse,wse_r_u,area_total,area_tot_u,area_detct,area_det_u,area_wse,width,p_dist_out,xtrk_dist,rdr_sig0,node_dist,flow_angle,n_good_pix,lat,lon"
    results = []
    for node in node_ids:
        # Create an empty dataframe for cases where no data is returned for a reach identifier
        empty_df = pd.DataFrame({
            "reach_id": np.int64(0),
            "node_id": np.int64(node),
            "river_name": "no_data",
            "time": 0.0,
            "time_str": datetime.datetime(1900, 1, 1).strftime("%Y-%m-%dT%H:%M:%S"),
            "crid":'PGC0',
            "pass_id": 0,
            "cycle_id": 0,
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
            "xtrk_dist_units": "m",
            "rdr_sig0_units":np.int64(1),
            "node_dist_units": "m",
            "flow_angle_units": "degrees",
            "n_good_pix_units": np.int64(1),
            "lat_units": "degrees",
            "lon_units": "degrees",
        }, index=[0])
        results.append(query_hydrocron(HYDROCRON_URL, node, start_time, end_time, fields, empty_df))
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
    df = df[df['node_q'] < 4]
    # remove any data where the time variable is no_data
    df = df[df['time_str'] != "no_data"]
    df = df[df['time'] != 0.0]
    # convert time_str to time_str
    df['time_str'] = pd.to_datetime(df['time_str'])
    return df

def main():
    client = Client(n_workers=4)    # Set to n workers for number of CPUs or parallel processes or set to 1 worker to run things on a single thread
    client
    """
    #basin_id_base = 74261
    #basin_ids = basin_id_base + np.array([r for r in range(2)])
    ohio_df = pd.read_csv('SWOT-GLOW_S_Alaska_fixed.csv')
    
    end_time = "2024-09-10T00:00:00Z"
    ## just look up all nodes individually
    #basins = np.array(ohio_df['node_id']).astype('int')
    # just grab all nodes in given basin (one level above reach)
    basins = np.array(ohio_df['node_id']/100000).astype('int')
    basin_ids = np.unique(basins)
    #basin_ids = [742628000,]
    """
    ocmulgee = 732547
    yellowstone1 = 742979
    yellowstone2 = 742981
    colorado = 75181000
    basin_ids = [ocmulgee, yellowstone1, yellowstone2, colorado]
    start_time = "2023-03-01T00:00:00Z"
    end_time = "2024-09-11T00:00:00Z"
    print(basin_ids)
    df = None
    for i,basin_id in enumerate(basin_ids):
        print("******* processing basin {} of {}".format(i,len(basin_ids)))
        this_df = query_main(basin_id, start_time=start_time, end_time=end_time)
        while isinstance(this_df, int):
            # connection field try again
            this_df = query_main(basin_id)
        if df is None:
            df = this_df
        else:
            df = pd.concat([df, this_df], ignore_index=True)
        # write out each time
        df.to_csv('swot_data_all_tmp.csv'.format(), index=False)
    df.to_csv('swot_data_all.csv'.format(), index=False)
    client.close()
    #breakpoint()

if __name__ == "__main__":
    main()

