#!/usr/bin/env python
'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.


This code follows the template from the podaac hydrocron notebook

TODO: make this more general with input reaches/basins to process.
'''
import rivscale.ingest

def main():
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
    end_time = "2026-09-11T00:00:00Z"
    df = rivscale.ingest.basin_loop(
        basin_ids, start_time, end_time,
        out_csv_name='swot_data_all_tmp')

if __name__ == "__main__":
    main()

