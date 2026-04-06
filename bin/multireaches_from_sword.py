#!/usr/bin/env python3
'''
Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams
'''

import numpy as np
import pandas as pd
import argparse
import os
import os.path
import rivscale.io
EXAMPLE=''

def main():
    parser = argparse.ArgumentParser(
        description='Calculate River multi-reach for each reach',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('sword_file', help='SWORD netcdf file')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()
    print('reading SWORD file:',args.sword_file)
    # create outdir if it doesnt exist
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    # read SWORD input granule
    sword_df, sword_node_df, d_up, d_down = rivscale.io.read_SWORD(
        args.sword_file)
    # make multireach stretch for each reach
    # skip (disconnected lake, dam, unrealizable topology) reaches
    skip_types = [3, 4, 5]
    stretch_d ={}
    up_reach_id_arr = np.zeros_like(np.array(sword_df['reach_id']))
    down_reach_id_arr = np.zeros_like(np.array(sword_df['reach_id']))
    N = len(sword_df['reach_id'])
    for k,rch in enumerate(sword_df['reach_id']):
        if np.mod(k, 1000)==0:
            print(f'reach {k} of {N}')
        this_up_id = 0
        for up_id in d_up['{}'.format(rch)]:
            if not up_id % 10 in skip_types:
                this_up_id = up_id
                #breakpoint()
                up_reach_id_arr[k] = this_up_id
                break
        this_down_id = 0
        for down_id in d_down['{}'.format(rch)]:
            if not down_id % 10 in skip_types:
                this_down_id = down_id
                down_reach_id_arr[k] = this_down_id
                break
        # make a dataframe with the multi-reach list of reach-ids
        this_stretch = [this_down_id, rch, this_up_id]
        stretch_d['{}'.format(rch)] = this_stretch
    stretch_df = pd.DataFrame(stretch_d)
    sword_df['up_reach_id'] = up_reach_id_arr
    sword_df['down_reach_id'] = down_reach_id_arr
    # Write out the csv file for each multi-reach
    head, tail = os.path.split(args.sword_file)
    outname = tail.replace('.nc', '_multireach.csv')
    outfile = os.path.join(args.outdir, outname)
    stretch_df.to_csv(outfile, index=False)
    # write out the SWORD dataframe with selecetd up and
    #     downstream for each reach
    outname = tail.replace('.nc', '.csv')
    outfile2 = os.path.join(args.outdir, outname)
    # TODO: make dirs if not exist
    sword_df.to_csv(outfile2, index=False)

if __name__ == '__main__':
    main()

