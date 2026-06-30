#!/usr/bin/env python3
'''
Copyright 2026, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

This code postprocesses the river streaches created by process_stretch_stack
into either a SWORD-like per-node database (for the estimate quantities/stats),
or a node-level per-pass-observation oriented database with the bundle-adjusted
corrected widths and reconstructed wse and widths.  Note that you first need to
process the stretch_stack.
'''

import pandas as pd
import numpy as np
import os.path
import argparse
import warnings
import rivscale.postprocess
import sqlite3

EXAMPLE=''

def main():
    parser = argparse.ArgumentParser(
        description='Process river stretch, reach, or multireach',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)
    parser.add_argument('config', help='config file')
    parser.add_argument('--force', default=False, action='store_true',
        help='force rerun and overwriting of output files')
    parser.add_argument(
        '-l', '--log-level', type=str, default="info",#default="debug",
        help="logging level, one of: debug info warning error")
    parser.add_argument('--kind', default='estimate',
        help='estimate or reconstruct')
    parser.add_argument('--outfile', default=None,
        help='output file name')
    #
    args = parser.parse_args()
    df = None
    outfile = None
    if args.outfile is None:
        outfile = f'postproc_{args.kind}.sqlite3'
    else:
        outfile = args.outfile
    if (os.path.exists(outfile)) and not args.force:
        print(f"  Output file {os.path.abspath(outfile)}"
                "\n    already exists, not rerunning",
                "\n    use --force if you want to force it to rerun")
        return
    #
    table_name = 'node_stats'
    this_outfile = None
    if args.kind=='reconstruct':
        table_name = 'node_data'
        this_outfile = outfile # accumulate on disc for "reconstruct"
    #
    df = rivscale.postprocess.accumulate_processed_df(
            args.config, kind=args.kind,
            outfile=this_outfile, table_name=table_name)
    #
    ####
    # write the sqlite3 database table
    ####
    if df is not None:
        conn = sqlite3.connect(outfile)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

