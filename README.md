# Release Note
Copyright 2024-, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

# river-spatial-scale
Repo for code to estimate river spatial scale parameters (e.g., along-river wse covariance/spectra etc) and apply them in a Bayes reconstruction approach to optimize noise-versus-resolution trade-offs using the multitemporal stack of information from SWOT.

# How to Run
## Defining a Stretch
A "stretch" represents the node-level measurements from ordered consecutive reaches defined by the SWOT prior river database, SWORD (i.e., SWORD defines the reaches and the up- and down-stream connectivity).  In general, a stretch can be any collection of nodes from a subreach to many connected reaches.  The processing starts with the user first defining the stretches to be processed (i.e., the list of ordered, connected reaches and a stretch "name").  To gerenate 3-reach multireach stretches around each reach with the stretch name being the reach name you can call the script:

`$ multireaches_from_sword.py <sword_netcdf_file> <outdir>`

This produces and output named

`<sword_netcdf_file>_multireach.csv`

## Making a Stretch Stack
The process of creating the stretch objects and processing them runs in a few steps based on config files (an example is in the config subdir).  The first script that generates the StretchStack object of the SWOT SP node data is (you can stage the RiverSP data locally or have it be ingested using the hydrochron script):

`$ make_stretch_stack.py <runtime.cfg>`

This creates products/files for each commanded stretch called

`<stretch_name>_stretch_stack.nc`

If the option to ingest the RiverSP data using hte hydrochron tools is commanded in the config file, a csv file with the node data will also be output.

For the stretches defined by multireach\_from\_sword.py the stretch\_name is the reach\_id.

Note that the runtime.cfg file defines which reaches/stretches to create/process as well as the input and output paths to files as well as controls how the SWOT data are obtained/input (e.g., ingested from podaac on the fly or run from predownloaded dataframes or RiverSP files etc).  The same runtime config can be used for multple processing steps and controls how each subprocessor behaves (including pointing to separate, potentially different parmater config files param.cfg).  Examples of these two config files are in the config subdir of the repository, users will need to adjust paths in the runtime config, but should only need to modify the param.cfg in special curcumstances.

## Making Reach-input Objects (Optional)
The stack of reach data can be optionally created from the RiverSP reach data by calling:

`$ make_stretch_average_from_reaches.py <runtime.cfg>`

This creates product(s)/file(s) called:
 
`<reach_id>_width_reach_average.nc`

`<reach_id>_wse_reach_average.nc`

`<reach_id>_height_width_reach_average.nc`

## Making Pekel Products (Optional)
The pekel-derived along-river width statistics can also be optionally created from special 'truth' river processing outputs of the Pekel occurrence maps thresholded at different water occurrence rates.

`$ pekel_width_stats.py <runtime.cfg>`

which produces a product with file-name:

`<stretch_name>_pekel_stats.nc`

## Processing the Stretch Stack
Now all the stretch_stack processing steps can be run from the stretch stack (and optionally the pekel width stats):

`$ process_stretch_stack.py <runtime.cfg>`

This outputs several products/files in the output directory:

`<stretch_name>_wse_stats.nc`

`<stretch_name>_width_stats.nc`

`<stretch_name>_dark_stats.nc`

`<stretch_name>_wse_stretch_average.nc`

`<stretch_name>_width_stretch_average.nc`

`<stretch_name>_height_width.nc`

`<stretch_name>_height_width_array.nc`

`<stretch_name>_flow_state.nc`

`<stretch_name>_bayes.nc`

Note that there are two modes to the process_stretch_stack.py script:

(1) for estimating the prior parameters for Bayes reconstruction (optionally also runing the reconstruction)

(2) for running the reconstruction from the stretch_stack and the already created prior files

Option 1 is the default, but the reconstructing process can be run after estimation of the priors by using the --kind reconstruct option. To be pedantic, we can call the following sequence to create the stretch_stack for multiple stretchs, and then estimate the priors, and then run the Bayes reconstruction:

`$ make_stretch_stack.py <runtime.cfg>`

`$ process_stretch_stack.py <runtime.cfg>`

`$ process_stretch_stack.py <runtime.cfg> --kind reconstruct`



## Plotting and Visualizing
The results/products can be plotted using:

`$ plot_stretch_products.py -c <runtime.cfg>`

If the -s option is given with a particular stretch name, the interactive plots are displayed for the given stretch, otherwise the plots are written to files in the corresponding subdirs.

Alternatively you can use the --infile option to plot a specific product:

`$ pekel_width_stats.py --infile /path/to/product/nc/file`

## Post-processing extraction of desired quantities
After running process_strecth_stack.py over the stretch(es), you can extract the estimated quantities (e.g., reference profiles, along_stats, etc) by calling postprocess_stretch_stack.py the same way you called process_strecth_stack.py.  That is, to get the per-node estimates (e.g., that could be stuffed into SWORD) that crop out the extra up-stream and down-stream info you can call this:

`$ postprocess_stretch_stack.py <runtime.cfg>`

which will produce a sqlite3 database called postproc_estimate.sqlite3 with node_id and various quantites for dark_water, wse, and width profiles/statistics (oriented like in a pandas dataframe).

Also, if you call it after running the reconstruction and also give it the --kind reconstruct, it will produce a database with the bayes-reconstructed wse and width as well as the bundle-adjusted width correction (and per-pass corrected width that is input the bayes reconstruction).  These are data-frame-like too, but oriented by both node_id, granule_id (e.g., a unique estimate for each observation of each node).  This is called like this:

`$ postprocess_stretch_stack.py <runtime.cfg> --kind reconstruct`

This produces a sqlite3 database called postproc_reconstruct.sqlite3.

# info for older code
The hydrochron.py script grabs data for the Ocmulgee, Colorado, and Yellowstone rivers and puts them in a pandas dataframe

The process_river_stretch.py script creates the multitemporal stack of multi-reach stretches an runs Bayes reconstruction
returning RiverSreatchData product instances.


# intial code used in flow wave paper:
Hana R. Thurman, George, H. Allen, Brent A. WIlliams, Arnaud Cerbelaud, and Cedric David "SWOT Captures Hydrologic Waves Traveling Down Rivers," Geophysical Research Letters, 2025 (in Review).

