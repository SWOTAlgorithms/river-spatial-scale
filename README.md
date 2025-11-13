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

`$ make_stretch_stack.py <config.cfg>`

This creates products/files for each commanded stretch called

`<stretch_name>_stretch_stack.nc`

If the option to ingest the RiverSP data using hte hydrochron tools is commanded in the config file, a csv file with the node data will also be output.

For the stretches defined by multireach\_from\_sword.py the stretch\_name is the reach\_id.

## Making Reach-input Objects (Optional)
The stack of reach data can be optionally created from the RiverSP reach data by calling:

`$ make_stretch_average_from_reaches.py <reach_avg.cfg>`

This creates product(s)/file(s) called:
 
`<reach_id>_width_reach_average.nc`

`<reach_id>_wse_reach_average.nc`

`<reach_id>_height_width_reach_average.nc`

## Making Pekel Products (Optional)
The pekel-derived along-river width statistics can also be optionally created from special 'truth' river processing outputs of the Pekel occurrence maps thresholded at different water occurrence rates.

`$ pekel_width_stats.py <stretch_stack.cfg>`

which produces a product with file-name:

`<stretch_name>_pekel_stats.nc`

## Processing the Stretch Stack
Now all the stretch_stack processing steps can be run from the stretch stack (and optionally the pekel width stats):

`$ process_stretch_stack.py <process.cfg>`

This outputs several products/files in the output directory:

`<stretch_name>_wse_stats.nc`

`<stretch_name>_width_stats.nc`

`<stretch_name>_dark_stats.nc`

`<stretch_name>_wse_stretch_average.nc`

`<stretch_name>_width_stretch_average.nc`

`<stretch_name>_height_width.nc`

`<stretch_name>_bayes.nc`

## Plotting and Visualizing
The results/products can be plotted using:

`$ plot_stretch_products.py -c <process.cfg>`

If the -s option is given with a particular stretch name, the interactive plots are displayed for the given stretch, otherwise the plots are written to files in the corresponding subdirs.

## Processing the Stretch Stack to Estimate Height/Width relationships on a per-node basis
A separate workflow has been created to estimate the height/width relationships for every node in a stretch/reach.  This approach optionally applies a per-pass width correction up front, then does quality and outlier filtering after which height/width relationships are estimated using percentiles for height and width for each node using only wse and width node estimates where both are valid and pass the quality filters.  There is also an option to estimate the percentiles over a multinode window for each node, effectively smoothing/relgularizing in the along river dimension to improve the statistics/estimates at the expense of spatial resolution.  There is a config called height_width_array.cfg in the config dir that can be used to set the various parameters of each step.

The now workflow can be called from the command-line like this:

`$ make_height_width_array.py <height_width_array.cfg>`

The plot_stretch_products.py script can also be used to plot a 3D visualization of the height/width estimates along the channel:

`$ plot_stretch_products.py -c <height_width_array.cfg> -s <stretch_name(i.e., reach_id)>`

# info for older code
The hydrochron.py script grabs data for the Ocmulgee, Colorado, and Yellowstone rivers and puts them in a pandas dataframe

The process_river_stretch.py script creates the multitemporal stack of multi-reach stretches an runs Bayes reconstruction
returning RiverSreatchData product instances.


# intial code used in flow wave paper:
Hana R. Thurman, George, H. Allen, Brent A. WIlliams, Arnaud Cerbelaud, and Cedric David "SWOT Captures Hydrologic Waves Traveling Down Rivers," Geophysical Research Letters, 2025 (in Review).

