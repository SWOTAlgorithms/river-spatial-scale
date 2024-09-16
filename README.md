# river-spatial-scale
Repo for code to estimate river spatial scale parameters (e.g., along-river wse covariance/spectra etc)

The hydrochron.py script grabs data for the Ocmulgee, Colorado, and Yellowstone rivers and puts them in a pandas dataframe

The process_river_stretch.py script creates the multitemporal stack of multi-reach stretches an runs Bayes reconstruction
returning RiverSreatchData product instances.


# intial code for proposal
This is the description for the state of the code at the time of the original science team proposal submission

The main script that stacks multitemporal SWOT data for one tile and estimates the mean wse profile and
covariance is bin/analyze_networks.py  

The scipt can be called like this:

python bin/analyze_networks.py <field_datafreme_dir> <rivertile_dir> <sword_file>


Yellowstone example:

python bin/analyze_networks.py /Users/bawillia/Desktop/plots/time-series/pt_drift_testing/NS_node_dataframes/node ~/Desktop/data/Yellowstone/ ~/Desktop/data/Yellowstone/SWOT_RiverDatabase_Cal_024_070R_20200101T000000_21000101T000000_20230808T191400_v216.nc


