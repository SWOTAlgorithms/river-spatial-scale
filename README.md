# river-spatial-scale
Repo for code to estimate river spatial scale parameters (e.g., along-river wse covariance/spectra etc)

The main script that stacks multitemporal SWOT data for one tile and estimates the mean wse profile and
covariance is bin/analyze_networks.py  

The scipt can be called like this:

python bin/analyze_networks.py <field_datafreme_dir> <rivertile_dir> <sword_file>


Yellowstone example:
python bin/analyze_networks.py /Users/bawillia/Desktop/plots/time-series/pt_drift_testing/NS_node_dataframes/node ~/Desktop/data/Yellowstone/ ~/Desktop/data/Yellowstone/SWOT_RiverDatabase_Cal_024_070R_20200101T000000_21000101T000000_20230808T191400_v216.nc


