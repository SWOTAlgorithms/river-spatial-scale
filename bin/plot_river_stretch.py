#!/usr/bin/env python
'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author(s): Brent Williams

'''

import pandas as pd
import numpy as np
import rivscale.plot
import rivscale.products
import matplotlib.pyplot as plt
def main():
    #stretch_names = ['Ocmulgee', 'Colorado', 'Yellowstone']
    #stretch_names = ['Willamette', 'Connecticut', 'North_Sask', 'Yukon', 'Garonne', 'Waimak']
    #stretch_names = ['Waimak',]
    stretch_names = ['Willamette',]
    outdir = 'Willamette_plots'
    #outdir = 'calval_plots'
    for name in stretch_names:
        # read in the already processed data
        stretch_data = rivscale.products.RiverStretchData.from_ncfile(
            '{}_stretch.nc'.format(name))
        #breakpoint()
        rivscale.plot.plot_stretch(stretch_data, title=name, outdir=outdir)
        #plt.show()
        #breakpoint()

if __name__ == "__main__":
    main()

