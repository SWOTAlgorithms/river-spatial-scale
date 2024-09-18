#!/usr/bin/env python
'''
Copyright (c) 2024-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Brent Williams

'''

import pandas as pd
import numpy as np
import rivscale.plot
import rivscale.products
import matplotlib.pyplot as plt
def main():
    #stretch_names = ['Ocmulgee', 'Colorado', 'Yellowstone']
    stretch_names = ['Willamette', 'Connecticut', 'North_Sask', 'Yukon', 'Garonne', 'Waimak']
    #stretch_names = ['Waimak',]
    for name in stretch_names:
        # read in the already processed data
        stretch_data = rivscale.products.RiverStretchData.from_ncfile(
            '{}_stretch.nc'.format(name))
        #breakpoint()
        rivscale.plot.plot_stretch(stretch_data, title=name)
        plt.show()
        breakpoint()

if __name__ == "__main__":
    main()

