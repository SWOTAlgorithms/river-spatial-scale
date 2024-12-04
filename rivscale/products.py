'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author (s): Brent Williams

Mimics the product class from RiverObs and the SWOT project "python" repo
'''
from collections import OrderedDict as odict
import textwrap
import numpy as np
from datetime import datetime

import swot.constants
from swot.product import Product, ProductTesterMixIn
from SWOTWater.products.constants import FILL_VALUES
from swot.lr.base_classes import AttrFillerMixIn

def textjoin(text):
    """Dedent join and strip text"""
    text = textwrap.dedent(text)
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

#DIMENSIONS_4D = odict([
#    ['num_times', 0], ['num_nodes', 0], ['num_reaches',0], ['num_percentiles', 0]])
#DIMENSIONS_2D = odict([['num_times', 0], ['num_nodes', 0]])
#DIMENSIONS_PCNT = odict([['num_percentiles', 0], ['num_nodes', 0]])
#DIMENSIONS_COV = odict([['num_times', 0], ['num_nodes', 0], ['num_nodes', 0]])

DIMENSIONS_ALL = odict([
    ['num_nodes', 0],
    ['num_times', 0],
    ['num_reaches',0],
    ['num_percentiles', 0],
    ['num_nodes2', 0]])
DIMENSIONS_2D = odict([['num_nodes', 0], ['num_times', 0]])
DIMENSIONS_PCNT = odict([['num_nodes', 0], ['num_percentiles', 0]])
DIMENSIONS_COV = odict([['num_nodes', 0], ['num_nodes2', 0]])
DIMENSIONS_POSTCOV = odict([['num_nodes', 0], ['num_nodes2', 0], ['num_times', 0]])

class RiverStretchData(Product):
    ATTRIBUTES = odict([
        ['description',{'dtype':'str', 'value': textjoin("""
            Container for potentially multireach sections of rivers
            holding 2D multitemporal data and Bayes reconstruction
            processing parameters and results
            """)}],
        ])
    DIMENSIONS = DIMENSIONS_ALL
    VARIABLES = odict([
        ['stretch_reaches', odict([['dimensions', odict([['num_reaches', 0]])]])],
        ['dist_out', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['local_node_id', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['time_id', odict([['dimensions', odict([['num_times', 0]])]])],
        ['cycle_id', odict([['dimensions', odict([['num_times', 0]])]])],
        #['swot_time', odict([['dimensions', odict([['num_times', 0]])]])],
        ['date_hour', odict([['dimensions', odict([['num_times', 0]])]])],
        ['wse', odict([['dimensions', DIMENSIONS_2D]])],
        ['width', odict([['dimensions', DIMENSIONS_2D]])],
        ['area_total', odict([['dimensions', DIMENSIONS_2D]])],
        ['wse_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['width_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['area_tot_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['node_q_b', odict([['dimensions', DIMENSIONS_2D]])],
        ['bayes_wse', odict([['dimensions', DIMENSIONS_2D]])],
        ['bayes_width', odict([['dimensions', DIMENSIONS_2D]])],
        ['bayes_wse_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['bayes_width_u', odict([['dimensions', DIMENSIONS_2D]])],
        ['bayes_wse_post_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['bayes_width_post_cov', odict([['dimensions', DIMENSIONS_POSTCOV]])],
        ['wse_reference', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_reference', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_cov', odict([['dimensions', DIMENSIONS_COV]])],
        ['width_cov', odict([['dimensions', DIMENSIONS_COV]])],
        ['wse_mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_mean', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_std', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_std', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_count', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['width_count', odict([['dimensions', odict([['num_nodes', 0]])]])],
        ['wse_percentiles', odict([['dimensions', DIMENSIONS_PCNT]])], 
        ['width_percentiles', odict([['dimensions', DIMENSIONS_PCNT]])],
        ['percentiles', odict([['dimensions', odict([['num_percentiles', 0]])]])],
        
    ])
    #for name, reference in VARIABLES.items():
    #    reference['dimensions'] = DIMENSIONS

