'''
Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author (s): Brent Williams

A place for common constants among the various products
'''
from collections import OrderedDict as odict

DIMENSIONS_ALL = odict([
    ['num_nodes', 0],
    ['num_times', 0],
    ['num_reaches', 0],
    ['num_percentiles', 0],
    ['num_nodes2', 0],
    ['num_hw_params', 0]])
DIMENSIONS_SAVG = odict([
    ['num_reaches', 0],
    ['num_times', 0],
    ['num_percentiles', 0]])
DIMENSIONS_ALONG = odict([
    ['num_reaches', 0],
    ['num_nodes', 0],
    ['num_percentiles', 0]])
DIMENSIONS_2D = odict([
    ['num_nodes', 0],
    ['num_times', 0]])
DIMENSIONS_PCNT = odict([
    ['num_nodes', 0],
    ['num_percentiles', 0]])
DIMENSIONS_PCNT2 = odict([
    ['num_times', 0],
    ['num_percentiles', 0]])
DIMENSIONS_COV = odict([
    ['num_nodes', 0],
    ['num_nodes2', 0]])
DIMENSIONS_POSTCOV = odict([
    ['num_nodes', 0],
    ['num_nodes2', 0],
    ['num_times', 0]])
DIMENSIONS_NTR = odict([
    ['num_nodes', 0],
    ['num_times', 0],
    ['num_reaches', 0]])

PERCENTILES = [5, 25, 32, 50, 68, 75, 95]

