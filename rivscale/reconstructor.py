'''
Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent Williams

This class defines a processor for running Bayes reconstruction giving the input 
'''

import logging
import os.path

import rivscale
import rivscale.io
import rivscale.estimate
import rivscale.data
import rivscale.filter
import rivscale.products.stretch_stack
import rivscale.products.along_stretch
import rivscale.products.stretch_average
import rivscale.products.height_width
import rivscale.products.flow_state
import rivscale.products.height_width_array
import rivscale.misc

LOGGER = logging.getLogger(__name__)

WARN_STR = '        already processed, not rerunning'

class Reconstructor(object):
    '''
    Turn a stretch_stack into multiple products holding information
    for a specific stretch aggregated over time.  These products include:
    1) along-river wse/width reference profiles and statistics
    2) along-river spatial covariance estimates (wse and width)
    3) height/width relationships (at node and at stretch-level)
    4) flow-state model giving typical along-river info for wse and width
       at multiple flow-states (e.g., binned in stretch-average wse bins)
    
    These are intended to be used in Bayes reconstruction on a per-pass-obs
    basis, but are also directly useful for science investigations
    '''
    def __init__(self, run_config, force=False):
        # if it is a filename read it into an object
        if isinstance(run_config, str):
            run_config = rivscale.misc.CfgParser()
            run_config.read(run_config)
        self.cfg_run =  run_config
        self.force = force
        self.cfg_param = rivscale.misc.CfgParser()
        # create a container to hold products
        self.products = {}
        # create a container for subprocessors
        self.processor_list = {}

        self.state = 'init'
        self.completed_sucessfully = False
        # load the data and set if it is runable
        self.isrunable = self.load_data()
        #
        
    def need_to_run(self):
        '''check if all output files already exist'''
        need = True
        if len(self.processor_list)>0:
            last_processor = self.processor_list[-1]
            # try each case for a valid object for the last product 
            if last_processor=='reconstruct':
                if self.products['bayes'] is not None:
                    need = False
            elif last_processor=='smooth_widths':
                if self.products['stretch_stack_smoothwidth'] is not None:
                    need = False
            elif last_processor=='width_correction':
                if self.products['stretch_stack_corr'] is not None:
                    need = False
            elif last_processor=='filter_stack':
                if self.products['stretch_stack_filt'] is not None:
                    need = False

        return need

    def load_param_config(self):
        success = True
        # try to read tha param config
        try:
            # read the param cfg
            self.cfg_param.read(self.cfg_run['main']['param_config'])
        except (FileNotFoundError, KeyError) as e:
            LOGGER.info(f'    Problem loading param config: {e}')
            success = False
        # check that output location exists if not try to create it
        out_path = None
        try:
            out_path = self.cfg_run['main']['out_path']
        except KeyError as e:
            LOGGER.info(f'    no out_path key in config: {e}')
            success = False
        if out_path is not None:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        ####
        # get subprocessor list in the order listed in in the
        # param config section headings
        ####
        # get processor list from section heading names
        self.processor_list = list(self.cfg_param.keys())
        # exclude the stretch-stack, stretch_avg, and pekel etc
        # assume we start processing with dark_stats
        # TODO: refine this
        #breakpoint()
        rm_keys = ['DEFAULT', 'main']
        for key in rm_keys:
            if key in self.processor_list:
                self.processor_list.remove(key)
        #breakpoint()
        return success

    def load_inputs(self):
        success = True
        # load the input file
        try:
            stretch_stack = \
                rivscale.products.stretch_stack.StretchStack.from_ncfile(
                    self.cfg_run['main']['stretch_stack_file'])
        except (FileNotFoundError, KeyError) as e:
            LOGGER.info(f'    Problem loading stretch_stack file: {e}')
            return False
        # populate witdh_u
        # TODO: fix the uncertainty itself instead of fudging it here  
        node_len = stretch_stack['area_total'] / stretch_stack['width']
        stretch_stack['width_u'] = stretch_stack['area_tot_u'] / node_len
        # make measurement uncert at least as much as signal uncert we assume
        stretch_stack['width_u'] = stretch_stack['width_u'] + 10
        # stuff it into the products container
        self.products['stretch_stack'] = stretch_stack

        # TODO: handle reach_avg and pekel
        # load the output files if they already exist
        # get a list of output product names
        product_list = [
            'wse_stats',
            'width_stats',
            'wse_avg',
            'width_avg',
            'height_width',
            'flow_state',
            'height_width_array',
            ]
        # create the files
        infiles = {}
        for prod_key in product_list:
            try:
                infiles[prod_key] = os.path.join(
                    self.cfg_run['main']['estimate_path'], '{}_{}.nc'.format(
                        self.cfg_run['main']['stretch_name'], prod_key))
            except KeyError as e:
                self.outfiles[prod_key] = None
        ####
        # initialize products, and try to read if exists
        ####
        # init the output products to None
        for prod_key in product_list:
            self.products[prod_key] = None
        # now go through and try to load the products if they already exist
        for prod_key in product_list:
            # try to load the file for this product
            this_prod = None
            if infiles[prod_key] is not None:
                #try to read it
                this_prod = rivscale.io.load_product(
                    infiles[prod_key], prod_key, False)
            if this_prod is None:
                success = False
            self.products[prod_key] = this_prod
        return success

    def load_outputs(self):
        # load the output files if they already exist
        # get a list of output product names
        self.product_list = [
            'stretch_stack_corr',
            'stretch_stack_filt',
            'stretch_stack_smoothwidth',
            'bayes',
            ]
        # create the output files
        self.outfiles = {}
        for prod_key in self.product_list:
            try:
                self.outfiles[prod_key] = os.path.join(
                    self.cfg_run['main']['out_path'], '{}_{}.nc'.format(
                        self.cfg_run['main']['stretch_name'], prod_key))
            except KeyError as e:
                self.outfiles[prod_key] = None
        ####
        # initialize products, and try to read if exists
        ####
        # init the output products to None
        for prod_key in self.product_list:
            self.products[prod_key] = None
        # now go through and try to load the products if they already exist
        # unless force is set
        for prod_key in self.product_list:
            # try to load the file for this product
            this_prod = None
            if self.outfiles[prod_key] is not None:
                #try to read it
                this_prod = rivscale.io.load_product(
                    self.outfiles[prod_key], prod_key, self.force)
            self.products[prod_key] = this_prod

    def load_data(self):
        # load the param config
        if not self.load_param_config():
            return False
        # load the input file
        if not self.load_inputs():
            return False
        # now load the outputs
        self.load_outputs()
        return True

    def run(self):
        '''run the worker to process a single stretch'''
        LOGGER.info('    Running the reconstruct Bayes processor')
        #breakpoint()
        if not self.isrunable:
            LOGGER.info('    processor not runable')
            return False
        all_success = True
        for processor_name in self.processor_list:
            success = self.run_processor(processor_name)
            if not(success):
                # return without processing rest
                all_success = False
                break
        return all_success

    def run_processor(self, processor_name):
        LOGGER.info(f'     processing: {processor_name}') 
        success = False
        if processor_name=='width_correction':
            success = self.run_width_correction(self.cfg_param[processor_name])
        elif processor_name=='filter_stack':
            success = self.run_filter_stack(self.cfg_param[processor_name])
        elif processor_name=='smooth_widths':
            success = self.run_smooth_widths(self.cfg_param[processor_name])
        elif processor_name=='reconstruct':
            success = self.run_reconstruct(self.cfg_param[processor_name])
        return success

    # define subprocessors
    def run_width_correction(self, cfg):
        corr_stack = self.products['stretch_stack_corr']
        if (self.products['stretch_stack_corr'] is not None) and (not self.force):
            LOGGER.info(WARN_STR)
        else:
            corr_stack = rivscale.estimate.process_width_correction(
                cfg, self.products['stretch_stack'])
            if corr_stack is not None:
                corr_stack.to_ncfile(self.outfiles['stretch_stack_corr'])
            # update products
            self.products['stretch_stack_corr'] = corr_stack
        if cfg['use_as_working_stack']:
            self.products['stretch_stack'] = corr_stack.copy()
        return True

 
    def run_filter_stack(self, cfg):
        filt_stack = self.products['stretch_stack_filt']
        if (self.products['stretch_stack_filt'] is not None) and (not self.force):
            LOGGER.info(WARN_STR)
        else:
            filt_stack = rivscale.estimate.process_filter_stack(
                cfg, self.products['stretch_stack'])
            if filt_stack is None:
                return False # return bad exist status
            filt_stack.to_ncfile(self.outfiles['stretch_stack_filt'])
            self.products['stretch_stack_filt'] = filt_stack
        if cfg['use_as_working_stack']:
            self.products['stretch_stack'] = filt_stack.copy()
        return True

    def run_smooth_widths(self, cfg):
        if (self.products['stretch_stack_smoothwidth'] is not None) and (not self.force):
            LOGGER.info(WARN_STR)
        else:
            corr_stack = rivscale.estimate.process_smooth_widths(
                cfg, self.products['stretch_stack'])
            if corr_stack is not None:
                corr_stack.to_ncfile(self.outfiles['stretch_stack_smoothwidth'])
            # update products
            self.products['stretch_stack_smoothwidth'] = corr_stack
        return True

    def run_reconstruct(self, cfg):
        if (self.products['bayes'] is not None) and (not self.force):
            LOGGER.info(WARN_STR)
        else:
            bayes = rivscale.reconstruct.process_bayes_reconstruction(
                cfg,
                self.products['stretch_stack'],
                self.products['wse_stats'],
                self.products['width_stats'],# TODO: handle Pekel
                self.products['height_width'])
            if bayes is not None:
                bayes.to_ncfile(self.outfiles['bayes'])
            self.products['bayes'] = bayes
        return True

