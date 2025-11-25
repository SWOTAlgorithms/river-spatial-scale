'''
Copyright 2025, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Author: Brent Williams

This class defines a processor for estimating the 
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

class Worker(object):
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
        self.runable = True
        self.cfg_param = rivscale.misc.CfgParser()
        # create a container to hold products
        self.products = {}
        # create a container for subprocessors
        self.processors = {}

        self.state = 'init'
        self.completed_sucessfully = False
        # check that it is runable
        self.runable = self.isrunable()
        
        # get a list of output product names
        self.product_list = [
            'stretch_stack_corr',
            'dark_stats',
            'stretch_stack_filt',
            'stretch_stack_smoothwidth',
            'wse_stats',
            'width_stats',
            'wse_avg',
            'width_avg',
            'height_width',
            'flow_state',
            'height_width_array',
            'bayes',
            ]
        # create the output files
        self.outfiles = {}
        for prod_key in self.product_list:
            self.outfiles[prod_key] = os.path.join(
                self.cfg_run['main']['out_path'], '{}_{}.nc'.format(
                    self.cfg_run['main']['stretch_name'], prod_key))

        ####
        # initialize products, and try to read if exists
        ####
        self.products['stretch_stack'] = None
        # init the output products to None
        for prod_key in self.product_list:
            self.products[prod_key] = None
        # now go through and try to load the products if they alreay exist
        # unless force is set
        for prod_key in self.product_list:
            # try to load the file for this product
            this_prod = rivscale.io.load_product(
                self.outfiles[prod_key], prod_key, self.force)
            self.products[prod_key] = this_prod

    def isrunable(self):
        runable  = True
        out_path = self.cfg_run['main']['out_path']
        if not os.path.exists(out_path):
            LOGGER.info(f'creating out_path: {out_path}')
            os.makedirs(out_path)
        # TODO: should we check for key error here too? 
        stretch_file = self.cfg_run['main']['stretch_stack_file']
        if not os.path.exists(stretch_file):
            LOGGER.info(f'input stretch_stack file does not exist: {stretch_file}')
            runable = False
        param_config = self.cfg_run['main']['param_config']
        if not os.path.exists(stretch_file):
            LOGGER.info(f'input param_config file does not exist: {param_config}')
            runable=False
        return runable

    def load_param_config(self):
        # try to read tha param config
        self.param_cfg = rivscale.misc.CfgParser()
        try:
            # read the param cfg
            self.cfg_param.read(self.cfg_run['main']['param_config'])
        except (FileNotFoundError, KeyError) as e:
            LOGGER.info(f'    Problem loading param config: {e}')
            self.runable = False
            return

    def need_to_run(self):
        '''check if all output files already exist'''
        

    def run(self):
        '''run the worker to process a single stretch'''
        LOGGER.info('    Running the estimate priors processor')
        self.load_param_config()
        if not self.runable:
            print('processor not runable')
            LOGGER.info('    processor not runable')
            return False
        ####
        # read the input stretch stack data
        ####
        try:
            stretch_stack = \
                rivscale.products.stretch_stack.StretchStack.from_ncfile(
                    self.cfg_run['main']['stretch_stack_file'])
        except (FileNotFoundError, KeyError) as e:
            LOGGER.info(f'    Problem loading stretch_stack file: {e}')
            self.runable = False
            return False
        # populate witdh_u
        # TODO: fix the uncertainty itself instead of fudging it here  
        node_len = stretch_stack['area_total'] / stretch_stack['width']
        stretch_stack['width_u'] = stretch_stack['area_tot_u'] / node_len
        # make measurement uncert at least as much as signal uncert we assume
        stretch_stack['width_u'] = stretch_stack['width_u'] + 10
        
        # stuff it into the products container
        self.products['stretch_stack'] = stretch_stack
        
        # TODO: handle reach_avg and Pekel

        ####
        # now process the subprocessors in the order listed in in the
        # param config section headings
        ####
        # get processor list from section heading names
        processor_list = list(self.cfg_param.keys())
        # exclude the stretch-stack, stretch_avg, and pekel etc
        # assume we start processing with dark_stats
        # TODO: refine this
        self.processor_list = processor_list[processor_list.index('dark_stats'):]
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
        elif processor_name=='dark_stats':
            success = self.run_dark_stats(self.cfg_param[processor_name])
        elif processor_name=='filter_stack':
            success = self.run_filter_stack(self.cfg_param[processor_name])
        elif processor_name=='smooth_widths':
            success = self.run_smooth_widths(self.cfg_param[processor_name])
        elif processor_name=='along_stats':
            success = self.run_along_stats(self.cfg_param[processor_name])
        elif processor_name=='stretch_average':
            success = self.run_stretch_average(self.cfg_param[processor_name])
        elif processor_name=='height_width':
            success = self.run_height_width(self.cfg_param[processor_name])
        elif processor_name=='flow_state':
            success = self.run_flow_state(self.cfg_param[processor_name])
        elif processor_name=='height_width_array':
            success = self.run_height_width_array(self.cfg_param[processor_name])
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

    def run_dark_stats(self, cfg):
        if (self.products['dark_stats'] is not None) and (not self.force):
            LOGGER.info(WARN_STR)
        else:
            dark_stats = rivscale.estimate.process_dark_stats(
                cfg, self.products['stretch_stack'])
            # write output
            if dark_stats is not None:
                dark_stats.to_ncfile(self.outfiles['dark_stats'])
            # update products
            self.products['dark_stats'] = dark_stats
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

    def run_along_stats(self, cfg):
        if (self.products['width_stats'] is not None) and (not self.force):
            LOGGER.info(WARN_STR)
        else:
            wse_stats, width_stats = rivscale.estimate.process_along_stats(
                cfg, self.products['stretch_stack'])
            # write outputs
            if wse_stats is not None:
                wse_stats.to_ncfile(self.outfiles['wse_stats'])
            if width_stats is not None:
                width_stats.to_ncfile(self.outfiles['width_stats'])
            #
            self.products['wse_stats'] = wse_stats
            self.products['width_stats'] = width_stats
            #
            if (wse_stats is None):
                print(" wse_stats not generated, skipping rest of processing")
                # TODO: should we check width too? but only of not using Pekel?
                #continue
                return False
        return True

    def run_stretch_average(self, cfg):
        if (self.products['width_avg'] is not None) and (not self. force):
            LOGGER.info(WARN_STR)
        else:
            wse_avg, width_avg = rivscale.estimate.process_stretch_average(
                cfg,
                self.products['stretch_stack'],
                self.products['wse_stats'],
                self.products['width_stats'])
            # write outputs
            if wse_avg is not None:
                wse_avg.to_ncfile(self.outfiles['wse_avg'])
            if width_avg is not None:
                width_avg.to_ncfile(self.outfiles['width_avg'])
            #
            self.products['wse_avg'] = wse_avg
            self.products['width_avg'] = width_avg
        return True

    def run_height_width(self, cfg):
        if (self.products['height_width'] is not None) and (not self.force):
            LOGGER.info(WARN_STR)
        else:
            height_width = rivscale.products.height_width.HeightWidthModel.from_objects(
                cfg,
                self.products['wse_avg'],
                self.products['width_avg'],
                self.products['width_stats'])# TODO: handle Pekel
            if height_width is not None:
                height_width.to_ncfile(self.outfiles['height_width'])
            self.products['height_width'] = height_width
        return True

    def run_flow_state(self, cfg):
        if (self.products['flow_state'] is not None) and (not self.force):
            LOGGER.info(WARN_STR)
        else:
            flow_state = rivscale.estimate.process_flow_state(
                cfg,
                self.products['stretch_stack'],
                self.products['wse_stats'],
                self.products['wse_avg'],
                )
            if flow_state is not None:
                flow_state.to_ncfile(self.outfiles['flow_state'])
            self.products['flow_state'] = flow_state
        return True

    def run_height_width_array(self, cfg):
        if (self.products['height_width_array'] is not None) and (not self.force):
            LOGGER.info(WARN_STR)
        else:
            hw_array = rivscale.estimate.process_height_width_array(
                cfg,
                self.products['stretch_stack'],
                self.products['wse_stats'],
                self.products['flow_state'],
                )
            if hw_array is not None:
                hw_array.to_ncfile(self.outfiles['height_width_array'])
            self.products['height_width_array'] = hw_array
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

