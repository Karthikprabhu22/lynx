#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import sys
import yaml

import click
from IPython.core import ultratb

import h5py
import yaml

import lynx
import hoover
import pymaster as nmt

from scipy.optimize import minimize
import emcee
import healpy as hp 
import matplotlib.pyplot as plt

import numpy as np

from lynx import Masking

_logger = logging.getLogger(__name__)


@click.command()
@click.option('-d', '--data_path', 'data_path', required=True,
                type=click.Path(exists=True), help='path to data configuration')
@click.option('-m', '--model_path', 'model_path', required=True,
                type=click.Path(exists=False), help='path to model configuration')
@click.option('-p', '--mask_path', 'mask_path', required=True,
                type=click.Path(exists=True), help='path to power spectrum configuration')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.version_option(lynx.__version__)
def main(data_path: Path, model_path: Path, mask_path: Path, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    masking = lynx.Masking(mask_path)
    fitting_masks = list(masking.get_fitting_indices())

    model_identifier, lnP = hoover.LogProb.load_model_from_yaml(model_path)

    with h5py.File(data_path, 'r') as f:
        maps = f['maps']
        nmc = maps.attrs['monte_carlo']

    for mask_name, wsp, mask, binning, beam in masking.get_powerspectrum_tools():
        # get the bandpower window function, which will be saved with the
        # computed spectra for comparison with theory later.
        bpw_window_function = wsp.get_bandpower_windows()

        logging.info(r"""
        Working on power spectra for mask: {:s}
        """.format(mask_name))

        for fitting_name, _ in fitting_masks:
            
            logging.info(r"""
            Working on fitting scheme: {:s}
            """.format(fitting_name)) 

            cl_mc = np.zeros((int(nmc / 2), 4, binning.get_n_bands()))

            for imc in np.arange(nmc)[::2]:
                
                jmc = imc + 1
                
                hdf5_record_1 = str(Path(model_identifier) / fitting_name / 'mc{:04d}'.format(imc))
                hdf5_record_2 = str(Path(model_identifier) / fitting_name / 'mc{:04d}'.format(jmc))

                logging.info(r"""
                Working on Monte Carlo realizations: {:d}, {:d}
                """.format(imc, jmc))

                logging.info(r"""
                    Reading from: {:s}
                    And from records: {:s}, {:s}
                """.format(data_path, hdf5_record_1, hdf5_record_2))
                
                with h5py.File(data_path, 'a') as f:
                    # Create a group which contains results for this sky
                    # patch, model, and MC realization.
                    opt = f[hdf5_record_1]
                    spec = opt.require_group('powerspectra/{:s}'.format(mask_name))
                    spec.attrs.update({'config': yaml.dump(masking.cfg)})
                    # save bandpower window function
                    dset = spec.require_dataset('bpw_window_function', shape=bpw_window_function.shape, dtype=bpw_window_function.dtype)
                    dset[...] = bpw_window_function
                    dset = spec.require_dataset('beam', shape=beam.shape, dtype=beam.dtype)
                    dset[...] = beam
                    # Create a dataset for the whole sky, and log the
                    # results for this patch in the corresponding indices.
                    # Do the same for the spectral parameters.
                    for component in lnP._components:
                        T_bar_1 = f[hdf5_record_1][component][...]
                        T_bar_2 = f[hdf5_record_2][component][...]
                        cl = compute_nmt_spectra(T_bar_1, T_bar_2, mask, wsp)
                        cl_dset = spec.require_dataset(component, dtype=cl.dtype, shape=cl.shape)             
                        cl_dset[...] = cl
        


def compute_nmt_spectra(qu1, qu2, mask, wsp):
    f1 = nmt.NmtField(mask, qu1, purify_b=True)
    f2 = nmt.NmtField(mask, qu2, purify_b=True) 
    cl_coupled = nmt.compute_coupled_cell(f1, f2)
    cl_decoupled = np.array(wsp.decouple_cell(cl_coupled))
    return cl_decoupled

if __name__ == '__main__':
    main()