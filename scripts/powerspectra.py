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

import jax
import jax.numpy as np
import numpy as old_np

from lynx import Masking

import pdb

# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

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
    
    # MASKING
    # -------
    # Read in masks from HDF5 record. These will be contained in
    # a group called /powerspectrum/apodized
    masking = Masking(mask_path)
    for mask_name, wsp, mask, binning, beam in masking.get_powerspectrum_tools():
        # get the bandpower window function, which will be saved with the
        # computed spectra for comparison with theory later.
        bpw_window_function = wsp.get_bandpower_windows()
        # define log probability from data and model definition
        lnPs = hoover.LogProb.load_data_from_hdf5_batch(data_path)
        # prepare results list. The results have to be saved outside of the
        # for loop, as the lnPs generator hogs to hdf5 resource.
        for i, lnP in enumerate(lnPs):
            model_identifier = lnP.load_model_from_yaml(model_path)
            opt_record = Path('fitting/optimization') / model_identifier / 'mc{:04d}'.format(i)
            record = Path('spectra') / model_identifier / mask_name / 'nmc{:04d}'.format(i)

            with h5py.File(data_path, 'a') as f:
                grp = f.require_group(str(record.parent))
                grp.attrs.update({'config': yaml.dump(masking.cfg)})
                # save bandpower window function
                dset = grp.require_dataset('bpw_window_function', shape=bpw_window_function.shape, dtype=bpw_window_function.dtype)
                dset[...] = bpw_window_function
                dset = grp.require_dataset('beam', shape=beam.shape, dtype=beam.dtype)
                dset[...] = beam

                # cycle through separated components and comput powerspectra
                # of each on the given mask
                for component in lnP._components:
                    T_bar = f[str(opt_record / component)][...]
                    cl = compute_nmt_spectra(T_bar, T_bar, mask, wsp)
                    cl_dset = f.require_dataset(str(record / component), dtype=cl.dtype, shape=cl.shape)             
                    cl_dset[...] = cl

def compute_nmt_spectra(qu1, qu2, mask, wsp):
    f1 = nmt.NmtField(mask, qu1)
    f2 = nmt.NmtField(mask, qu2) 
    cl_coupled = nmt.compute_coupled_cell(f1, f2)
    cl_decoupled = np.array(wsp.decouple_cell(cl_coupled))
    return cl_decoupled

if __name__ == '__main__':
    main()