#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import sys

import click
from IPython.core import ultratb

import h5py
import yaml

import lynx
import hoover

from scipy.optimize import minimize
import emcee

import jax
import jax.numpy as np
import numpy as old_np

import pdb

# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

_logger = logging.getLogger(__name__)


@click.command()
@click.option('-d', '--data_path', 'data_path', required=True,
                type=click.Path(exists=True), help='path to data configuration')
@click.option('-m', '--model_path', 'model_path', required=True,
                type=click.Path(exists=False), help='path to model configuration')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.version_option(lynx.__version__)
def main(data_path: Path, model_path: Path, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # define log probability from data and model definition
    lnPs = hoover.LogProb.load_data_from_hdf5_batch(data_path)
    # prepare results list. The results have to be saved outside of the
    # for loop, as the lnPs generator hogs to hdf5 resource.

    for i, lnP in enumerate(lnPs):
        model_identifier = lnP.load_model_from_yaml(model_path)
        opt_record = Path('fitting/optimization') / model_identifier / 'mc{:04d}'.format(i)
        # draw a starting point for the optimization from prior
        theta_0 = lnP.theta_0(seed=i)

        logging.info(r"""
        -----------------------------------------
        Monte Carlo: {:d}
            theta_0: {:s} 
        """.format(i, " , ".join([str(r) for r in theta_0])))

        # Run the optimization.
        # Note, if using Nelder-Mead, can not write dictionary to
        # HDF5 file attribute, as simplex is too complex an object.
        res = minimize(lnP, theta_0, args=(True), method="Powell")

        logging.info(r"""
            success: {:b}
            result: {:s}
            Writing result to: {:s} 
            in the HDF5 record: {:s}
        -----------------------------------------
        """.format(res.success, " , ".join([str(r) for r in res.x]), data_path, opt_record.as_posix()))

        # save optimization results in new subgroup of the input
        # data record.
        with h5py.File(data_path, 'a') as f:
            # create group for each MC iteration
            opt = f.require_group(opt_record.as_posix())
            # save optimization results in this group's attributes
            opt.attrs.clear()
            opt.attrs.update(res)
            # create datasets for amplitude and covariance
            for component in lnP._components:
                T_bar = lnP.get_amplitude_expectation(res.x, component=component)
                T_bar_dset = opt.require_dataset(component, shape=T_bar.shape, dtype=T_bar.dtype)
                T_bar_dset[...] = T_bar
            N_T = lnP.get_amplitdue_covariance(res.x)
            N_T_dset = opt.require_dataset('N_T', shape=N_T.shape, dtype=N_T.dtype)
            N_T_dset[...] = N_T

if __name__ == '__main__':
    main()