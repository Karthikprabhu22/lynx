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
                type=click.Path(exists=False), help='path to masking configuration')
@click.option('-c', '--cosmo_path', 'cosmo_path', required=True,
                type=click.Path(exists=False), help='path to cosmo configuration')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.version_option(lynx.__version__)
def main(data_path: Path, mask_path: Path, model_path: Path, cosmo_path: Path, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # MASKING
    # -------
    # Read in masks from HDF5 record. These will be contained in
    # a group called /powerspectrum/apodized
    masking = Masking(mask_path)
    for mask_name, wsp, mask, binning in masking.get_powerspectrum_tools():
        # define log probability from data and model definition
        lnPs = hoover.LogProb.load_data_from_hdf5_batch(data_path)
        # prepare results list. The results have to be saved outside of the
        # for loop, as the lnPs generator hogs to hdf5 resource.
        cl = []
        for i, lnP in enumerate(lnPs):
            model_identifier = lnP.load_model_from_yaml(model_path)
            mask_record = Path('spectra') / model_identifier / mask_name
            record = mask_record / 'nmc{:04d}/cmb'.format(i)

            with h5py.File(data_path, 'r') as f:
                mc_cl = f[str(record)][...]
                cl.append(mc_cl)
        # take just the BB spectrum (index 3)
        cl = np.array(cl)[:, 3, :]
        
        print(cl.shape)
        cl_mean = np.mean(cl, axis=0)
        cl_covar = np.dot(cl.T, cl) / float(len(cl)) - cl_mean ** 2
        cl_covar = np.dot((cl - cl_mean).T, (cl - cl_mean))
        print(cl_mean.shape)
        print(cl_covar.shape)

        ells = binning.get_effective_ells()

        fig, ax = plt.subplots(1, 1)
        ax.errorbar(ells, cl_mean, yerr=np.sqrt(np.diag(cl_covar)))
        plt.show()

        bpw_windows = wsp.get_bandpower_windows()
        bb_bpw_windows = bpw_windows[3, :, 3, :]

        print(cosmo_path)
        lnP = lynx.BBLogLike(data=(cl_mean, cl_covar), bpw_window_function=bb_bpw_windows, model_config_path=cosmo_path)
        res = minimize(lnP, (0.07, 1.1), args=(True), method='Nelder-Mead')
        print(res)


if __name__ == '__main__':
    main()