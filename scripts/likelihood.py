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

import numpy as np
from scipy import stats

from lynx import Masking

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
    
    masking = lynx.Masking(mask_path)
    
    fitting_masks = list(masking.get_fitting_indices())

    model_identifier, lnP = hoover.LogProb.load_model_from_yaml(model_path)

    with h5py.File(data_path, 'r') as f:
        maps = f['maps']
        nmc = maps.attrs['monte_carlo']

    for mask_name, wsp, mask, binning, beam in masking.get_powerspectrum_tools():

        for fitting_name, _ in fitting_masks:
            
            cl_mc = np.zeros((int(nmc / 2), 4, binning.get_n_bands()))
            
            for imc in np.arange(int(nmc / 2)):    
                
                hdf5_record = str(Path(model_identifier) / fitting_name / 'mc{:04d}'.format(imc * 2) / 'powerspectra' / mask_name / 'cmb')

                with h5py.File(data_path, 'r') as f:
                    cl_mc[imc] = f[hdf5_record][...]

            ee_mean, ee_cov = compute_mean_cov(cl_mc[:, 0])
            bb_mean, bb_cov = compute_mean_cov(cl_mc[:, 3])

            bpw_windows = wsp.get_bandpower_windows()
            ee_bpw_windows = bpw_windows[0, :, 0, :]
            bb_bpw_windows = bpw_windows[3, :, 3, :]
            bpws = binning.get_effective_ells()

            with h5py.File("data/planck_2018_acc.h5", 'r') as f:
                ee = f['lensed_scalar'][:ee_bpw_windows.shape[-1], 1]
                bb = f['lensed_scalar'][:bb_bpw_windows.shape[-1], 2]

            fig, ax = plt.subplots(1, 1)
            ax.set_title("BB bandpower covariance matrix")
            ax.imshow(bb_cov, origin='lower', extent=[bpws[0], bpws[-1], bpws[0], bpws[-1]])
            ax.set_xlabel(r"$\ell_b$")
            ax.set_ylabel(r"$\ell_b$")
            fig.savefig("reports/figures/bb_cov.pdf", bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            ax.plot(bpws, np.dot(ee_bpw_windows, ee), 'k-', label='Binned theory input')
            ax.errorbar(bpws, ee_mean, np.sqrt(np.diag(ee_cov)), color='k', fmt='o', label='Cleaned estimate')
            ax.set_yscale('log')
            ax.set_xlabel(r"$\ell_b$")
            ax.set_ylabel(r"$C_{\ell_b}^{\rm EE}~[{\rm \mu K}^2]$")
            ax.legend()
            fig.savefig("reports/figures/ee_recovered.pdf", bbox_inches='tight')

            lnP = lynx.BBLogLike(data=(bb_mean, bb_cov), bpw_window_function=bb_bpw_windows, model_config_path=cosmo_path)
            res = minimize(lnP, lnP.theta_0(), args=(True), method='Nelder-Mead')

            fig, ax = plt.subplots(1, 1)
            samples = np.random.multivariate_normal(res.x, lnP.covariance(res.x), 100)
            for sample in samples:
                ax.plot(bpws, lnP.model(sample), 'C0-', alpha=0.05)
            ax.plot(bpws, np.dot(bb_bpw_windows, bb), 'k-', label='Binned theory input')
            ax.errorbar(bpws, bb_mean, np.sqrt(np.diag(bb_cov)), color='k', fmt='o', label='Cleaned estimate')
            ax.set_yscale('log')
            ax.set_xlabel(r"$\ell_b$")
            ax.set_ylabel(r"$C_{\ell_b}^{\rm BB}~[{\rm \mu K}^2]$")
            ax.legend()
            fig.savefig("reports/figures/bb_recovered.pdf", bbox_inches='tight')

            print(res.x)
            print(np.sqrt(np.diag(lnP.covariance(res.x))))
            print('Reduced chi2: ', lnP.chi2(res.x))

def compute_mean_cov(arr):
    assert arr.ndim == 2
    nmc = float(arr.shape[0])
    mean = np.mean(arr, axis=0)
    diff = arr - mean[None, :]
    cov = diff[:, None, :] * diff[:, :, None]
    cov = np.sum(cov, axis=0) / nmc
    return mean, cov

if __name__ == '__main__':
    main()