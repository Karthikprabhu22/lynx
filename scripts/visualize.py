#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import sys

import click

import h5py
import yaml

import lynx
import hoover
import pymaster as nmt

from scipy.optimize import minimize
import emcee
import healpy as hp 
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import pandas as pd
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
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.version_option(lynx.__version__)
def main(data_path: Path, mask_path: Path, model_path: Path, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    masking = lynx.Masking(mask_path)
    fitting_masks = list(masking.get_fitting_indices())

    model_identifier, lnP = hoover.LogProb.load_model_from_yaml(model_path)

    with h5py.File(data_path, 'r') as f:
        sky_config = yaml.load(f.attrs['config'], Loader=yaml.FullLoader)
        nmc = sky_config['monte_carlo']
        frequencies = np.array(sky_config['frequencies'])
        nside = sky_config['nside']

    amplitude_output_shape = (2, hp.nside2npix(nside))
    parameter_output_shape = (hp.nside2npix(nside),)
        
    for fitting_name, fitting_parameters in fitting_masks:
        T_bar = {comp: np.zeros(amplitude_output_shape) for comp in lnP._components}
        par = {par: np.zeros(parameter_output_shape) for par in lnP.free_parameters}
        for imc in range(nmc):
            logging.info(r"""
            Working on Monte Carlo realization: {:d}
            """.format(imc))
            hdf5_record = str(Path(model_identifier) / fitting_name / 'mc{:04d}'.format(imc))

            logging.info(r"""
            Working on fitting scheme: {:s}
            """.format(fitting_name)) 
            
            with h5py.File(data_path, 'a') as f:
                # Create a group which contains results for this sky
                # patch, model, and MC realization.
                opt = f[hdf5_record]
                # Create a dataset for the whole sky, and log the
                # results for this patch in the corresponding indices.
                # Do the same for the spectral parameters.
                for component in lnP._components:
                    T_bar[component] += opt[component][...]

                for parameter in lnP.free_parameters:
                    par[parameter] += opt[parameter][...]

        T_bar = {key: value / float(nmc) for key, value in T_bar.items()}
        par = {key: value / float(nmc) for key, value in par.items()}

        hp.mollview(T_bar['cmb'][0])
        hp.mollview(par['beta_d'])
        plt.show()

def plot_fisher(mean, cov, truth=None, fpath=None, xlabel=None, ylabel=None):
    nb = 128
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(4, 4, hspace=0., wspace=0.)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_ax = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_ax = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

    # scatter points on the main axes
    plot_fisher_2d(mean, cov, main_ax)
    if truth is not None:
        main_ax.axvline(truth[0], color='gray', linestyle='--')
        main_ax.axhline(truth[1], color='gray', linestyle='--')

    # histogram on the attached axes
    sigma = np.sqrt(cov[0, 0])
    x_arr = mean[0] - 4 * sigma + 8 * sigma * np.arange(nb) / (nb - 1.)
    p_arr = np.exp(- (x_arr - mean[0]) ** 2 / (2 * sigma ** 2))
    x_ax.plot(x_arr, p_arr, color='tan', label='Validation')
    if truth is not None:
        x_ax.axvline(truth[0], color='gray', linestyle='--', label=True)
    x_ax.invert_yaxis()

    l, h = x_ax.get_legend_handles_labels(legend_handler_map=None)
    label = fig.add_subplot(grid[0, 0], visible=False)
    main_ax.legend(l, h, frameon=False, loc='upper right', bbox_to_anchor=(1., 1.))

    sigma = np.sqrt(cov[1, 1])
    y_arr = mean[1] - 4 * sigma + 8 * sigma * np.arange(nb) / (nb - 1.)
    p_arr = np.exp(- (y_arr - mean[1]) ** 2 / (2 * sigma ** 2))
    y_ax.plot(p_arr, y_arr, color='tan')
    if truth is not None:
        y_ax.axhline(truth[1], color='gray', linestyle='--')
    y_ax.invert_xaxis()

    x_ax.set_xlabel(xlabel)
    y_ax.set_ylabel(ylabel)

    x_ax.get_yaxis().set_visible(False)
    y_ax.get_xaxis().set_visible(False)

    main_ax.get_xaxis().set_visible(False)
    main_ax.get_yaxis().set_visible(False)

    if fpath is not None:
        fig.savefig(fpath, bbox_inches='tight')
    return


def plot_fisher_1d(mean, cov, i, ax, labels=None):
    fontsize = 16
    nb = 128
    sigma = np.sqrt(cov[i, i])

    x_arr = mean[i] - 4 * sigma + 8 * sigma * np.arange(nb) / (nb - 1.)
    p_arr = np.exp(- (x_arr - mean[i]) ** 2 / (2 * sigma ** 2))
    ax.plot(x_arr, p_arr)

    ax.set_xlim([mean[i] - 3. * sigma, mean[i] + 3. * sigma])

    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize - 2)
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize - 2)

    return

def plot_fisher_2d(mean, cov, ax, labels=None):
    fontsize = 16
    w, v = np.linalg.eigh(cov)
    angle = 180. * np.arctan2(v[1, 0], v[0, 0]) / np.pi
    
    a_1s=np.sqrt(2.3 * w[0])
    b_1s=np.sqrt(2.3 * w[1])
    a_2s=np.sqrt(6.17 * w[0])
    b_2s=np.sqrt(6.17 * w[1])

    sigma_00 = np.sqrt(cov[0, 0])
    sigma_11 = np.sqrt(cov[1, 1])

    e_1s = Ellipse(xy=mean, width=2 * a_1s, height=2 * b_1s, angle=angle, color='tan')
    e_2s = Ellipse(xy=mean, width=2 * a_2s, height=2 * b_2s, angle=angle, alpha=0.5, color='tan')

    ax.add_artist(e_1s)
    ax.add_artist(e_2s)
    ax.set_xlim([mean[0] - 3 * sigma_00, mean[0] + 3 * sigma_00])
    ax.set_ylim([mean[1] - 3 * sigma_11, mean[1] + 3 * sigma_11])

    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize - 2)
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize - 2)

    return 

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