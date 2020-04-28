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
        sky_config = yaml.load(f.attrs['config'], Loader=yaml.FullLoader)
        nmc = sky_config['monte_carlo']

    for mask_name, wsp, mask, binning, beam in masking.get_powerspectrum_tools():

        for fitting_name, _ in fitting_masks:

            results_dir = Path("data/results") / "_".join([mask_name, model_identifier, fitting_name])
            results_dir.mkdir(exist_ok=True, parents=True)

            cl_mc = np.zeros((int(nmc / 2), 4, binning.get_n_bands()))
            cln_bb_cov = np.zeros((int(nmc / 2), binning.get_n_bands(), binning.get_n_bands()))
            for imc in np.arange(int(nmc / 2)):    

                hdf5_record = str(Path(model_identifier) / fitting_name / 'mc{:04d}'.format(imc * 2) / 'powerspectra' / mask_name / 'cmb')

                with h5py.File(data_path, 'r') as f:
                    cl_mc[imc] = f[hdf5_record][...]
                    cln_bb_cov[imc] = f[hdf5_record + '_cln_cov'][...]


            cln_bb_cov = np.mean(cln_bb_cov, axis=0)

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
            fig.savefig(results_dir / "bb_cov.pdf", bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            ax.plot(bpws, np.dot(ee_bpw_windows, ee), 'k-', label='Binned theory input')
            ax.errorbar(bpws, ee_mean, np.sqrt(np.diag(ee_cov)), color='k', fmt='o', label='Cleaned estimate')
            ax.set_yscale('log')
            ax.set_xlabel(r"$\ell_b$")
            ax.set_ylabel(r"$C_{\ell_b}^{\rm EE}~[{\rm \mu K}^2]$")
            ax.legend()
            fig.savefig(results_dir / "ee_recovered.pdf", bbox_inches='tight')

            lnP = lynx.BBLogLike(data=(bb_mean, bb_cov), bpw_window_function=bb_bpw_windows, model_config_path=cosmo_path)
            res = minimize(lnP, lnP.theta_0(), args=(True), method='Nelder-Mead')

            fig, ax = plt.subplots(1, 1)
            samples = np.random.multivariate_normal(res.x, lnP.covariance(res.x), 100)
            samples = np.array([lnP.model(sample) for sample in samples])
            samples += np.random.multivariate_normal(np.zeros_like(bpws), cln_bb_cov, 100)
            pp_mean = np.mean(samples, axis=0)
            pp_std = np.std(samples, axis=0)
            ax.errorbar(bpws+1, pp_mean, color='C0', fmt='o', yerr=pp_std)
            for s in samples:
                ax.plot(bpws, s, 'C1', alpha=0.1)
            #for sample in samples:
            #    ax.plot(bpws, lnP.model(sample) + , 'C1-', alpha=0.05)
            ax.plot(bpws, np.dot(bb_bpw_windows, bb), 'k-', label='Binned theory input')
            ax.plot(bpws, np.sqrt(np.diag(cln_bb_cov)), 'g-', label=r'$\sigma(C_\ell^{\rm BB})$')
            ax.errorbar(bpws, bb_mean, np.sqrt(np.diag(bb_cov)), color='k', fmt='o', label='Cleaned estimate')
            ax.set_yscale('log')
            ax.set_ylim(1e-7, 4e-6)
            ax.set_xlabel(r"$\ell_b$")
            ax.set_ylabel(r"$C_{\ell_b}^{\rm BB}~[{\rm \mu K}^2]$")
            ax.legend()
            fig.savefig(results_dir / "bb_recovered.pdf", bbox_inches='tight')

            plot_fisher(res.x, lnP.covariance(res.x), truth=[1., 0.], fpath=results_dir / "ellipses.pdf", xlabel=r"$A_L$", ylabel=r"$r$")

            print(res.x)
            print(np.sqrt(np.diag(lnP.covariance(res.x))))
            print('Reduced chi2: ', lnP.chi2(res.x))

            np.save(results_dir / "parameter_ML.npy", res.x)
            np.save(results_dir / "parameter_covariance.npy", lnP.covariance(res.x))

            results = pd.DataFrame({
                'mean': pd.Series(res.x, index=lnP.free_parameters),
                '1-sigma': pd.Series(np.sqrt(np.diag(lnP.covariance(res.x))), index=lnP.free_parameters)
            })
            print(results)
            results.to_csv(results_dir / "results.csv")

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