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

import healpy as hp 
import matplotlib.pyplot as plt
import numpy as np

_logger = logging.getLogger(__name__)


@click.command()
@click.option('-p', '--mask_path', 'mask_path', required=True,
                type=click.Path(exists=False), help='path to masking configuration')
@click.option('-c', '--cosmo_path', 'cosmo_path', required=True,
                type=click.Path(exists=False), help='path to cosmo configuration')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.version_option(lynx.__version__)
def main(mask_path: Path, cosmo_path: Path, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    nside = 128
    nmc = 30
    mask = make_binary_mask(nside)
    masking = lynx.Masking(mask_path)
    masking.save_fitting_mask('one', mask)
    masking.calculate_apodizations(mask)
    masking.calculate_mode_coupling()
    cl_theory = read_cl(cosmo_path)
    ells = np.arange(0, len(cl_theory[0]))
    for mask_name, wsp, mask, binning, beam in masking.get_powerspectrum_tools():
        print(r"""
        Mask name: {:s}
        Fsky: {:.03f} 
        """.format(mask_name, np.mean(mask)))

        ee_bpw_window = wsp.get_bandpower_windows()[0, :, 0, :]
        bb_bpw_window = wsp.get_bandpower_windows()[3, :, 3, :]

        cl_mc = np.zeros((nmc, 4, binning.get_n_bands()))
        
        for i in range(nmc):
            cmb = get_cmb_realization(nside, cosmo_path, np.pi/180.)
            cl_mc[i] = compute_nmt_spectra(cmb[1:], cmb[1:], mask, wsp)

        cl_mean = np.mean(cl_mc, axis=0)
        cl_covar_ee = (cl_mc[:, 0] - cl_mean[0])[..., None, :] * (cl_mc[:, 0] - cl_mean[0])[..., None]
        cl_covar_ee = np.sum(cl_covar_ee, axis=0) / float(nmc - 1)

        cl_covar_bb = (cl_mc[:, 3] - cl_mean[3])[..., None, :] * (cl_mc[:, 3] - cl_mean[3])[..., None]
        cl_covar_bb = np.sum(cl_covar_bb, axis=0) / float(nmc - 1)
        
        fig, ax = plt.subplots(1, 1)
        ax.errorbar(binning.get_effective_ells(), cl_mean[0], yerr=np.sqrt(np.diag(cl_covar_ee)))
        #ax.plot(binning.get_effective_ells(), cl_mean[0], 'C0-')
        ax.plot(binning.get_effective_ells(), np.dot(ee_bpw_window, cl_theory[1][:ee_bpw_window.shape[-1]]), 'C0--')

        #ax.plot(binning.get_effective_ells(), cl_mean[3], 'C1-')
        ax.plot(binning.get_effective_ells(), np.dot(bb_bpw_window, cl_theory[2][:bb_bpw_window.shape[-1]]), 'C1--')
        ax.errorbar(binning.get_effective_ells(), cl_mean[3], yerr=np.sqrt(np.diag(cl_covar_bb)))

        ax.set_xlim(0, 3*nside)
        ax.set_yscale('log')
        plt.show()

def compute_nmt_spectra(qu1, qu2, mask, wsp):
    f1 = nmt.NmtField(mask, qu1, purify_b=True)
    f2 = nmt.NmtField(mask, qu2, purify_b=True) 
    cl_coupled = nmt.compute_coupled_cell(f1, f2)
    cl_decoupled = np.array(wsp.decouple_cell(cl_coupled))
    return cl_decoupled

def make_binary_mask(nside, lonlat=(0, 80), rad=0.7):
    mask = np.zeros(hp.nside2npix(nside))
    idx = hp.query_disc(nside, hp.ang2vec(*lonlat, lonlat=True), rad)
    mask[idx] = 1
    return mask

def read_cl(cl_path):
    with h5py.File(cl_path, 'r') as f:
        cl_total = np.swapaxes(f['lensed_scalar'][...], 0, 1)
    return cl_total

def get_cmb_realization(nside, cl_path, fwhm):
    cl_total = read_cl(cl_path)
    return  np.array(hp.smoothing(hp.synfast(cl_total, nside, new=True, verbose=False), fwhm=fwhm, verbose=False))

if __name__ == '__main__':
    main()