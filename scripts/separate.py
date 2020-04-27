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

from scipy.optimize import minimize
import emcee
import healpy as hp

import numpy as np

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
        sky_config = yaml.load(f.attrs['config'], Loader=yaml.FullLoader)
        nmc = sky_config['monte_carlo']
        frequencies = np.array(sky_config['frequencies'])
        nside = sky_config['nside']

    amplitude_output_shape = (2, hp.nside2npix(nside))
    parameter_output_shape = (hp.nside2npix(nside),)

    for imc in range(nmc):
        logging.info(r"""
        Working on Monte Carlo realization: {:d}
        """.format(imc))

        with h5py.File(data_path, 'a') as f:
            data = f['maps/monte_carlo/data_mc{:04d}'.format(imc)][...]
            cov = f['maps/monte_carlo/cov'][...]
        
        for fitting_name, fitting_parameters in fitting_masks:

            hdf5_record = str(Path(model_identifier) / fitting_name / 'mc{:04d}'.format(imc))

            logging.info(r"""
            Working on fitting scheme: {:s}
            """.format(fitting_name)) 

            for patch_num, indices in fitting_parameters:
                
                logging.info(r"""
                Working on patch number: {:d}
                """.format(patch_num))

                lnP.data_setup(data=data[:, :, indices], covariance=cov[:, :, indices], frequencies=frequencies)

                # Run the optimization.
                # Note, if using Nelder-Mead, can not write dictionary to
                # HDF5 file attribute, as simplex is too complex an object.
                theta_0 = lnP.theta_0(seed=imc + patch_num * nmc)
                res = minimize(lnP, theta_0, args=(True), method="Powell")
    
                logging.info(r"""
                    Optimization success: {:b}
                    Optimization result: {:s}
                """.format(res.success, " , ".join([str(r) for r in res.x])))

                # save optimization results in new subgroup of the input
                # data record.
                logging.info(r"""
                    Results written to: {:s}
                    Record in file: {:s}
                """.format(data_path, hdf5_record))
                with h5py.File(data_path, 'a') as f:
                    # Create a group which contains results for this sky
                    # patch, model, and MC realization.
                    opt = f.require_group(hdf5_record)

                    # Log the indices of the patch, and the optimization
                    # results for this patch.
                    idx = opt.require_dataset('patch{:03d}_indices'.format(patch_num), shape=indices.shape, dtype=indices.dtype)
                    result = opt.require_dataset('patch{:03d}_ML_theta', shape=res.x.shape, dtype=res.x.dtype)
                    result.attrs.update(res)
                    idx[...] = indices

                    # Create a dataset for the whole sky, and log the
                    # results for this patch in the corresponding indices.
                    # Do the same for the spectral parameters.
                    for component in lnP._components:
                        T_bar = lnP.get_amplitude_expectation(res.x, component=component)
                        T_bar_dset = opt.require_dataset(component, shape=amplitude_output_shape, dtype=T_bar.dtype)
                        T_bar_dset[..., indices] = T_bar

                    for i, par in enumerate(lnP.free_parameters):
                        par_dset = opt.require_dataset(par, shape=parameter_output_shape, dtype=res.x[i].dtype)
                        par_dset[indices] = res.x[i]

if __name__ == '__main__':
    main()