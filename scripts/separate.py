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
import healpy as hp
from schwimmbad import MultiPool
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

    with open(data_path) as f:
        data_cfg = yaml.load(f, Loader=yaml.FullLoader)
    nside = data_cfg['nside']
    npix = hp.nside2npix(nside)
    nmc = data_cfg['monte_carlo']
    amplitude_output_shape = (nmc, 2, npix)
    parameter_output_shape = (nmc, npix)
    frequencies = np.array(data_cfg['frequencies'])

    with h5py.File(data_cfg['hdf5_path'], 'r') as f:
        data = f['maps/monte_carlo/data'][...]
        cov = f['maps/monte_carlo/cov'][...]

    masking = lynx.Masking(mask_path)
    fitting_masks = list(masking.get_fitting_indices())
    
    tasks = get_tasks(data, cov, nmc, frequencies, fitting_masks, model_path)

    with MultiPool() as pool:
        results = pool.map(do_fitting, tasks)

    with h5py.File(data_cfg['hdf5_path'], 'a') as f:
        for result in results:
            save_data(f, amplitude_output_shape, parameter_output_shape, result)

def save_data(f, amplitude_output_shape, parameter_output_shape, arg):
    (patch_num, imc, indices, fitting_name, model_identifier, lnP, res, outputs) = arg
    logging.info(f"Saving MC: {imc:04d}")

    # Create a group which contains results for this sky
    # patch, model, and MC realization.
    opt = f.require_group(f"{model_identifier}/{fitting_name}")

    # Log the indices of the patch, and the optimization
    # results for this patch.
    idx = opt.require_dataset(f"patch_{patch_num:d}_indices", shape=indices.shape, dtype=indices.dtype)
    idx[...] = indices
    result = opt.require_dataset(f"patch_{patch_num:d}_ML_theta", shape=res.x.shape, dtype=res.x.dtype)
    result[...] = res.x

    # Create a dataset for the whole sky, and log the
    # results for this patch in the corresponding indices.
    # Do the same for the spectral parameters.
    for comp, output in outputs.items():
        for key, value in output.items(): 
            dset = opt.require_dataset(f"{comp}_{key}", shape=amplitude_output_shape, dtype=value.dtype)
            dset[imc, ..., indices] = value

    for i, par in enumerate(lnP.free_parameters):
        par_dset = opt.require_dataset(par, shape=parameter_output_shape, dtype=res.x[i].dtype)
        par_dset[imc, indices] = res.x[i]
    

def do_fitting(args):
    """ Function to carry out the fitting for a subset of the data.
    """
    (patch_num, imc, indices, fitting_name, data, covariance, frequencies, model_path) = args
    logging.info(f"Fitting MC: {imc:d}")
    model_identifier, lnP = hoover.LogProb.load_model_from_yaml(model_path)
    lnP.data_setup(data=data, covariance=covariance, frequencies=frequencies)
    # Run the optimization.
    # Note, if using Nelder-Mead, can not write dictionary to
    # HDF5 file attribute, as simplex is too complex an object.
    theta_0 = lnP.theta_0()
    res = minimize(lnP, theta_0, args=(True), method="Powell")
    outputs = {}
    for comp in lnP._components:
        outputs.update(
            {f"{comp}": {"T_bar": lnP.get_amplitude_expectation(res.x, component=comp),
                    "N_T": lnP.get_amplitdue_covariance(res.x, component=comp),}})
                    
    return patch_num, imc, indices, fitting_name, model_identifier, lnP, res, outputs


def get_tasks(data, covariance, nmc, frequencies, fitting_masks, model_path):
    for imc in range(nmc):
        for fitting_name, fitting_parameters in fitting_masks:
            logging.info(f"Working on fitting scheme: {fitting_name}") 
            for patch_num, indices in fitting_parameters:
                logging.info(f"Working on patch number: {patch_num:d}")
                yield (patch_num, imc, indices, fitting_name, data[imc][..., indices], covariance[imc][..., indices], frequencies, model_path)


if __name__ == '__main__':
    main()