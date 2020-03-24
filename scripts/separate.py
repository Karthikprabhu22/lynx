#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import sys

import click
from IPython.core import ultratb

import h5py
from scipy.optimize import minimize

import lynx
import jax
import emcee
import hoover
from hoover.tools import WhiteNoise
import pysm
import pysm.units as u
import jax.numpy as np
import numpy as old_np
import yaml

import pdb

# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

_logger = logging.getLogger(__name__)


@click.command()
@click.option('-d', '--data_path', 'data_path', required=True,
                type=click.Path(exists=True), help='path to data configuration')
@click.option('-m', '--model_path', 'model_path', required=True,
                type=click.Path(exists=False), help='path to model configuration')
@click.option('-r', '--record', 'record', required=True,
                type=str, help='name of record in hdf5 file')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.version_option(lynx.__version__)
def main(data_path: Path, model_path: Path, record:str, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # define log probability from data and model definition
    lnP = hoover.LogProb()
    lnP.load_data_from_hdf5(data_path, record)
    lnP.load_model_from_yaml(model_path)

    # draw a starting point for the optimization from prior
    theta_0 = lnP.theta_0()

    logging.info(r"""
    Running optimization with:
    theta_0: {:s} 
    """.format(" , ".join([str(r) for r in theta_0])))

    # Run the optimization.
    # Note, if using Nelder-Mead, can not write dictionary to
    # HDF5 file attribute, as simplex is too complex an object.
    res = minimize(lnP, theta_0, args=(True), method="Powell")

    logging.info(r"""
    Completed optimization
    ----------------------
    success: {:b}
    result: {:s}
    """.format(res.success, " , ".join([str(r) for r in res.x])))

    logging.info(r"""
    Writing result to: {:s} 
    in record: {:s}
    """.format(data_path, record))

    # save optimization results in new subgroup of the input
    # data record.
    with h5py.File(data_path, 'a') as f:
        grp = f[record]
        opt = grp.require_group('fitting/optimization')
        opt.attrs.clear()
        opt.attrs.update(res)

def run_sampling(log_prob, p0, ndim, nwalkers, checkpoint=None, seed=748):    
    # set an array of starting positions for the different walkers
    key = jax.random.PRNGKey(seed)
    p0 = p0[None, :] + jax.random.normal(key, shape=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, 100)
    return sampler

if __name__ == '__main__':
    main()