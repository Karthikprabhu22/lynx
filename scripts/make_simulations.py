#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import sys

import click
from IPython.core import ultratb

import h5py

import lynx
from hoover.tools import WhiteNoise
import pysm
import pysm.units as u
import jax.numpy as np
import numpy as old_np
import yaml

# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

_logger = logging.getLogger(__name__)


@click.command()
@click.option('-c', '--config', 'cfg_path', required=True,
              type=click.Path(exists=True), help='path to config file')
@click.option('-o', '--output_path', 'output_path', required=True,
              type=click.Path(exists=False), help='path to config file')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.version_option(lynx.__version__)
def main(cfg_path: Path, output_path: Path, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # YOUR CODE GOES HERE! Keep the main functionality in src/lynx
    # est = lynx.models.Estimator()
    with open(cfg_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    sky = pysm.Sky(nside=cfg['nside'], preset_strings=cfg['skymodel'])
    freqs = old_np.array(cfg['frequencies']) * u.GHz
    sens = np.array(cfg['sensitivities']) * u.uK_CMB
    sens = np.array([w.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(f)) for w, f in zip(sens, freqs)])
    out = np.concatenate(list(sky.get_emission(f)[None, 1:, ...] for f in freqs))
    noise = WhiteNoise(sens=sens)
    noise_map = old_np.asarray(noise.map(cfg['nside']))
    noisy_maps = out + noise_map
    cov = noise.get_pix_var_map(cfg['nside'])

    with h5py.File(output_path, 'a') as f:
        simulation = f.require_group(cfg['name'])
        data_dset = simulation.require_dataset('data', shape=noisy_maps.shape, dtype=noisy_maps.dtype)
        data_dset[...] = noisy_maps
        cov_dset = simulation.require_dataset('cov', shape=cov.shape, dtype=cov.dtype)
        cov_dset[...] = cov
        simulation.attrs.update(cfg)

if __name__ == '__main__':
    main()