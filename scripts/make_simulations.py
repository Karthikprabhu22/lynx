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
@click.option('-o', '--output_dir', 'output_dir', required=True,
              type=click.Path(exists=False), help='path to config file')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.version_option(lynx.__version__)
def main(cfg_path: Path, output_dir: Path, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    with open(cfg_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    freqs = old_np.array(cfg['frequencies']) * u.GHz
    nside = cfg['nside']
    components = cfg['skymodel']
    sensitivities = cfg['sensitivities']
    nmc = cfg['monte_carlo']
    outpath = (Path(output_dir).absolute() / cfg['name']).with_suffix('.h5')

 
    logging.info(r"""
    Simulation
    ----------
    frequencies: {:s}
    nside: {:d}
    components: {:s}
    sensitivities: {:s}
    nmc: {:d}
    """.format(", ".join([str(f) for f in freqs]), nside, ", ".join(components), ", ".join([str(s) for s in sensitivities]), nmc))
    # Generate sky signal
    sky = pysm.Sky(nside=nside, preset_strings=components)
    signal = np.concatenate(list(sky.get_emission(f)[None, 1:, ...] for f in freqs))

    # Make noise generator
    sens = np.array(sensitivities) * u.uK_CMB
    sens = np.array([w.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(f)) for w, f in zip(sens, freqs)])
    noise_generator = WhiteNoise(sens=sens)
    
    noise = old_np.asarray(noise_generator.map(nside))
    data = signal + noise
    cov = noise_generator.get_pix_var_map(nside)

    logging.info(r"""
    Output path: {:s}
    """.format(str(outpath)))

    with h5py.File(outpath, 'a') as f:
        f.attrs.update({
            'config': yaml.dump(cfg)
        })
        maps = f.require_group('maps')
        cov_dset = maps.require_dataset('cov', shape=cov.shape, dtype=cov.dtype)
        cov_dset[...] = cov
        maps.attrs.update(cfg)
        for i in range(nmc):
            data_dset = maps.require_dataset('data_mc{:04d}'.format(i), shape=data.shape, dtype=data.dtype)
            data_dset[...] = signal + noise_generator.map(nside, seed=i)

if __name__ == '__main__':
    main()