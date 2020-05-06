#!/usr/bin/env python
# -*- coding: utf-8 -*-

#######################################################################
#
# This script is used to generate simulations for B-mode forecasting.
# 
# 
#
#
#
#
#
#######################################################################

import logging
from pathlib import Path
import sys

import click
import h5py

import lynx
from hoover.tools import WhiteNoise
import pysm
import pysm.units as u
import jax.numpy as np
import numpy as old_np
import yaml
import healpy as hp 

_logger = logging.getLogger(__name__)


@click.command()
@click.option('-d', '--config', 'cfg_path', required=True,
              type=click.Path(exists=True), help='path to config file')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.version_option(lynx.__version__)
def main(cfg_path: Path, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    freqs = old_np.array(cfg['frequencies']) * u. GHz
    nside = cfg['nside']
    components = cfg['skymodel']['args']
    sensitivities = cfg['sensitivities']
    nmc = cfg['monte_carlo']
    beams = cfg['fwhm']
    outpath = cfg['hdf5_path']
    half_mission_noise = cfg['half_mission_noise']
    cosmo_path = cfg['cosmo_path']
    
    if half_mission_noise:
        sensitivities = [s * np.sqrt(2.) for s in sensitivities]
    
    logging.info(f"""
    Frequencies: {freqs!s}
    Nside: {nside:04d}
    Components: {components!s}
    Sensitivities: {sensitivities!s}
    Number of Monte Carlo Simulations: {nmc:05d}
    """)

    # Generate sky signal
    sky = pysm.Sky(nside=nside, **components)
    fgnd = (sky.get_emission(f) for f in freqs)
    fgnd = (hp.smoothing(s, fwhm=b / 60. * np.pi / 180., verbose=False)[None, 1:, ...] for b, s in zip(beams, fgnd))
    fgnd = np.concatenate(list(fgnd))

    # Make noise generator
    sens = np.array(sensitivities) * u.uK_CMB
    sens = np.array([w.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(f)) for w, f in zip(sens, freqs)])
    noise_generator = WhiteNoise(sens=sens)

    cov = noise_generator.get_pix_var_map(nside)

    logging.info(f"Output path: {outpath}")

    with h5py.File(outpath, 'a') as f:
        f.attrs.update({'config': yaml.dump(cfg)})
        maps = f.require_group('maps')
        monte_carlo = maps.require_group('monte_carlo')
        components = maps.require_group('components')

        data_dset = monte_carlo.require_dataset('data', shape=(nmc, len(freqs), 2, hp.nside2npix(nside)), dtype=np.float32)

        cov_dset = monte_carlo.require_dataset('cov', shape=(nmc, len(freqs), 2, hp.nside2npix(nside)), dtype=np.float32)
        cov_dset[...] = cov.astype(np.float32)
        
        for imc in np.arange(nmc)[::2]:

            logging.info(f"Working on CMB MC: {imc:04d}")

            cmb = get_cmb_realization(nside, cosmo_path, beams, freqs, seed=imc)

            for j in range(imc, imc + 2):

                logging.info(f"Working on noise MC: {j:04d}")

                data = fgnd + cmb + noise_generator.map(nside, seed=j)

                logging.debug(f"Data shape: {data.shape!r}")

                data_dset[j] = data 

def get_cmb_realization(nside, cl_path, beams, frequencies, seed=100):
    with h5py.File(f"{cl_path}", 'r') as f:
        cl_total = np.swapaxes(f['lensed_scalar'][...], 0, 1)
    cmb = hp.synfast(cl_total, nside, new=True, verbose=False)
    cmb = [hp.smoothing(cmb, fwhm=b / 60. * np.pi/180., verbose=False)[1:] * u.uK_CMB for b in beams]
    return np.array([c.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(f)) for c, f in zip(cmb, frequencies)])

if __name__ == '__main__':
    main()