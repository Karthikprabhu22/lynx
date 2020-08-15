#!/usr/bin/env python
# -*- coding: utf-8 -*-

#######################################################################
#
# This script is used to consolidate CMBS4 simulations for B-mode forecasting using lynx.
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
import os 

import click
import h5py
from astropy.io import fits
from astropy.wcs import WCS

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
    sensitivities = cfg['sensitivities']
    nmc = cfg['monte_carlo']
    beams = cfg['fwhm']
    outpath = cfg['hdf5_path']
    half_mission_noise = cfg['half_mission_noise']
    mapspath = cfg['maps_path']
    maskpath = cfg['mask_path']
    
    realizations = ["0000","0001","0002","0003","0004"]
    freq = ["f020_b11","f030_b77","f040_b58","f085_b27","f095_b24","f145_b16","f155_b15","f220_b11","f270_b08"]
    if half_mission_noise:
        sensitivities = [s * np.sqrt(2.) for s in sensitivities]
    
    logging.info(f"""
    Frequencies: {freqs!s}
    Nside: {nside:04d}
    Sensitivities: {sensitivities!s}
    Number of Monte Carlo Simulations: {nmc:05d}
    """)
    
    cov = old_np.nan_to_num(hp.read_map(maskpath+"n0512.fits", verbose=False))
    
    logging.info(f"Output path: {outpath}")
    
    with h5py.File(outpath, 'a') as f:
        f.attrs.update({'config': yaml.dump(cfg)})
        maps = f.require_group('maps')
        monte_carlo = maps.require_group('monte_carlo')

        data_dset = monte_carlo.require_dataset('data', shape=(nmc, len(freqs), 2, hp.nside2npix(nside)), dtype=np.float32)

        cov_dset = monte_carlo.require_dataset('cov', shape=(nmc, len(freqs), 2, hp.nside2npix(nside)), dtype=np.float32)
        for i in range(len(freqs)):
            cov_dset[:,i,:,:] = np.array(sensitivities)[i]/cov.astype(np.float32)
        
        for imc in np.arange(nmc):
            for j in range(len(freq)):
                file = "cmbs4_04p00_comb_AL0p1_"+freq[j]+"_ellmin30_map_0512_mc_"+realizations[imc]+".fits"
                data = old_np.array([old_np.nan_to_num(fits.open(mapspath+file)[1].data['Q-POLARISATION']),old_np.nan_to_num(fits.open(mapspath+file)[1].data['U-POLARISATION'])])
                data_dset[imc,j] = data 

if __name__ == '__main__':
    main()