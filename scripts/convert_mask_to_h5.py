#!/usr/bin/env python
# -*- coding: utf-8 -*-

#######################################################################
#
# This script is used to convert rhits map from fits to h5 format
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
import pysm3
import pysm3.units as u
import jax.numpy as np
import numpy as old_np
import yaml
import healpy as hp 
import pymaster as nmt

from astropy.io import fits
from astropy.wcs import WCS


_logger = logging.getLogger(__name__)


@click.command()
@click.option('-d', '--config', 'cfg_path', required=True,
              type=click.Path(exists=True), help='path to config file')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)

def main(cfg_path: Path, log_level: int):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    nside = cfg['nside']
    outpath = cfg['hdf5_path']
    
    mask_file = 'data/n0512.fits'
    mask_data = old_np.nan_to_num(hp.read_map(mask_file, verbose=False, dtype=np.float32))
    mask_apodized = nmt.mask_apodization(mask_data,5., apotype="C2")
    mask_binary = old_np.where(mask_data>0.0, 1.0, 0.0)



    with h5py.File(outpath, 'w') as f:
        ns0512 = f.require_group('ns0512')        
        data_dset = ns0512.require_dataset('binary', shape=(hp.nside2npix(nside),), dtype=np.float32)
        data_dset[...] = np.nan_to_num(mask_binary)

        apodized = ns0512.require_group('apodized') 
        data_dset2 = apodized.require_dataset('basic', shape=(hp.nside2npix(nside),), dtype=np.float32)
        data_dset2[...] = np.nan_to_num(mask_apodized)

if __name__ == '__main__':
    main()
