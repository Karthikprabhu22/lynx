#!/usr/bin/env python
# -*- coding: utf-8 -*-
#This script is used to make worKspace object for use in namaster using parameters defined in masking yaml file
import logging
from pathlib import Path
import sys
import os 

import click
import h5py
from astropy.io import fits
from astropy.wcs import WCS
import pymaster as nmt
import yaml
import healpy as hp 
import numpy as np


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
    mask_record = cfg['masks']['fitting']['one']['record']
    mask_apo_record = cfg['masks']['powerspectrum']['linear_lmin30lmax383']['record']
    wsp_path = cfg['masks']['powerspectrum']['linear_lmin30lmax383']['nmt_wsp']['path']
    beam_fwhm = cfg['masks']['powerspectrum']['linear_lmin30lmax383']['nmt_field']['args']['beam']
    ell_ini = cfg['masks']['powerspectrum']['linear_lmin30lmax383']['nmt_bin']['args']['ell_ini']
    ell_end = cfg['masks']['powerspectrum']['linear_lmin30lmax383']['nmt_bin']['args']['ell_end']

    npix = hp.nside2npix(nside)
    lmax = 3*nside -1
    ells = np.arange(0, lmax + 1)
    def get_beam(fwhm, lmax):
        """ FWHM in arcminutes.
        """
        fwhm = np.radians(fwhm / 60.)
        ells = np.arange(lmax + 1)
        assert ells[0] == 0
        assert ells[-1] == lmax
        sigma = fwhm / np.sqrt(8. * np.log(2))
        return np.exp(-(ells + 1) * ells / 2. * sigma ** 2)

    common_beam = get_beam(beam_fwhm, lmax)
    with h5py.File(outpath, 'r') as f:
        mask = f[mask_apo_record][...]    

    b = nmt.NmtBin.from_edges(ell_ini,ell_end, is_Dell=True)

    ell_arr = b.get_effective_ells()

    map_q = np.random.randn(npix)
    map_u = np.random.randn(npix)

    f_2 = nmt.NmtField(mask, [map_q,map_u], purify_b=True, beam=common_beam)
    wsp00 = nmt.NmtWorkspace()
    wsp00.compute_coupling_matrix(f_2, f_2, b)
    wsp00.write_to(wsp_path)
if __name__ == '__main__':
    main()
