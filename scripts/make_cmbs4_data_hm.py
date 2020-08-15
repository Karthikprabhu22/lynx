#!/usr/bin/env python
# -*- coding: utf-8 -*-

#######################################################################
#
# This script is used to consolidate CMBS4 half mission simulations for B-mode forecasting using lynx.
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
import pysm3
import pysm3.units as u
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


    def factor_CMB_RJ(nu):
        x = 0.0176086761 * nu
        ex = np.exp(x)
        factor = ex * (x / (ex - 1)) ** 2
        return factor
    nside = 512
    lmax = 500
    new_beam_fwhm = 60.
    
    def beam(fwhm, lmax):
        """ FWHM in arcminutes.
        """
        fwhm = np.radians(fwhm / 60.)
        ells = np.arange(lmax + 1)
        assert ells[0] == 0
        assert ells[-1] == lmax
        sigma = fwhm / np.sqrt(8. * np.log(2))
        return np.exp(-(ells + 1) * ells / 2. * sigma ** 2)

    common_beam = beam(new_beam_fwhm, lmax)


    def reconvolve(maps,old_beam_fwhm):
        deconvolution = 1. / beam(old_beam_fwhm, lmax)
        maps_alms = hp.map2alm(maps) 
        maps_alms_new = [hp.almxfl(alms, deconvolution * common_beam) for alms in maps_alms]
        newmap = hp.alm2map(maps_alms_new, nside)
        return old_np.array([newmap[1],newmap[2]])

    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    freqs = old_np.array(cfg['frequencies']) 
    nside = cfg['nside']
    npix = hp.nside2npix(nside)
    sensitivities = old_np.array(cfg['sensitivities']) 
    #sensitivities = np.array([w.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(f)) for w, f in zip(sensitivities, freqs)])
    sensitivities = np.array([w*factor_CMB_RJ(f) for w,f in zip(sensitivities,freqs)])
    sensitivities = sensitivities*1e-6
    

    nmc = cfg['monte_carlo']
    beams = cfg['fwhm']
    outpath = cfg['hdf5_path']
    half_mission_noise = cfg['half_mission_noise']
    mapspath = cfg['maps_path']
    maskpath = cfg['mask_path']
    noisepath = cfg['noise_path']
    gdustpath = cfg['gdust_path']
    gsyncpath = cfg['gsync_path']

    realizations = ["0000","0001","0002","0003","0004","0005","0006","0007","0008","0009"]
    freq = ["f030_b77","f040_b58","f085_b27","f095_b24","f145_b16","f155_b15","f220_b11","f270_b08"]

    if half_mission_noise:
        sensitivities = [s * np.sqrt(2.) for s in sensitivities]
    
    logging.info(f"""
    Frequencies: {freqs!s}
    Nside: {nside:04d}
    Sensitivities: {sensitivities!s}
    Number of Monte Carlo Simulations: {nmc:05d}
    """)
    
    mask_file = maskpath+'n0512.h5'
    with h5py.File(mask_file, 'r') as f:
        mask_data = f['ns0512/apodized/basic'][...]
        mask_binary = f['ns0512/binary'][...]
    cov = mask_data.astype(np.float32)

    logging.info(f"Output path: {outpath}")


    with h5py.File(outpath, 'a') as f:
        f.attrs.update({'config': yaml.dump(cfg)})
        maps = f.require_group('maps')
        monte_carlo = maps.require_group('monte_carlo')

        data_dset = monte_carlo.require_dataset('data', shape=(nmc, len(freqs), 2, hp.nside2npix(nside)), dtype=np.float32)

        cov_dset = monte_carlo.require_dataset('cov', shape=(nmc, len(freqs), 2, hp.nside2npix(nside)), dtype=np.float32)
        for i in range(len(freqs)):
            cov_dset[:,i,:,:] = old_np.nan_to_num(np.array(sensitivities)[i]/cov.astype(np.float32))
        
        for imc in np.arange(nmc)[::2]:
            for j in range(len(freq)):
                cmb_file = "cmbs4_04_llcdm_"+freq[j]+"_ellmin30_map_0512_mc_"+realizations[imc]+".fits"
                dust_file = "gdust_"+freq[j]+"_ellmin30_map_0512_mc_"+realizations[imc]+".fits"
                sync_file = "gsync_"+freq[j]+"_ellmin30_map_0512_mc_"+realizations[imc]+".fits"
                cmb = reconvolve(old_np.nan_to_num(hp.read_map(mapspath+cmb_file,field=[0,1,2],dtype=np.float32)),beams[j])
                sync = reconvolve(old_np.nan_to_num(hp.read_map(gsyncpath+sync_file,field=[0,1,2],dtype=np.float32)),beams[j])
                dust = reconvolve(old_np.nan_to_num(hp.read_map(gdustpath+dust_file,field=[0,1,2],dtype=np.float32)),beams[j])

                for k in range(imc, imc + 2):          
                    noise_file = "cmbs4_04_noise_"+freq[j]+"_ellmin30_map_0512_mc_"+realizations[k]+".fits"
                    noise = old_np.nan_to_num(hp.read_map(noisepath+noise_file,field=[1,2],dtype=np.float32))
                    data = old_np.nan_to_num(cmb +  dust + sync + np.sqrt(2)*noise)
                    data_dset[k,j] = data.astype(np.float32) * factor_CMB_RJ(freqs[j])

if __name__ == '__main__':
    main()
