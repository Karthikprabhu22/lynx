import yaml
import h5py
from pathlib import Path

import pymaster as nmt

import healpy as hp
import numpy as np

class Masking(object):
    def __init__(self, cfg_path):
        with open(cfg_path) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.hdf5_path = Path(self.cfg['hdf5_path'])
        self.nside = self.cfg['nside']
        self.masks = {k: v for k, v in self.cfg['masks'].items()}

    def __str__(self):
        return yaml.dump(self.cfg, allow_unicode=True, default_flow_style=False)

    def get_masks(self, application):
        """ Application must be 'fitting' or 'powerspectrum'
        """
        for name, value in self.masks[application].items():
            with h5py.File(self.hdf5_path, 'r') as f:
                mask = f[value['record']][...]
            yield name, mask

    def calculate_mode_coupling(self):
        # iterate through masks defined in the power spectrum list
        for var in self.masks['powerspectrum'].values():
            # read apodized mask from hdf5 file
            with h5py.File(self.hdf5_path, 'r') as f:
                mask = f[var['record']][...]
            # create the binning object
            binning = nmt.NmtBin.from_edges(**var['nmt_bin']['args'])
            # define some random fields on which to calculate the
            # mode coupling matrix
            beam = var['nmt_field']['args'].pop('beam', None)
            if beam is not None:
                beam = _gaussian_beam(beam / 60. * np.pi / 180., np.arange(3 * self.nside))
            q1, u1, q2, u2 = np.random.randn(4, len(mask))
            f1 = nmt.NmtField(mask, [q1, u1], **var['nmt_field']['args'], beam=beam)
            f2 = nmt.NmtField(mask, [q2, u2], **var['nmt_field']['args'], beam=beam)
            # get namaster workspace object and compute coupling matrix
            wsp = nmt.NmtWorkspace()
            wsp.compute_coupling_matrix(f1, f2, binning)
            path = Path(var['nmt_wsp']['path'])
            if path.exists():
                path.unlink()
            # save to file
            wsp.write_to(str(path))
    
    def save_fitting_mask(self, name, mask):
        with h5py.File(self.hdf5_path, 'a') as f:
            dset = f.require_dataset(self.cfg['masks']['fitting'][name]['record'], shape=mask.shape, dtype=mask.dtype)
            dset[...] = mask

    def get_binary(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            mask = f[self.cfg['masks']['binary']['record']][...]
        return mask

    def get_nmt_workspaces(self, recalculate=False):
        if recalculate:
            self.calculate_mode_coupling()
        for name, value in self.masks['powerspectrum'].items():
            wsp = nmt.NmtWorkspace()
            wsp.read_from(value['nmt_wsp']['path'])
            yield name, wsp

    def calculate_apodizations(self, mask_in):
        for value in self.masks['powerspectrum'].values():
            apo_mask = nmt.mask_apodization(mask_in, **value['nmt_apo']['args'])
            with h5py.File(self.hdf5_path, 'a') as f:
                dset = f.require_dataset(value['record'], shape=apo_mask.shape, dtype=apo_mask.dtype)
                dset[...] = apo_mask

    def get_fitting_indices(self):
        for name, value in self.masks['fitting'].items():
            # read in fitting mask
            with h5py.File(self.hdf5_path, 'r') as f:
                mask = f[value['record']][...].astype(int)
            
            # remove the 0 region, which is masked out
            regions = set(mask)
            if 0 in regions:
                regions.remove(0)
            # loop through the fitting regions, and return
            # the corresponding healpix indices
            fitting_parameters = [(i, np.where(mask==i)[0].astype(int)) for i in regions]
            yield name, fitting_parameters

    def get_powerspectrum_tools(self, recalculate=False):
        if recalculate:
            self.calculate_mode_coupling()
        for name, value in self.masks['powerspectrum'].items():
            # read in the precomputed namaster workspace
            wsp = nmt.NmtWorkspace()
            wsp.read_from(value['nmt_wsp']['path'])

            # get the corresponding apodized mask
            with h5py.File(self.hdf5_path, 'r') as f:
                mask = f[value['record']][...]

            # get the binning scheme
            binning = nmt.NmtBin.from_edges(**value['nmt_bin']['args'])

            # get beam array
            beam = value['nmt_field']['args'].pop('beam', None)
            if beam is not None:
                beam = _gaussian_beam(beam / 60. * np.pi / 180., np.arange(3 * self.nside))
            yield name, wsp, mask, binning, beam
        return

def _gaussian_beam(fwhm, ells):
    r""" Function to calculate a Gaussian beam function in harmonic space.

    Parameters
    ----------
    fwhm: float
        Full width at half-maximum of the Gaussian beam, in radians.
    ells: ndarray
        Array containing the multipoles over which to calculate the beam.

    Returns
    -------
    ndarray
        Array containing the beam function.
    """
    sigma = fwhm / np.sqrt(8. * np.log(2))
    return np.exp(- ells * (ells + 1) * sigma ** 2 / 2.)