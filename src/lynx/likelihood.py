import jax.numpy as np
from jax import grad, hessian, jit
import jax
import camb
from pathlib import Path
import yaml
import h5py

__all__ = ['BBLogLike']
class BBLogLike(object):
    """ Object to calculate the log likelihood of a 
    given set of cosmological parameters.
    """
    def __init__(self, data=None, model=None, model_config_path=None, bpw_window_function=None):
        """ Setup the class with the observed data
        and its covariance. This should be BB bandpowers
        and their covariance matrix provided as a tuple.
        """
        self.data_setup(data)
        if model is not None:
            self.model_setup(model)
        elif model_config_path is not None:
            self.load_model_from_yaml(model_config_path)
        if bpw_window_function is not None:
            self.bpw_window_function_setup(bpw_window_function)
        else:
            self.lmax = len(data) - 1
        self.cosmo_setup()

    def __call__(self, theta, ret_neg=False):
        # if a bandpower window function has been defined, apply that
        # to the theoretical power spectrum
        model = self.model(theta)
        lnprior = self._lnprior(theta)
        lnp = lnprior + _log_gaussian(model, self.obs, self.inv_cov)
        if ret_neg:
            return - lnp
        return lnp 

    def model(self, theta):
        pars = {k: v for k, v in zip(self.free_parameters, theta)}
        # if a bandpower window function has been defined, apply that
        # to the theoretical power spectrum
        if self.apply_filtering:
            return self._Cl_BB_bpw(**pars)
        else:
            return self._Cl_BB(**pars)

    def bpw_window_function_setup(self, bpw_window_function):
        """ Method to set up filtering of theoretical power spectrum
        to compare with estimated bandpowers. 

        This is implemented by convolving with a set of bandpower
        window functions, computed using `NaMaster`.

        Parameters
        ----------
        filtering: ndarray
            Numpy array of shape (npws, nells).
        """
        nbpw, nells = bpw_window_function.shape
        try:
            assert nbpw == len(self.obs)
        except AssertionError:
            raise AssertionError(r"""
            Bandpower window function must have same number
            of bandpowers as observed power spectrum.
            Got:
                nbpw: {:d}
                len self obs: {:d} 
            """.format(nbpw, len(self.obs)))
        self.lmax = nells - 1
        self.bpw_window_function = bpw_window_function
        self.apply_filtering = True
        return

    def data_setup(self, data):
        obs, cov = data
        self.obs = obs
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)

    def load_model_from_yaml(self, fpath):
        with open(fpath) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.model_setup(cfg)
        return cfg['identifier']

    def model_setup(self, config):
        self.camb_ini_path = Path(config['camb_ini'])
        model = config['model']
        self._components = list(model.keys())
        self._priors = {}
        self._fixed_parameters = {}
        for component in model.values():
            # get list of free parameters for this component
            prior = component.get('varied', None)
            # if there are not free parameters prior is None
            if prior is not None: 
                self._priors.update(prior)
            # do the same for the fixed parameters
            fix_par = component.get('fixed', None)
            if fix_par is not None:
                self._fixed_parameters.update(fix_par)

        self.free_parameters = sorted(self._priors.keys())

    def cosmo_setup(self):
        """ This method runs CAMB with the cosmology specified
        by the configuration file `self.camb_ini_path`. From
        this run we take the primordial tensor BB spectrum and
        the lensing BB spectrum as templates for the likelihood
        BB model.
        """
        saved_output_path = self.camb_ini_path.with_suffix('.h5')
        if saved_output_path.exists():
            with h5py.File(saved_output_path.as_posix(), 'r') as f:
                if ('tensor' in f.keys()) and ('lensed_scalar' in f.keys()):
                    self._Cl_BB_prim = f['tensor'][:self.lmax + 1, 2]
                    self._Cl_BB_lens = f['lensed_scalar'][:self.lmax + 1, 2]
        else:
            pars = camb.read_ini(self.camb_ini_path.as_posix())
            pars.InitPower.set_params(r=1)
            pars.WantTensors = True
            results = camb.get_results(pars)
            powers = results.get_cmb_power_spectra(CMB_unit='muK', raw_cl=True)
            with h5py.File(saved_output_path.as_posix(), 'a') as f:
                for key, value in powers.items():
                    dset = f.require_dataset(key, dtype=value.dtype, shape=value.shape)
                    dset[...] = value
            self._Cl_BB_prim = np.array(powers['tensor'][:self.lmax+1, 2])
            self._Cl_BB_lens = np.array(powers['lensed_scalar'][:self.lmax+1, 2])

    def theta_0(self, npoints=None, seed=7837):
        """ Function to generate a starting guess for optimization
        or sampling by drawing a random point from the prior.

        Parameters
        ----------
        npoints: int (optional, default=None)
            If npoints is not None, function returns an array
            of draws from the prior of dimension (npoints, ndim).
            This can be useful for initializing a set of optimization
            runs, or samplers.
        seed: int (optional, default=7837)
            Seed for the PRNG key used by `jax`. `jax` has a subtly
            different approach to random number generation to numpy.
            Worth reading about this before setting this number.

        Returns
        -------
        ndarray
            Array of length the number of free parameters.
        """
        key = jax.random.PRNGKey(seed)
        if npoints is not None:
            shape = (npoints, len(self.free_parameters))
        else:
            shape = (len(self.free_parameters),)
        out = jax.random.normal(key, shape=shape, dtype=np.float32)
        means = []
        stds = []
        for par in self.free_parameters:
            mean, std = self._priors[par]
            means.append(mean)
            stds.append(std)
        means = np.array(means)
        stds = np.array(stds)
        return means + out * stds
    
    def covariance(self, theta):
        """ This method calculates the covariance of the 
        likeilihood parameters, `theta`, by computing the
        Hessian of the likelihood at position `theta`.

        Parameters
        ----------
        theta: ndarray
            Position in parameter space. 
    
        Returns
        -------
        ndarray
            Hessian matrix of the negative likelihood,
            corresponds to the parameter covarianace
            matrix at maximum likelihood.
        """
        hfunc = hessian(self.__call__, argnums=0)(theta, True)
        return np.linalg.inv(hfunc)

    def chi2(self, theta):
        res = self.model(theta) - self.obs
        res = res[4:-3]
        cov = self.cov[4:-3, 4:-3]
        inv_cov = np.linalg.inv(cov)
        dof = float(len(res) - len(self.free_parameters))
        return np.dot(res.T, np.dot(inv_cov, res)) / dof

    def _Cl_BB_bpw(self, r, A_L):
        return np.dot(self.bpw_window_function, self._Cl_BB(r, A_L))

    def _Cl_BB(self, r, A_L):
        return r * self._Cl_BB_prim + A_L * self._Cl_BB_lens

    def _lnprior(self, theta):
        logprior = 0
        for arg, par in zip(theta, self.free_parameters):
            mean, std = self._priors[par]
            inv_cov = 1. / std ** 2
            logprior += _log_gaussian(arg, mean, inv_cov)
        return logprior

@jit
def _log_gaussian(x, mean, inv_cov):
    r""" Function used to calculate a Gaussian prior.
    """
    diff = x - mean
    return - 0.5 * np.dot(np.transpose(diff), np.dot(inv_cov, diff))  