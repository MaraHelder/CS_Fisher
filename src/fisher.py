import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import warnings
from .utils import sigma8_derivative, names_to_latex, get_Pk_of_k_a_IA

def get_Cell_data_vector(cosmo: ccl.Cosmology,
                         z: np.ndarray, dndz: np.ndarray,
                         A_IA: float, eta: float = None, z0 :float = 0.62,
                         ell: np.ndarray = None, p_of_k_a : dict = None) \
        -> dict:
    """
    Computes the cosmic shear C_ell's. Assumes constant A_IA.

    Args:
        cosmo (object): A CCL cosmology object.
        z (array): The redshift values where the n(z) has been computed.
        dndz (array): The redshift distribution n(z) for each redshift bin.
            If 1-D, one redshift bin is assumed. Otherwise, the dimensions should be
            (n_z_bins, n_z_values).
        A_IA (float): Value of the intrinsic alignment amplitude.
        eta (float): Value of the redshift power-law index (A~A_IA*(1+z)**eta).
            Default is None, without scaling.
        z0 (float): The pivot redshift for the power-law scaling.
            Default is 0.62 (not used if eta is None).
        ell (array): Values of multipoles where the C_ell's will be computed.

    Returns:
        c_ells (dict): A dictionary with keys corresponding to tracers and values to C-ells.
            The tracers will be 'zi-zj', where i,j denote the redshift bins.
            :param z0:
    """
    dndz_use = np.atleast_2d(dndz)
    n_z_bins = dndz_use.shape[0]
    if A_IA is None:
        ia_bias = None
    else:
        if eta is None:
            ia_bias = (z, A_IA*np.ones_like(z))
        else:
            ia_bias = (z, A_IA * ((1+z)/(1+z0))**eta)

    c_ells = {}
    for z1 in range(n_z_bins):
        for z2 in range(n_z_bins):
            if z2 < z1: continue
            if p_of_k_a is None:
                tracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz_use[z1]),
                                                ia_bias=ia_bias)
                tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z, dndz_use[z2]),
                                                ia_bias=ia_bias)
                c_ells[f'z{z1}-z{z2}'] = ccl.angular_cl(cosmo, tracer1, tracer2, ell)
            else:
                wl_tracer_z1 = ccl.WeakLensingTracer(cosmo,dndz = (z, dndz_use[z1]))
                wl_tracer_z2 = ccl.WeakLensingTracer(cosmo,dndz = (z, dndz_use[z2]))
                ia_tracer_z1 = ccl.WeakLensingTracer(cosmo,dndz = (z, dndz_use[z1]), has_shear=False, ia_bias = ia_bias, use_A_ia=False)
                ia_tracer_z2 = ccl.WeakLensingTracer(cosmo,dndz = (z, dndz_use[z2]), has_shear=False, ia_bias = ia_bias, use_A_ia=False)

                c_ell_GG = ccl.angular_cl(cosmo, wl_tracer_z1, wl_tracer_z2, ell)
                c_ell_GI = ccl.angular_cl(cosmo, wl_tracer_z1, ia_tracer_z2, ell, p_of_k_a=p_of_k_a[0])
                c_ell_IG = ccl.angular_cl(cosmo, ia_tracer_z1, wl_tracer_z2, ell, p_of_k_a=p_of_k_a[0])
                c_ell_II = ccl.angular_cl(cosmo, ia_tracer_z1, ia_tracer_z2, ell, p_of_k_a=p_of_k_a[1])
                c_ells[f'z{z1}-z{z2}'] = c_ell_GG + c_ell_GI + c_ell_IG + c_ell_II
    return c_ells


def get_nt_nz(c_ells: dict) -> (int, int):
    """
    Returns the number of redshift bins, given a C_ell's dictionary.

    Args:
        c_ells (dict): A dictionary obtained with ```get_Cell_data_vector```.
    """
    tracer_combinations = list(c_ells.keys())
    # FIXME: This doesn't work for more than 10 bins!
    n_zbins = len(np.unique([x[1] for x in tracer_combinations]))
    return n_zbins


def get_covariance(ell: np.ndarray, c_ells: dict,
                   n_bar: {float, iter}, sig_e: float,
                   f_sky: float = 1., Delta_ell: np.ndarray = None)\
        -> np.ndarray:
    """
    Computes the Gaussian shape-noise covariance terms of the given C_ell's.

    Args:
        ell (array): Values of ell multipoles where the C_ell's have been computed.
        c_ells (dict): A dictionary obtained with ```get_Cell_data_vector```.
        n_bar (float or iterable): Number density in steradians. If float, the value is
            for the whole survey and will be divided by the number of redshift bins internally.
            If iterable, it must have the number density per redshift bin in ascending
             bin order (from low to high redshift).
        sig_e (float or iterable): Ellipticity rms.
        f_sky (float): Fraction of sky covered by the assumed survey, from 0 to 1.
        Delta_ell (None or array): The distance between the ell bins. If None, the bins
            are assumed to be linear and have a distance of 1.

    Returns:
        covariance (np.ndarray): A 3-dimensional array where the first axis
            contains the numerical value of the covariance for the given ell's,
            and the second and third axes contain the two combination of redshift
            bins correlated.
    """
    n_bar = np.atleast_1d(n_bar)
    n_z_bins = get_nt_nz(c_ells)
    assert len(n_bar) == 1 or len(n_bar) == n_z_bins, \
        'Number density per bin provided but not for all redshift bins.'
    if len(n_bar) == 1:
        n_bar = np.full(n_z_bins, n_bar[0] / n_z_bins)
    tracer_combinations = c_ells.keys()
    c_ell_vec_full = c_ells.copy()

    noise = dict()
    for z1 in range(n_z_bins):
        for z2 in range(n_z_bins):
            if f'z{z1}-z{z2}' not in tracer_combinations:
                c_ell_vec_full[f'z{z1}-z{z2}'] = c_ells[f'z{z2}-z{z1}']
            if z1 == z2:
                noise[f'z{z1}-z{z2}'] = sig_e**2/(2*n_bar[z1])
            else:
                noise[f'z{z1}-z{z2}'] = 0

    if Delta_ell is None:
        cov_prefactor = 1 / ((2 * ell + 1) * f_sky)
    else:
        cov_prefactor = 1 / (2 * f_sky * ell * Delta_ell)
    covariance = np.zeros((len(ell), len(tracer_combinations), len(tracer_combinations)))
    for i, tc1 in enumerate(tracer_combinations):
        for j, tc2 in enumerate(tracer_combinations):
            if i > j: continue
            t1, t2 = tc1.split('-')
            t3, t4 = tc2.split('-')
            covariance[:, i, j] = ((c_ell_vec_full[f'{t1}-{t3}'] + noise[f'{t1}-{t3}'])
                                   * (c_ell_vec_full[f'{t2}-{t4}'] + noise[f'{t2}-{t4}'])
                                   + (c_ell_vec_full[f'{t1}-{t4}'] + noise[f'{t1}-{t4}'])
                                   * (c_ell_vec_full[f'{t2}-{t3}'] + noise[f'{t2}-{t3}'])
                                   ) * cov_prefactor
            covariance[:, j, i] = covariance[:, i, j]
    return covariance


def compute_SNR(c_ells: dict, covariance: np.ndarray) -> float:
    """
    Computes the signal-to-noise ratio given a C_ell's dictionary and
    a covariance array.

    Args:
        c_ells (dict): A dictionary of C_ell's obtained with ```get_Cell_data_vector```.
        covariance (array): An array containing the covariance obtained with
            ```get_covariance```.
    """
    N_ells = covariance.shape[0]
    data = np.array(list(c_ells.values())).T
    SNR = 0
    for l in range(N_ells):
        invcov = np.linalg.inv(covariance[l])
        SNR += np.dot(np.dot(data[l].T, invcov), data[l])
    return SNR ** 0.5


def compute_d_Cells(n_points: int,
                    cosmo_params: dict, astro_params: dict, redshift_params: dict,
                    z: np.ndarray, dndz: np.ndarray, ell: np.ndarray,
                    verbose: bool = False) -> np.ndarray:
    """
    Computes the numerical derivatives of the C_ell's with respect to the input
    cosmological, intrinsic alignment and photo-z uncertainty parameters using
    the n_point stencil method.

    Args:
        n_points (float): The number of points to use for the stencil method to
            compute the derivatives.
        cosmo_params (dict): A dictionary with the keys "fiducial" and a LIST of
            all fiducial values for the cosmological parameters used. It should
            also contain the key "shift" with a LIST of values for the step-size
            of the derivatives for each of the parameters. If some parameter has
            a corresponding "None" shift, it is not being varied.
        astro_params (dict): Similar to the one above but now contains the
            astrophysical systematic parameter(s): intrinsic alignments (A_IA)
            and baryonic feedback (logT_AGN).
        redshift_params (dict): Similar to the one above but now contains the
            shifts on the redshift distributions.
        z (array): The redshift values where the n(z) has been computed.
        dndz (array): The redshift distribution n(z) for each redshift bin.
            If 1-D, one redshift bin is assumed. Otherwise, the dimensions should be
            (n_z_bins, n_z_values).
        ell (array): Values of ell multipoles where the C_ell's have been computed.
        verbose (bool): If true, print statements for when derivatives are taken
            for different parameters will be created.

    Returns:
        d_Cells (np.ndarray): A 3-dimensional array. Its first axis contains the
            ell value. The second axis contains the different parameters with
            respect to which derivatives are taken. These are the parameters in
            cosmo_params, IA_params and redshift_params, in numerical ascending
            order. The third axis contains the different combination of redshift
            bins that were used to compute the C_ell's.
    """
    dndz_use = np.atleast_2d(dndz)
    N_cosmo = len(cosmo_params['fiducial'])
    N_z_bins = dndz_use.shape[0]
    N_Cells = int(np.sum(np.arange(1, N_z_bins+1)))
    N_params_varied = int(np.count_nonzero(cosmo_params['shift']) +
                          np.count_nonzero(astro_params['shift']) +
                          np.count_nonzero(redshift_params['shift']))
    d_Cells = np.zeros((len(ell), N_params_varied, N_Cells))

    # TODO: Automate this for whichever n_points.
    if n_points == 3:
        coeff = [-1 / 2, 1 / 2]
    elif n_points == 5:
        coeff = [1 / 12, -2 / 3, 2 / 3, -1 / 12]
    elif n_points == 7:
        coeff = [-1 / 60, 3 / 20, -3 / 4, 3 / 4, -3 / 20, 1 / 60]
    elif n_points == 9:
        coeff = [1 / 280, -4 / 105, 1 / 5, -4 / 5, 4 / 5, -1 / 5, 4 / 105, -1 / 280]
    else:
        raise ValueError(f'The n_points {n_points:d} given is not supported (use 3, 5, 7 or 9).')
    step_coeff = np.arange(-(n_points // 2), n_points // 2 + 1, 1)
    step_coeff = step_coeff[step_coeff != 0]

    params_fiducial = np.concatenate((cosmo_params['fiducial'], astro_params['fiducial'], redshift_params['fiducial']))
    params_name = np.concatenate((cosmo_params['name'], astro_params['name'], redshift_params['name']))
    params_shift = np.concatenate((cosmo_params['shift'], astro_params['shift'], redshift_params['shift']))

    di = 0
    for pi, p in enumerate(params_fiducial):
        if params_shift[pi] is None: continue
        # TODO: verbose is currently not accessible.
        if verbose: print(f'Computing derivatives for parameter '
                          f'{params_name[pi]} with fiducial value '
                          f'{params_fiducial[pi]} and derivative shift '
                          f'{params_shift[pi]}.')
        if params_name[pi] == 'A_s':
            params_shift[pi] *= 1.e-9
        for n in range(n_points-1):
            param_in = params_fiducial.copy()
            param_in[pi] += step_coeff[n] * params_shift[pi]
            # Define cosmology input dictionary
            cosmo_params_in = param_in[:N_cosmo]
            cosmo_in_dict = dict(zip(cosmo_params['name'], cosmo_params_in))
            if 'Omega_m' in cosmo_in_dict.keys():
                cosmo_in_dict['Omega_c'] = cosmo_in_dict['Omega_m'] - cosmo_in_dict['Omega_b']
                cosmo_in_dict.pop('Omega_m')

            # Define the IA input parameters
            A_IA_in = param_in[np.isin(params_name, 'A_IA')][0]
            eta_in = param_in[np.isin(params_name, 'eta')][0]
            a1h_in = param_in[np.isin(params_name, 'a1h')][0]
            a1_in = param_in[np.isin(params_name, 'a1')][0]
            a2_in = param_in[np.isin(params_name, 'a2')][0]
            ad_in = param_in[np.isin(params_name, 'ad')][0]

            # Define the baryon feedback parameters
            try:
                logT_AGN_in = param_in[np.isin(params_name, 'logT_AGN')][0]
                if logT_AGN_in is not None:
                    baryons_dict = {"kmax": 20.0,
                                   "halofit_version": "mead2020_feedback",
                                   "HMCode_logT_AGN": logT_AGN_in}
                else: baryons_dict = {}
            except:
                baryons_dict = {}

            # Define the input redshift distributions
            dndz_in = dndz_use.copy()
            if pi >= len(params_fiducial)-N_z_bins:
                delta_z_in = param_in[pi]
                shifted_bin = pi-(len(params_fiducial)-N_z_bins)
                interp_dndz = interp1d(z, dndz_use[shifted_bin], bounds_error=False, fill_value=0)
                dndz_in[shifted_bin] = interp_dndz(z + delta_z_in)
                dndz_in[shifted_bin] /= np.trapezoid(dndz_in[shifted_bin], z+delta_z_in)

            cosmo_in = ccl.Cosmology(**cosmo_in_dict,
                                     matter_power_spectrum='camb',
                                     extra_parameters={"camb": {"dark_energy_model": "ppf"} | baryons_dict})

            if (a1h_in is None) and (a1_in is None) and (a2_in is None) and (ad_in is None):
                C_ells = get_Cell_data_vector(cosmo_in, z, dndz_in, A_IA_in, eta_in, ell=ell)
            else:
                pk_of_k_a = get_Pk_of_k_a_IA(cosmo_in, a1h=a1h_in, A_IA=A_IA_in, a_1=a1_in, a_2=a2_in, a_d=ad_in,
                                             z=z) # TODO: add k_arr and a_arr as input parameters to this function.
                C_ells = get_Cell_data_vector(cosmo_in, z, dndz_in, A_IA_in, eta_in, ell=ell, p_of_k_a=pk_of_k_a)
            d_Cells[:, di, :] += coeff[n] * np.array(list(C_ells.values())).T / params_shift[pi]
        di += 1
    return d_Cells


def compute_Fisher_matrix(d_Cells: np.ndarray, covariance: np.ndarray):
    """
    Computes the Fisher matrix given C_ell's derivatives and covariance arrays.

    Args:
        d_Cells (array): The output from ```compute_d_Cells```.
        covariance (array): The output from ```get_covariance```.

    Returns:
        fisher (array): An array of size NxN, where N is the number of
            parameters varied (given by d_Cells.shape[1]).
    """
    N_params = d_Cells.shape[1]

    fisher = np.zeros((N_params, N_params))
    for l in range(covariance.shape[0]):
        invcov = np.linalg.inv(covariance[l])
        fisher += (d_Cells[l].dot(invcov)).dot(d_Cells[l].T)
    return fisher


class fisher_matrix(object):

    def __init__(self, *, cosmo=None, z=None, dndz=None,
                 ell=None, sigma_e=None, n_bar=None, fsky=None,
                 Delta_ell=None, n_points=3,
                 cosmo_params=None, astro_params=None, redshift_params=None,
                 fisher_from_input=None,
                 IA_params=None):

        if IA_params is not None:
            warnings.warn('IA_params is deprecated and will be removed. Change to "astro_params" instead.',
                          DeprecationWarning, stacklevel=2)
            astro_params = IA_params

        self.cosmo = cosmo
        self.z = z
        self.dndz = dndz
        self.ell = ell
        self.sigma_e = sigma_e
        self.n_bar = n_bar
        self.fsky = fsky
        self.Delta_ell = Delta_ell
        self.n_points = n_points
        self.cosmo_params = cosmo_params
        self.astro_params = astro_params
        self.redshift_params = redshift_params
        self.fisher_from_input = fisher_from_input

        if cosmo_params is None:
            self.cosmo_params = {'name': [], 'fiducial': [], 'shift': []}
        if astro_params is None:
            self.astro_params = {'name': ['A_IA', 'eta', 'a1h', 'a1', 'a2', 'ad'], 'fiducial': [None, None, None, None, None, None], 'shift': [None, None, None, None, None, None]}
        if redshift_params is None:
            self.redshift_params = {'name': [], 'fiducial': [], 'shift': []}

        assert all(s in self.cosmo_params for s in ['name', 'fiducial']) and \
               all(s in self.astro_params for s in ['name', 'fiducial']) and \
               all(s in self.redshift_params for s in ['name', 'fiducial'])

        self.A_IA = np.array(self.astro_params['fiducial'])[np.isin(self.astro_params['name'], 'A_IA')][0]
        self.a1h = np.array(self.astro_params['fiducial'])[np.isin(self.astro_params['name'], 'a1h')][0]
        self.eta = np.array(self.astro_params['fiducial'])[np.isin(self.astro_params['name'], 'eta')][0]
        self.a1 = np.array(self.astro_params['fiducial'])[np.isin(self.astro_params['name'], 'a1')][0]
        self.a2 = np.array(self.astro_params['fiducial'])[np.isin(self.astro_params['name'], 'a2')][0]
        self.ad = np.array(self.astro_params['fiducial'])[np.isin(self.astro_params['name'], 'ad')][0]
        for param_dict in [self.cosmo_params, self.astro_params, self.redshift_params]:
            if 'latex' not in param_dict:
                param_dict['latex'] = param_dict['name']
            if 'shift' not in param_dict:
                param_dict['shift'] = [-1.]*len(param_dict['name'])

        if fisher_from_input is None:
            self.C_ell = get_Cell_data_vector(self.cosmo, self.z, self.dndz, self.A_IA, self.eta, ell=self.ell, p_of_k_a=get_Pk_of_k_a_IA(self.cosmo, a1h=self.a1h, A_IA=self.A_IA, a_1=self.a1, a_2=self.a2, a_d=self.ad, z=self.z))
            self.data_covariance = get_covariance(self.ell, self.C_ell,
                                                  self.n_bar, self.sigma_e,
                                                  f_sky=self.fsky, Delta_ell=self.Delta_ell)
            self.SNR = compute_SNR(self.C_ell, self.data_covariance)
            self.d_C_ell = compute_d_Cells(n_points=self.n_points,
                                           cosmo_params=self.cosmo_params,
                                           astro_params=self.astro_params,
                                           redshift_params=self.redshift_params,
                                           z=self.z, dndz=self.dndz, ell=self.ell)
            self.fisher_matrix = compute_Fisher_matrix(self.d_C_ell, self.data_covariance)
        else:
            self.fisher_matrix = fisher_from_input

        params_shift = np.concatenate((self.cosmo_params['shift'],
                                       self.astro_params['shift'],
                                       self.redshift_params['shift']))
        ids_varied = np.where(params_shift != None)[0]

        self.parameters = np.concatenate((self.cosmo_params['name'],
                                          self.astro_params['name'],
                                          self.redshift_params['name'])
                                         )[ids_varied]
        self.fiducial_parameters = np.concatenate((self.cosmo_params['fiducial'],
                                                   self.astro_params['fiducial'],
                                                   self.redshift_params['fiducial'])
                                                  )[ids_varied]
        self.latex_parameters = np.concatenate((self.cosmo_params['latex'],
                                                self.astro_params['latex'],
                                                self.redshift_params['latex'])
                                               )[ids_varied]
        self.covariance = np.linalg.inv(self.fisher_matrix)
        self.dim = len(self.parameters)

    def __getitem__(self, *keys):
        """ Determines behavior of `self[key]` """
        keys_in = np.atleast_1d(keys[0])
        assert all(s in self.parameters for s in keys_in), \
            f'Parameter is not in the Fisher matrix list of parameters.'

        keys_indices = np.arange(len(self.parameters))[np.isin(self.parameters, keys_in)]
        return self.fisher_matrix[np.ix_(keys_indices, keys_indices)]

    def fix_parameters(self, parameters):
        parameters_in = np.atleast_1d(parameters)
        assert all(s in self.parameters for s in parameters_in)
        keys_indices = np.arange(len(self.parameters))[~np.isin(self.parameters, parameters_in)]

        ret = self.copy()
        ret.fisher_matrix = self.fisher_matrix[np.ix_(keys_indices, keys_indices)]
        ret.covariance = np.linalg.inv(ret.fisher_matrix)
        #ret.d_C_ell = ret.d_C_ell[:, keys_indices, :]

        ret.parameters = ret.parameters[keys_indices]
        ret.fiducial_parameters = ret.fiducial_parameters[keys_indices]
        ret.latex_parameters = ret.latex_parameters[keys_indices]
        ret.dim = len(ret.parameters)

        ret.cosmo_params = None
        ret.astro_params = None
        ret.redshift_params = None

        return ret

    def copy(self):
        fm = fisher_matrix.__new__(fisher_matrix)
        for key, item in self.__dict__.items():
            if isinstance(item, np.ndarray):
                # Only items that might change need to be deep-copied.
                fm.__dict__[key] = item.copy()
            else:
                fm.__dict__[key] = item
        return fm

    def add_prior(self, parameters: {str, iter},
                  sigma_parameters: {float, iter}):
        """
        Adds a Gaussian prior to the parameters provided.

        Args:
            parameters (iterable): The parameter(s) for which a Gaussian prior
                will be added to the Fisher matrix.
            sigma_parameters (iterable): The scale of the Gaussian distribution
                of the prior.
        """
        parameters_ = np.atleast_1d(parameters)
        if not np.any(np.isin(parameters_, self.parameters)):
            return
        sigma_parameters_ = np.atleast_1d(sigma_parameters)
        assert len(parameters_) == len(sigma_parameters_)
        for i, param in enumerate(parameters_):
            param_id = np.isin(self.parameters, param)
            self.fisher_matrix[param_id, param_id] += 1./sigma_parameters_[i]**2
        self.covariance = np.linalg.inv(self.fisher_matrix)
        self.priors = [parameters_, sigma_parameters_]

    def marginalised_covariance(self, parameters: iter) -> np.ndarray:
        """
        Returns the marginalised covariance of the Fisher matrix.

        Args:
            parameters (iterable): The parameter(s) over which the marginalised
                covariance will be computed.
        """
        # FIXME: This returns the matrix with parameters sorted.
        param_indices = np.arange(len(self.parameters))[np.isin(self.parameters, parameters)]
        return self.covariance[np.ix_(param_indices, param_indices)]

    def transform_fisher_matrix(self, jacobian: np.ndarray, param_old: str,
                                param_new: str, param_new_value: float):
        assert param_new not in self.parameters, ('The parameter you are trying to transform'
                                                  'is already in the Fisher matrix.')
        ret = self.copy()
        ret.fisher_matrix = np.dot(np.dot(jacobian.T, self.fisher_matrix), jacobian)
        ret.covariance = np.linalg.inv(ret.fisher_matrix)

        param_idx = np.arange(self.dim)[np.isin(self.parameters, param_old)]
        ret.parameters[param_idx] = param_new
        ret.fiducial_parameters[param_idx] = param_new_value
        ret.latex_parameters[param_idx] = names_to_latex(param_new)

        # FIXME: Is there a better way to do this and keep these quantities?
        ret.d_C_ell = None
        ret.cosmo_params = None
        ret.astro_params = None
        ret.redshift_params = None

        return ret

    def transform_A_s_to_sigma8(self):
        cosmo_dict = dict(zip(self.cosmo_params['name'], self.cosmo_params['fiducial']))
        cosmo_params_for_derivative = np.intersect1d(self.cosmo_params['name'], self.parameters)
        dsig8_arr = np.array([sigma8_derivative(cosmo_dict, i, 0.01, 3) for i in cosmo_params_for_derivative])
        dsig8 = dict(zip(cosmo_params_for_derivative, dsig8_arr))

        # we computed partial derivatives d sig8/d cosmo_params but we need d cosmo_params/d sig8.
        # For that, we need the inverse Jacobian.
        A_s_idx = np.arange(self.dim)[np.isin(self.parameters, 'A_s')]
        M = np.identity(self.dim)
        for ip, p in enumerate(self.parameters):
            if p in self.cosmo_params['name']:
                M[ip, A_s_idx] = dsig8[p]
        M = np.linalg.inv(M)

        return self.transform_fisher_matrix(M, 'A_s', 'sigma8', self.cosmo.sigma8())

    def transform_A_s_to_S8(self):
        cosmo_dict = dict(zip(self.cosmo_params['name'], self.cosmo_params['fiducial']))
        cosmo_params_for_derivative = np.intersect1d(self.cosmo_params['name'], self.parameters)
        dsig8_arr = np.array([sigma8_derivative(cosmo_dict, i, 0.01, 3) for i in cosmo_params_for_derivative])
        dsig8 = dict(zip(cosmo_params_for_derivative, dsig8_arr))

        # we computed partial derivatives d sig8/d cosmo_params but we need d cosmo_params/d S8.
        # In general, d S8/ d theta = S8/sigma8 d sigma8/d theta
        # But d S8/d Omega_m += sigma8/(0.6*S8)
        # Then, we need the inverse Jacobian.
        Omega_m_idx = np.arange(self.dim)[np.isin(self.parameters, 'Omega_m')]
        A_s_idx = np.arange(self.dim)[np.isin(self.parameters, 'A_s')]
        sigma8 = self.cosmo.sigma8().astype(float)
        Omega_m = self.fiducial_parameters[Omega_m_idx].astype(float)
        S8 = (Omega_m/0.3)**0.5 * sigma8

        M = np.identity(self.dim)
        for ip, p in enumerate(self.parameters):
            if p in self.cosmo_params['name']:
                M[A_s_idx, ip] = S8 / sigma8 * dsig8[p]
        M[A_s_idx, Omega_m_idx] += sigma8**2/(0.6*S8)
        M = np.linalg.inv(M)

        return self.transform_fisher_matrix(M, 'A_s', 'S8', S8[0])

    def transform_A_s_to_m9A_s(self):
        M = np.identity(self.dim)
        A_s_idx = np.arange(self.dim)[np.isin(self.parameters, 'A_s')]
        M[A_s_idx, A_s_idx] = 1.e-9

        return self.transform_fisher_matrix(M, 'A_s', 'A_s_9',
                                            1.e9*self.fiducial_parameters[A_s_idx][0])

    def transform_A_s_to_logA_s(self):
        M = np.identity(self.dim)
        A_s_idx = np.arange(self.dim)[np.isin(self.parameters, 'A_s')]
        M[A_s_idx, A_s_idx] = np.log(10) * self.fiducial_parameters[A_s_idx]

        return self.transform_fisher_matrix(M, 'A_s', 'logA_s',
                                            np.log10(self.fiducial_parameters[A_s_idx][0]))

    def figure_of_merit(self, parameters: iter) -> float:
        """
        Returns the figure-of-merit for the given parameters.

        Args:
            parameters (iterable): The parameter(s) over which the figure-of-
                merit will be computed.
        """
        return np.sqrt(np.linalg.det(np.linalg.inv(self.marginalised_covariance(parameters))))

    def draw_covariance_ellipse(self, ax: object, parameters: iter, mu: iter = None,
                                CL: float = 0.95, color: str = None, fill: bool = True,
                                **kwargs)\
            -> (object, tuple, tuple, tuple, tuple):
        """
        Draws an ellipse with a marginalised covariance matrix computed over
        the parameters provided.

        Args:
          ax (object): Axes matplotlib object where to draw.
          parameters (2-dim iterable): The parameters over which the ellipse
            will be drawn.
          mu (iter): An iterable with 2 elements, the centers of the ellipse.
            If None, the fisher matrix object's fiducial values will be the center.
          CL (float): Confidence interval on which to draw the ellipse (0<CL<1).
          color (str or None): The color of ellipse (leave None for default).
          fill (bool): whether to return filled ellipses or not.

        Returns:
            ax (object): The matplotlib axes object that contains the ellipse.
            xlim (tuple): The limits on the x-axis that encompass the ellipse.
            ylim (tuple): The limits on the y-axis that encompass the ellipse.
            mu (tuple): The center of the ellipse.
            latex (tuple): The latex string of the parameters.
        """

        # TODO: Make this into an iterable to be able to draw multiple ellipses with one call.
        # Convert CL into covariance scale factor.
        assert 0 < CL < 1
        scale = chi2.ppf(q=CL, df=2)

        assert len(parameters) == 2
        C = self.marginalised_covariance(parameters)
        if mu is None:
            # FIXME: These returns parameters sorted.
            mu = self.fiducial_parameters[np.isin(self.parameters, parameters)]
        latex = self.latex_parameters[np.isin(self.parameters, parameters)]
        # Below we have 128 bit float numbers for precision when using A_s
        a = C[0, 0].astype(np.float64)  # sigma_x^2
        b = C[0, 1].astype(np.float64)  # sigma_xy
        c = C[1, 1].astype(np.float64) # sigma_y^2

        # semi-major axis (squared and scaled)
        lambda1 = scale * 0.5 * (a + c + np.sqrt((a - c) ** 2 + 4 * b ** 2))
        # semi-minor axis (squared and scaled)
        lambda2 = scale * 0.5 * (a + c - np.sqrt((a - c) ** 2 + 4 * b ** 2))
        # angle
        angle = np.rad2deg(0.5 * np.arctan2(2 * b, a - c))

        # Multiply by 2 because Ellipse takes major/minor axis
        width = 2 * np.sqrt(lambda1)
        height = 2 * np.sqrt(lambda2)

        ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle,
                          facecolor=color, edgecolor=color, fill=fill, **kwargs)
        ax.add_artist(ellipse)
        xlim = np.sqrt(scale * C[0, 0])
        ylim = np.sqrt(scale * C[1, 1])
        return ax, xlim, ylim, mu, latex

    def validate_fisher_matrix(self, shifts: np.ndarray = None,
                               FoM_parameters: iter = None,
                               shift_parameters: iter = None) -> np.ndarray:
        """
        Vary the derivative step size for each parameter varied in the matrix and
        compute the figure-of-merit of some parameters to study the matrix's
        stability.

        Args:
            shifts (array): An array with the numerical value of the shifts to
                be explored, used to compute the numerical derivatives.
            FoM_parameters (iterable): The parameter(s) over which the figure-
                of-merit will be computed with each new derivative.
            shift_parameters (iterable): The parameters over which the derivatives
                will be computed, each with the values given in "shifts". If None,
                all parameters of the Fisher matrix are used.

        Returns:
            FoM_output (array): A 2-dimensional array that contains the resulting
                figure-of-merit. The varied parameters are in its first axis and
                the different shift values in its second axis.
        """
        if self.fisher_from_input is not None:
            raise ValueError('Cannot validate fisher matrix given from input.')
        if self.cosmo_params is None:
            raise ValueError('Cannot validate fisher matrix that has been copied, '
                             'or that has no cosmology parameters.')
        if shifts is None:
            shifts = np.geomspace(1e-3, 1e-1, 10)
        if shift_parameters is None:
            shift_parameters = self.parameters
        shift_parameters = np.atleast_1d(shift_parameters)
        N_parameters = len(shift_parameters)
        FoM_output = np.zeros((N_parameters, len(shifts)))

        for i, shift in enumerate(shifts):
            for pi, p in enumerate(shift_parameters):

                cosmo_params_shift = self.cosmo_params.copy()
                astro_params_shift = self.astro_params.copy()
                redshift_params_shift = self.redshift_params.copy()

                for param_dict in [cosmo_params_shift, astro_params_shift, redshift_params_shift]:
                    param_dict['shift'] = [None]*len(param_dict['name'])
                    if p in param_dict['name']:
                        pi_where = np.isin(param_dict['name'], p).nonzero()[0][0]
                        param_dict['shift'][pi_where] = shift
                try:
                    d_Cell_shift = compute_d_Cells(self.n_points,
                                                   cosmo_params=cosmo_params_shift,
                                                   astro_params=astro_params_shift,
                                                   redshift_params=redshift_params_shift,
                                                   z=self.z, dndz=self.dndz, ell=self.ell)
                except:
                    warnings.warn(f'Unable to compute derivative of {p} for shift {shift:.4f}.')
                    FoM_output[pi, i] = -1
                    continue
                d_Cell_fiducial = self.d_C_ell.copy()
                d_Cell_fiducial[:, pi, :] = d_Cell_shift[:, 0, :]

                fisher_shift = fisher_matrix(fisher_from_input=compute_Fisher_matrix(d_Cell_fiducial,
                                                                                     self.data_covariance),
                                             cosmo_params=self.cosmo_params,
                                             astro_params=self.astro_params,
                                             redshift_params=self.redshift_params)
                try: fisher_shift.add_prior(self.priors[0], self.priors[1])
                except: pass
                FoM_output[pi, i] = fisher_shift.figure_of_merit(FoM_parameters)
        return FoM_output
