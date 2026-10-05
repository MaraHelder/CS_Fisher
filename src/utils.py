import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt

def names_to_latex(parameter_name, dollar_signs=True):
    if parameter_name=='Omega_m':
        latex_name = r'\Omega_\mathrm{m}'
    elif parameter_name=='Omega_b':
        latex_name = r'\Omega_\mathrm{b}'
    elif parameter_name == 'h':
        latex_name = 'h'
    elif parameter_name == 'n_s':
        latex_name = r'n_\mathrm{s}'
    elif parameter_name == 'sigma8':
        latex_name = r'\sigma_8'
    elif parameter_name == 'A_s':
        latex_name = r'A_\mathrm{s}'
    elif parameter_name == 'w0':
        latex_name = 'w_0'
    elif parameter_name == 'wa':
        latex_name = 'w_a'

    elif parameter_name == 'A_IA':
        latex_name = r'A_\mathrm{IA}'
    elif parameter_name == 'eta':
        latex_name = r'\eta'
    elif parameter_name == 'a1h':
        latex_name = r'a_\mathrm{1h}'
    elif parameter_name == 'logT_AGN':
        latex_name = r'\log T_\mathrm{AGN}'

    elif parameter_name.startswith('dz'):
        z_bin = parameter_name.split('dz')[1]
        latex_name = rf'\Delta z_{z_bin}'

    elif parameter_name == 'A_s_9':
        latex_name = '10^{9}A_s'
    elif parameter_name == 'logA_s':
        latex_name = r'\log_{10}\left(A_s\right)'
    elif parameter_name == 'S8':
        latex_name = 'S_8'

    else:
        raise ValueError(f'Not recognised parameter {parameter_name} and cannot return its latex string.')

    if dollar_signs:
        return fr'${latex_name}$'
    else:
        return fr'{latex_name}'

def sigma8_derivative(cosmo_dict, parameter, shift, n_points):
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

    dsigma8 = 0.
    if parameter == 'A_s':
        shift *= 1.e-9
    for n in range(n_points - 1):
        param_in = cosmo_dict[parameter]
        param_in += step_coeff[n] * shift
        # Define cosmology input dictionary

        cosmo_in_dict = cosmo_dict.copy()
        cosmo_in_dict[parameter] = param_in
        if 'Omega_m' in cosmo_in_dict.keys():
            cosmo_in_dict['Omega_c'] = cosmo_in_dict['Omega_m'] - cosmo_in_dict['Omega_b']
            cosmo_in_dict.pop('Omega_m')

        cosmo_in = ccl.Cosmology(**cosmo_in_dict, matter_power_spectrum='camb',
                                 extra_parameters={"camb": {"dark_energy_model": "ppf"}})

        dsigma8 += coeff[n] * cosmo_in.sigma8() / shift

    return dsigma8

def get_Pk_of_k_a_IA(cosmo: ccl.Cosmology, a1h: float,
                     A_IA: float, a_1: float, a_2: float, a_d: float, z: np.ndarray,
                     k_arr: np.ndarray = np.geomspace(1E-3, 1e3, 128), a_arr: np.ndarray = np.linspace(0.1, 1, 32)) \
        -> tuple[ccl.pk2d.Pk2D, ccl.pk2d.Pk2D]:
    """
    Computes the intrinsic alignment power spectra P(k,a) for the GI and II terms using the halo model or TATT model. If a1h is not None, the halo model is used.

    Args:
        cosmo (object): A CCL cosmology object.
        a1h (float): Value of the 1-halo term amplitude.
        A_IA (float): Value of the intrinsic alignment amplitude.
        a_1 (float): Value of the first alignment bias.
        a_2 (float): Value of the second alignment bias.
        a_d (float): Value of the overdensity alignment bias.
        z (np.ndarray): An array of redshifts at which to compute the power spectra.
        k_arr (np.ndarray): An array of wavenumbers (in units of 1/Mpc) at which to compute the power spectra.
        a_arr (np.ndarray): An array of scale factors at which to compute the power spectra.

    Returns:
        p_of_k_a (tuple): A tuple containing the P(k,a) for the GI and II term (ccl.pk2d.Pk2D objects) in that order.
    """
    if a1h is None & (a_1 is not None or a_2 is not None or a_d is not None):
        c_1,c_d,c_2 = pt.translate_IA_norm(cosmo, z=z, a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2 = False)
        ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(z,c_1), c2=(z,c_2), cdelta=(z,c_d))
        ptc = pt.EulerianPTCalculator(with_NC=False, with_IA=True, # TODO: with_NC?
                      log10k_min=np.log10(k_arr.min()), log10k_max=np.log10(k_arr.max()), nk_per_decade=20, arr=a_arr) # TODO:nk_per_decade?
        ptc.update_ingredients(cosmo)

        pk_II = ptc.get_biased_pk2d(ptt_i, ptt_i)
        pk_GI = ptc.get_biased_pk2d(ccl.tracers.CMBLensingTracer(cosmo), ptt_i)
        return [pk_GI, pk_II]
    elif a1h is not None & (a_1 is None & a_2 is None & a_d is None):
        lk_arr = np.log(k_arr)

        # the halo mass definition
        hm_def = '200m'
        # the Duffy 2008 concentration-mass relation,
        cM = ccl.halos.ConcentrationDuffy08(mass_def=hm_def)
        # the Tinker 2010 halo mass function,
        nM = ccl.halos.MassFuncTinker10(mass_def=hm_def)
        # the Tinker 2010 halo bias,
        bM = ccl.halos.HaloBiasTinker10(mass_def=hm_def)
        # the halo model calculator
        hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hm_def)
        # the NFW halo profile
        NFW =  ccl.halos.HaloProfileNFW(mass_def=hm_def, concentration=cM, truncated=True, fourier_analytic=True)
        # the satellite shear HOD profile
        sat_gamma_HOD = ccl.halos.SatelliteShearHOD(concentration=cM, mass_def=hm_def, a1h=a1h, b=-2)
        
        C1rhocrit = 5e-14*ccl.physical_constants.RHO_CRITICAL
        C = A_IA * C1rhocrit * cosmo['Omega_m'] / cosmo.growth_factor(a_arr)

        pk_II_1h_ss = ccl.halos.halomod_Pk2D(cosmo, hmc, sat_gamma_HOD, get_2h = False, a_arr=a_arr, lk_arr=lk_arr)
        pk_GI_1h_s = ccl.halos.halomod_Pk2D(cosmo, hmc, NFW, prof2 = sat_gamma_HOD, get_2h = False, a_arr=a_arr, lk_arr=lk_arr)

        k1h = 4*cosmo['h'] #1/Mpc
        k2h = 6*cosmo['h'] #1/Mpc

        pk_II_NLA_windowed = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=lk_arr,
                                        pk_arr=C.reshape(-1,1)**2*cosmo.nonlin_matter_power(np.e**lk_arr, a_arr)*np.exp(-(k_arr/k2h)**2).reshape(1,-1),
                                        is_logp=False)
        pk_GI_NLA_windowed = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=lk_arr,
                                        pk_arr=-C.reshape(-1,1)*cosmo.nonlin_matter_power(np.e**lk_arr, a_arr)*np.exp(-(k_arr/k2h)**2).reshape(1,-1),
                                        is_logp=False)

        pk_II_1h_ss_windowed = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=lk_arr,
                                            pk_arr=pk_II_1h_ss(k_arr, a_arr)*(1-np.exp(-(k_arr/k1h)**2)).reshape(1,-1),
                                            is_logp=False)
        pk_GI_1h_s_windowed = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=lk_arr,
                                            pk_arr=pk_GI_1h_s(k_arr, a_arr)*(1-np.exp(-(k_arr/k1h)**2)).reshape(1,-1),
                                            is_logp=False)
        pk_II_windowed = pk_II_NLA_windowed+pk_II_1h_ss_windowed
        pk_GI_windowed = pk_GI_NLA_windowed+pk_GI_1h_s_windowed

        p_k_of_k_a = [pk_II_windowed, pk_GI_windowed]
        
        return p_k_of_k_a