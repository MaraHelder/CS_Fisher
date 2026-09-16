import numpy as np
import pyccl as ccl


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
