import os
import numpy as np
import pyccl as ccl
from src import fisher_matrix, names_to_latex
from src.configs import DictAsMember


# TODO: I should change this all...
def save_fisher_matrix(config: DictAsMember, fisher: fisher_matrix, name: 'str'):
    output_dir = config.paths.output.fisher_matrix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, f'{config.name}_{name}.npy'),
            fisher.fisher_matrix)


def save_fisher_matrix_validation(config: DictAsMember,
                                  shifts_array: np.ndarray,
                                  validation_array: np.ndarray,
                                  name: str):
    output_dir = config.paths.output.fisher_matrix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.savez(os.path.join(output_dir, f'{config.name}_{name}.npz'),
             shifts=shifts_array, validation=validation_array)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, f'{config1.name}_{config2.name}_{name}.npy'),
            bias_vector)


def load_fisher_matrix(config: DictAsMember,
                       name: str) -> fisher_matrix:
    fisher_matrix_array = np.load(os.path.join(config.paths.output.fisher_matrix,
                                               f'{config.name}_{name}.npy'))
    # Load the cosmology
    ccl_cosmo_params = dict(zip(config.cosmology.keys(), config.cosmology.values()))
    if 'Omega_m' in ccl_cosmo_params.keys():
        ccl_cosmo_params['Omega_c'] = ccl_cosmo_params['Omega_m'] - ccl_cosmo_params['Omega_b']
        ccl_cosmo_params.pop('Omega_m')
    cosmo = ccl.Cosmology(**ccl_cosmo_params,
                          matter_power_spectrum='camb',
                          extra_parameters={"camb": config.baryons_dict})

    N_z_bins = config.redshift_distributions.sources.N_z_bins

    cosmo_params = {'name': list(config.cosmology.keys()),
                    'fiducial': list(config.cosmology.values()),
                    'shift': [config.derivatives.step_size[x] for x in list(config.cosmology.keys())],
                    'latex': [names_to_latex(x) for x in config.cosmology.keys()]}
    astro_params = {'name': list(config.IA.keys()) + list(config.baryons.keys()),
                    'fiducial': list(config.IA.values()) + list(config.baryons.values()),
                    'shift': [config.derivatives.step_size[x] for x in list(config.IA.keys()) + list(config.baryons.keys())],
                    'latex': [names_to_latex(x) for x in list(config.IA.keys())+list(config.baryons.keys())]}
    redshift_params = {'name': [f'dz{i+1}' for i in range(N_z_bins)],
                       'fiducial': [0.] * N_z_bins,
                       'shift': [config.derivatives.step_size.delta_z] * N_z_bins,
                       'latex': [r'$\Delta z_{%i}$' % (i + 1) for i in range(N_z_bins)]}

    fisher_matrix_object = fisher_matrix(cosmo=cosmo,
                                         fisher_from_input=fisher_matrix_array,
                                         cosmo_params=cosmo_params,
                                         astro_params=astro_params,
                                         redshift_params=redshift_params)
    return fisher_matrix_object


def load_fisher_matrix_validation(config: DictAsMember,
                                  name: str) -> (np.ndarray, np.ndarray):
    data = np.load(os.path.join(config.paths.output.fisher_matrix,
                                f'{config.name}_{name}.npz'))
    return data['shifts'], data['validation']
