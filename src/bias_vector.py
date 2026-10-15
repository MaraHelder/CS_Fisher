from itertools import pairwise
import numpy as np
import pyccl as ccl

from src import redshift_distributions, IO
from src.configs import load_config
from src.utils import names_to_latex

from .fisher import get_Cell_data_vector, compute_Fisher_matrix, get_covariance, compute_d_Cells, get_Pk_of_k_a_IA

def compute_bias_vector(config1, config2):
    print(f'Running bias vector analysis for {config1.name} assumed and {config2.name}.')

    # Load the cosmologies for both configurations
    ccl_cosmo_params_config1 = dict(zip(config1.cosmology.keys(), config1.cosmology.values()))
    ccl_cosmo_params_config2 = dict(zip(config2.cosmology.keys(), config2.cosmology.values()))

    if 'Omega_m' in ccl_cosmo_params_config1.keys():
        ccl_cosmo_params_config1['Omega_c'] = ccl_cosmo_params_config1['Omega_m'] - ccl_cosmo_params_config1['Omega_b']
        ccl_cosmo_params_config1.pop('Omega_m')
    if 'Omega_m' in ccl_cosmo_params_config2.keys():
        ccl_cosmo_params_config2['Omega_c'] = ccl_cosmo_params_config2['Omega_m'] - ccl_cosmo_params_config2['Omega_b']
        ccl_cosmo_params_config2.pop('Omega_m')

    cosmo_config1 = ccl.Cosmology(**ccl_cosmo_params_config1,   
                                  matter_power_spectrum='camb',
                                  extra_parameters = {"camb": config1.baryons_dict})
    cosmo_config2 = ccl.Cosmology(**ccl_cosmo_params_config2,   
                                  matter_power_spectrum='camb',
                                  extra_parameters = {"camb": config2.baryons_dict})

    # ell binning setup (should be the same for both configurations)
    # test that bin_start, bin_end, N_bins, ell_min, and ell_max are the same for both configurations
    if (config1.ell_binning.cosmic_shear.bin_start != config2.ell_binning.cosmic_shear.bin_start or
        config1.ell_binning.cosmic_shear.bin_end != config2.ell_binning.cosmic_shear.bin_end or
        config1.ell_binning.cosmic_shear.N_bins != config2.ell_binning.cosmic_shear.N_bins or
        config1.ell_binning.cosmic_shear.ell_min != config2.ell_binning.cosmic_shear.ell_min or
        config1.ell_binning.cosmic_shear.ell_max != config2.ell_binning.cosmic_shear.ell_max):
        raise ValueError("The ell binning parameters must be the same for both configurations.")
    
    ell_bins = np.geomspace(config1.ell_binning.cosmic_shear.bin_start,
                            config1.ell_binning.cosmic_shear.bin_end,
                            config1.ell_binning.cosmic_shear.N_bins+1)
    ell_bins = ell_bins[
        (ell_bins >= config1.ell_binning.cosmic_shear.ell_min) &
        (ell_bins <= config1.ell_binning.cosmic_shear.ell_max)
        ].astype(int)
    ell_arr = np.array([(a + b) / 2 for a, b in pairwise(ell_bins)]).astype(int)   
    Delta_ell = np.array([b - a for a, b in pairwise(ell_bins)])

    # Define the forecasting setup (should be the same for both configurations)
    # test that e_rms, z_min, z_max, N_z_values, and nbar are the same for both configurations
    if (config1.forecast.e_rms != config2.forecast.e_rms or
        config1.forecast.z_min != config2.forecast.z_min or
        config1.forecast.z_max != config2.forecast.z_max or
        config1.forecast.N_z_values != config2.forecast.N_z_values or
        config1.redshift_distributions.sources.nbar != config2.redshift_distributions.sources.nbar):
        raise ValueError("The forecasting parameters must be the same for both configurations.")

    sigma_e = config1.forecast.e_rms
    N_z_bins = config1.redshift_distributions.sources.N_z_bins
    z_arr = np.linspace(config1.forecast.z_min, config1.forecast.z_max,
                        config1.forecast.N_z_values)
    nz_arr, zbin_edges = (
        redshift_distributions.get_redshift_distributions(
            z_arr, N_z_bins,
            config1.redshift_distributions.sources.z0,
            config1.redshift_distributions.sources.a,
            config1.redshift_distributions.sources.sigma_z))

    # Compute both Cells data vectors and its difference
    print(f'Computing Cells data vectors for both configurations.')
    pk_of_k_a_config1= get_Pk_of_k_a_IA(cosmo_config1, a1h=config1.IA.a1h, A_IA=config1.IA.A_IA,
                                        a_1=config1.IA.a_1, a_2=config1.IA.a_2, a_d=config1.IA.a_d,
                                        z=z_arr)
    C_ells_config1= get_Cell_data_vector(cosmo_config1, z_arr, nz_arr, config1.IA.A_IA, config1.IA.eta,
                                        ell=ell_arr, p_of_k_a=pk_of_k_a_config1)

    pk_of_k_a_config2= get_Pk_of_k_a_IA(cosmo_config2, a1h=config2.IA.a1h, A_IA=config2.IA.A_IA,
                                        a_1=config2.IA.a_1, a_2=config2.IA.a_2, a_d=config2.IA.a_d,
                                        z=z_arr)
    C_ells_config2= get_Cell_data_vector(cosmo_config2, z_arr, nz_arr, config2.IA.A_IA, config2.IA.eta,
                                        ell=ell_arr, p_of_k_a=pk_of_k_a_config2)
    dC_ells = np.array(list(C_ells_config2.values())) - np.array(list(C_ells_config1.values()))

    # Compute the covariance matrix
    print(f'Computing covariance matrix for configuration 1.') 
    covariance = get_covariance(ell_arr, C_ells_config1, n_bar=config1.redshift_distributions.sources.nbar,
                                       sig_e=sigma_e, f_sky=config1.forecast.f_sky, Delta_ell=Delta_ell)

    # Compute the derivatives of the Cells data vector with respect to the cosmological parameters for configuration 1
    print(f'Computing derivatives of Cells data vector for configuration 1.')
    cosmo_params = {'name': list(config1.cosmology.keys()),
                        'fiducial': list(config1.cosmology.values()),
                        'shift': [config1.derivatives.step_size[x] for x in list(config1.cosmology.keys())],
                        'latex': [names_to_latex(x) for x in config1.cosmology.keys()]}
    astro_params = {'name': list(config1.IA.keys())+list(config1.baryons.keys()),
                        'fiducial': list(config1.IA.values())+list(config1.baryons.values()),
                        'shift': [config1.derivatives.step_size[x] for x in list(config1.IA.keys())+list(config1.baryons.keys())],
                        'latex': [names_to_latex(x) for x in list(config1.IA.keys())+list(config1.baryons.keys())]}
    redshift_params = {'name': [f'dz{i+1}' for i in range(N_z_bins)],
                           'fiducial': [0.] * N_z_bins,
                           'shift': [config1.derivatives.step_size.delta_z] * N_z_bins,
                           'latex': [r'$\Delta z_{%i}$' % (i + 1) for i in range(N_z_bins)]}
    
    dC_ells_dtheta = compute_d_Cells(config1.derivatives.stencil_points, cosmo_params, astro_params, redshift_params, z_arr, nz_arr, ell_arr)

    # Compute the Fisher matrix for configuration 1
    print(f'Computing Fisher matrix for configuration 1.')
    fisher_matrix_config1 = compute_Fisher_matrix(dC_ells_dtheta, covariance)
    inverse_fisher_matrix_config1 = np.linalg.inv(fisher_matrix_config1)

    # Compute the bias vector
    print(f'Computing bias vector.')
    bias_vector = # TODO: How to compute the sum, understand the dimensions of covariance
    bias = inverse_fisher_matrix_config1 @ bias_vector

    print(f'Writing output.')
    IO.save_bias_vector(config1, config2, bias, 'bias_vector')

if __name__ == '__bias_vector__':
    import argparse
    parser = argparse.ArgumentParser(
        description='',
        epilog='')
    parser.add_argument('-c1', '--config1', dest='config1',
                        help='Location of the first configuration yaml file.',
                        default='configs/config_main.yaml')
    parser.add_argument('-c2', '--config2', dest='config2',
                        help='Location of the second configuration yaml file.',
                        default='configs/config_main.yaml')
    args = parser.parse_args()

    config1 = load_config(args.config1)
    config2 = load_config(args.config2)

    compute_bias_vector(config1, config2)