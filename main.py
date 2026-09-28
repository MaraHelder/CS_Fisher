import pyccl as ccl
import numpy as np
from itertools import pairwise
from src import load_config, fisher_matrix, redshift_distributions, IO, names_to_latex


def main(config):
    print(f'Running fisher analysis for {config.name}.')

    # Load the cosmology
    ccl_cosmo_params = dict(zip(config.cosmology.keys(), config.cosmology.values()))
    if 'Omega_m' in ccl_cosmo_params.keys():
        ccl_cosmo_params['Omega_c'] = ccl_cosmo_params['Omega_m'] - ccl_cosmo_params['Omega_b']
        ccl_cosmo_params.pop('Omega_m')
    cosmo = ccl.Cosmology(**ccl_cosmo_params,
                          matter_power_spectrum='camb',
                          extra_parameters = {"camb": config.baryons_dict})

    # ell binning setup
    ell_bins = np.geomspace(config.ell_binning.cosmic_shear.bin_start,
                                config.ell_binning.cosmic_shear.bin_end,
                                config.ell_binning.cosmic_shear.N_bins+1)
    ell_bins = ell_bins[
        (ell_bins >= config.ell_binning.cosmic_shear.ell_min) &
        (ell_bins <= config.ell_binning.cosmic_shear.ell_max)
        ].astype(int)
    ell_arr = np.array([(a + b) / 2 for a, b in pairwise(ell_bins)]).astype(int)
    Delta_ell = np.array([b - a for a, b in pairwise(ell_bins)])

    # Define the forecasting setup
    sigma_e = config.forecast.e_rms
    N_z_bins = config.redshift_distributions.sources.N_z_bins
    z_arr = np.linspace(config.forecast.z_min, config.forecast.z_max,
                        config.forecast.N_z_values)
    nz_arr, zbin_edges = (
        redshift_distributions.get_redshift_distributions(
            z_arr, N_z_bins,
            config.redshift_distributions.sources.z0,
            config.redshift_distributions.sources.a,
            config.redshift_distributions.sources.sigma_z))

    # Define the fisher matrix setup
    cosmo_params = {'name': list(config.cosmology.keys()),
                    'fiducial': list(config.cosmology.values()),
                    'shift': [config.derivatives.step_size[x] for x in list(config.cosmology.keys())],
                    'latex': [names_to_latex(x) for x in config.cosmology.keys()]}
    astro_params = {'name': list(config.IA.keys())+list(config.baryons.keys()),
                    'fiducial': list(config.IA.values())+list(config.baryons.values()),
                    'shift': [config.derivatives.step_size[x] for x in list(config.IA.keys())+list(config.baryons.keys())],
                    'latex': [names_to_latex(x) for x in list(config.IA.keys())+list(config.baryons.keys())]}
    redshift_params = {'name': [f'dz{i+1}' for i in range(N_z_bins)],
                       'fiducial': [0.] * N_z_bins,
                       'shift': [config.derivatives.step_size.delta_z] * N_z_bins,
                       'latex': [r'$\Delta z_{%i}$' % (i + 1) for i in range(N_z_bins)]}

    # Compute the fisher matrix
    print(f'Computing fisher matrix.')
    fm = fisher_matrix(cosmo=cosmo, z=z_arr, dndz=nz_arr, ell=ell_arr,
                       sigma_e=sigma_e,
                       n_bar=config.redshift_distributions.sources.nbar,
                       fsky=config.forecast.fsky, Delta_ell=Delta_ell,
                       n_points=config.derivatives.stencil_points, cosmo_params=cosmo_params,
                       astro_params=astro_params, redshift_params=redshift_params)
    # Add Gaussian photo-z priors
    mean_z = np.trapezoid(nz_arr * z_arr, z_arr)
    scale_z = config.redshift_distributions.sources.sigma_delta_z*(1+mean_z)
    fm.add_prior(redshift_params['name'], scale_z)

    print(f'Writing output.')
    IO.save_fisher_matrix(config, fm, 'fisher_matrix')

    # Fisher matrix validation
    print(f'Producing fisher matrix validation plot.')
    shifts = np.geomspace(1e-3, 1e-1, 32)
    FoM_parameters = config.derivatives.validation
    FoM_validation = fm.validate_fisher_matrix(shifts, FoM_parameters)

    print(f'Writing output.')
    IO.save_fisher_matrix(config, fm, 'fisher_matrix')
    IO.save_fisher_matrix_validation(config, shifts, FoM_validation, 'validation_fisher_matrix')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='',
        epilog='')
    parser.add_argument('-c', '--config', dest='config',
                        help='Location of the configuration yaml file.',
                        default='configs/config_main.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    main(config)
