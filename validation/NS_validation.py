import os
# First thing, restrict to 1 CPU per thread for using MPI properly.
os.environ["OMP_NUM_THREADS"] = "1"

from nautilus import Prior, Sampler
from mpi4py.futures import MPIPoolExecutor
import mpi4py.MPI as MPI
from scipy.stats import norm #, multivariate_normal
import pyccl as ccl
from scipy.interpolate import interp1d
from itertools import pairwise
import argparse
import sys
sys.path.append('../')
from src import load_config, fisher_matrix, redshift_distributions, IO, fisher, names_to_latex
import numpy as np


def parse_args():
    # Parse the arguments and return the Namespace.
    parser = argparse.ArgumentParser(
        description='',
        epilog='')
    parser.add_argument('-c', '--config', dest='config',
                        help='Location of the configuration yaml file.',
                        default='../configs/config_main.yaml')
    parser.add_argument('-b', '--biased_config', dest='biased_config',
                        help='Location of the configuration yaml file for analysing '
                             'a biased cosmology.',
                        default=None)
    parser.add_argument('-o', '--output', dest='output_filename',
                        help='Location of the output hdf5 file.',
                        default='NS_validation.hdf5')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='If set, the sampling will continue from the last iteration.',
                        default=False)
    return parser.parse_args()


try:
    # Manage the MPI processes so that they receive the configuration.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        # If main process, read argument parser and send it to all other processes.
        # Note: broadcasting seems to mess up the printing and maybe more.
        args = parse_args()
        for i in range(1, size):
            comm.send(args, dest=i, tag=99)
    #    comm.bcast(args, root=0, )
    else:
        # Receive argument parser from main process.
        args = comm.recv(source=0, tag=99)
except:
    args = parse_args()

config = load_config(args.config)
print(f'Running fisher analysis for {config.name}.')
# Load the cosmology
ccl_cosmo_params = dict(zip(config.cosmology.keys(), config.cosmology.values()))
if 'Omega_m' in ccl_cosmo_params.keys():
    ccl_cosmo_params['Omega_c'] = ccl_cosmo_params['Omega_m'] - ccl_cosmo_params['Omega_b']
    ccl_cosmo_params.pop('Omega_m')
cosmo = ccl.Cosmology(**ccl_cosmo_params,
                      matter_power_spectrum='camb',
                      extra_parameters={"camb": {'dark_energy_model': 'ppf'} | config.baryons_dict})

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
IO.save_fisher_matrix(config, fm, 'fm_sampling_validation')

# Compute the theory data and covariance
C_ell_vec = fm.C_ell
covariance_matrix = fm.data_covariance
data = np.array(list(C_ell_vec.values())).flatten()
N_ell = len(ell_arr)
N_tracer_combinations = len(C_ell_vec.keys())
cov = np.zeros((N_ell*N_tracer_combinations, N_ell*N_tracer_combinations))
for i in range(N_tracer_combinations):
    for j in range(N_tracer_combinations):
        cov[i*N_ell:(i+1)*N_ell, j*N_ell:(j+1)*N_ell] = np.diag(covariance_matrix[:, i, j])
invcov = np.linalg.inv(cov)

# Priors
prior = Prior()
for ip, p in enumerate(config.sampling_validation.keys()):
    if config.sampling_validation[p] is None:
        continue
    elif p == 'delta_z':
        for i in range(N_z_bins):
            prior.add_parameter(f'dz_{i + 1}', dist=norm(loc=0.0,
                scale=config.redshift_distributions.sources.sigma_delta_z*(1 + mean_z[i])))
    elif len(config.sampling_validation[p]) == 2:
        prior.add_parameter(p, dist=tuple(config.sampling_validation[p]))
    else:
        raise ValueError(f'Input sampling parameter in config file had problem: {p}.')

if args.biased_config is not None:
    config = load_config(args.biased_config)

# Likelihood
def likelihood(param_dict):
    cosmo_in_dict = config.cosmology.copy()
    # These three: if set from fm, cannot run bias analysis.
    bayrons_dict_in = config.baryons_dict.copy()
    A_IA_in = fm.A_IA # FIXME: This should be config
    eta_in = fm.eta
    dndz_in = nz_arr

    for ip, p in enumerate(config.sampling_validation.keys()):
        if config.sampling_validation[p] is None:
            continue
        if p in config.cosmology.keys():
            if p == 'A_s':
                cosmo_in_dict[p] = 1.e-9 * param_dict[p]
            else:
                cosmo_in_dict[p] = 1. * param_dict[p]
        elif p == 'logA_s':
            cosmo_in_dict['A_s'] = 10. ** param_dict[p]
        elif p == 'logT_AGN':
            bayrons_dict_in = {'kmax': 20, "halofit_version": "mead2020_feedback",
                               "HMCode_logT_AGN": 1.*param_dict['logT_AGN']}
        elif p == 'A_IA':
            A_IA_in = 1.*param_dict['A_IA']
        elif p == 'eta':
            eta_in = 1.*param_dict['eta']
        elif p == 'delta_z':
            dndz_in = np.zeros(nz_arr.shape)
            for i in range(N_z_bins):
                interp_dndz = interp1d(z_arr, nz_arr[i], bounds_error=False, fill_value=0)
                dndz_in[i] = interp_dndz(z_arr + 1.*param_dict[f'dz_{i+1}'])
        else:
            raise ValueError(f'Input sampling parameter in config file had problem: {p}.')

    if cosmo_in_dict['w0']+cosmo_in_dict['wa'] > 0:
        return -np.inf, -np.inf
    if 'Omega_m' in cosmo_in_dict.keys():
        cosmo_in_dict['Omega_c'] = cosmo_in_dict['Omega_m'] - cosmo_in_dict['Omega_b']
        cosmo_in_dict.pop('Omega_m')
    try:
        bayrons_dict_in.update({"dark_energy_model": "ppf"})
        cosmo_in = ccl.Cosmology(**cosmo_in_dict,
                                 matter_power_spectrum='camb',
                                 extra_parameters={"camb": bayrons_dict_in})
        model = fisher.get_Cell_data_vector(cosmo_in, z_arr, dndz_in, A_IA_in, eta=eta_in, ell=ell_arr)
        model = np.array(list(model.values())).flatten()
        sigma8 = cosmo_in.sigma8()
        S8 = sigma8 * np.sqrt(cosmo_in['Omega_m']/0.3)
        #return multivariate_normal.logpdf(model, mean=data, cov=cov)
        return -0.5 * np.dot(np.dot(data-model, invcov), data-model), S8
    except:
        print(f'Problem with parameters: {param_dict.values()}')
        return -np.inf, -np.inf


if __name__ == '__main__':
    filename = args.output_filename
    resume = args.resume

    n_live_points = 3000
    f_live = 0.01

    try: pool_type = MPIPoolExecutor()
    except: pool_type = None

    sampler = Sampler(prior, likelihood,
                      blobs_dtype=[("S8", float)],
                      filepath=filename, resume=resume,
                      n_live=n_live_points,
                      pool=pool_type)
    sampler.run(verbose=True, discard_exploration=True, f_live=f_live)
