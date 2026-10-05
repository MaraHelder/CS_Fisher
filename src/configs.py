import functools
import yaml
import os
from numpy import pi


class DictAsMember(dict):
    # Allows for getting attributes of dictionaries
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

    def __dir__(self):
        # Allows autocompletion
        return self.keys()


class StringConcatenation(yaml.YAMLObject):
    # Allows to join two variables in yaml file
    yaml_loader = yaml.SafeLoader
    yaml_tag = '!join'

    @classmethod
    def from_yaml(cls, loader, node):
        return functools.reduce(lambda a, b: os.path.join(a.value, b.value), node.value)


def load_config(config_file):
    with open(config_file, 'r') as cf:
        config = yaml.safe_load(cf)

    # Check for sigma8 and A_s.
    if 'sigma8' in config['cosmology'].keys() and 'A_s' in config['cosmology'].keys():
        raise ValueError('Cannot pass both sigma8 and A_s. '
                                                  'Set one to None.')

    # Check for IA and baryons in the config file.
    if 'IA' not in config.keys():
        config['IA'] = {'A_IA': None, 'a1h': None, 'a1': None, 'a2': None, 'ad': None}
    if 'baryons' not in config.keys():
        config['baryons'] = {'logT_AGN': None}
    config.update(baryons_dictionary(config))

    # Check for either halo model (a1h) or TATT model (a1, a2, ad) in the config file.
    if (config['IA']['a1h'] is not None) and (config['IA']['a1'] is not None or config['IA']['a2'] is not None or config['IA']['ad'] is not None):
        raise ValueError('Cannot pass both halo model (a1h) and TATT model (a1, a2, ad). '
                         'Set one to None.')

    # Add redshift information from SRD (Y1 and Y10)
    config.update(redshift_distributions_for_year(config['forecast']['year']))

    # Correct if some parameters are missing from the derivatives list, don't vary them.
    for key in (list(config['cosmology'].keys())+list(config['IA'].keys())+
                list(config['baryons'].keys())+['delta_z']):
        if key not in config['derivatives']['step_size']:
            config['derivatives']['step_size'][key] = None

    # Check if the parameters in the derivatives list have a fiducial value.
    for key in config['derivatives']['step_size'].keys():
        if config['derivatives']['step_size'][key] is not None:
            if key in config['cosmology'].keys():
                assert config['cosmology'][key] is not None, \
                    (f'Parameter {key} does not have a fiducial value but is expected'
                     f'to be varied under "derivatives/step_size".')

    return DictAsMember(config)


def redshift_distributions_for_year(year):
    # TODO: add lens numbers
    if year == 10:
        neff = 27  # arcmin2 from SRD p.54
        z0, a = 0.11, 0.68
        sigma_delta_z = 0.001
    elif year == 1:
        neff = 10
        z0, a = 0.13, 0.78
        sigma_delta_z = 0.002
    else:
        raise ValueError('Only year 1 and year 10 is supported.')
    N_z_bins = 5
    sigma_z = 0.05
    nbar = neff * (60 * 180 / pi) ** 2

    source_redshift_distributions = {
        'redshift_distributions': {
            'sources': {
                'z0': z0,
                'a': a,
                'sigma_z': sigma_z,
                'sigma_delta_z': sigma_delta_z,
                'nbar': nbar,
                'N_z_bins': N_z_bins
            }
        }
    }
    return source_redshift_distributions


def baryons_dictionary(config):
    if 'baryons' not in config.keys():
        return {"baryons_dict": {}}
    else:
        if 'logT_AGN' not in config['baryons'].keys():
            return {"baryons_dict": {}}
        else:
            if config['baryons']['logT_AGN'] is None:
                return {"baryons_dict": {}}
            else:
                return {"baryons_dict": {"kmax": 20.0, "halofit_version": "mead2020_feedback",
                                         "HMCode_logT_AGN": config['baryons']['logT_AGN']}}
