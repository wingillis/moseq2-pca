"""
Helper functions for reading and loading PCA data.
"""

import h5py
from pathlib import Path
from moseq2_pca.util import read_yaml


def load_pcs_for_cp(pca_file, config_data):
    """
    Load computed Principal Components for Model-free Changepoint Analysis.

    Args:
    pca_file (str | Path): path to pca h5 file to read PCs
    config_data (dict): config parameters

    Returns:
    pca_file (Path): path to pca components
    changepoint_params (dict): dict of relevant changepoint parameters
    missing_data (bool): Indicates whether to use mask_params for missing data pca
    mask_params (dict): Mask parameters to use when computing CPs
    """

    pca_file = Path(pca_file)
    print(f'Loading PCs from {pca_file}')
    with h5py.File(pca_file, 'r') as f:
        pca_components = f[config_data['pca_path']][()]

    # get the yaml for pca, check parameters
    pca_yaml = pca_file.with_suffix('.yaml')

    if pca_yaml.exists():
        pca_config = read_yaml(pca_yaml)

        missing_data = pca_config.get('missing_data', False)
        if missing_data:
            print('Detected missing data...')
            mask_params = {
                'mask_height_threshold': pca_config['mask_height_threshold'],
                'mask_threshold': pca_config['mask_threshold']
            }
        else:
            mask_params = None

        if missing_data and not Path(config_data['pca_file_scores']).exists():
            raise RuntimeError("Need PCA scores to impute missing data, run apply pca first")

    # Pack changepoint parameters
    changepoint_params = {
        'k': config_data['klags'],
        'sigma': config_data['sigma'],
        'peak_height': config_data['threshold'],
        'peak_neighbors': config_data['neighbors'],
    }

    return pca_components, changepoint_params, missing_data, mask_params
