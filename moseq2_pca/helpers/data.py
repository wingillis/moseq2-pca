"""
Helper functions for reading and loading PCA data.
"""

import h5py
from pathlib import Path
from moseq2_pca.util import read_yaml
from moseq2_pca.helpers.parameters import MouseProcessingParams, MaskParams, SVDConfig, create_dataclass_from_dict

def get_pca_paths(config_data, output_dir):
    """
    Helper function for changepoints_wrapper to perform data-path existence checks.
    Returns paths to saved pre-trained PCA components and PCA Scores files.

    Args:
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str | Path): path to directory to store PCA data

    Returns:
    config_data (dict): updated config_data dict with the pc component and pc score paths
    pca_file_components (Path): path to trained pca file
    pca_file_scores (Path): path to pca_scores file
    """

    # Check if there is PCA file from config_data
    if config_data.get('pca_file', None) is not None:
        pca_file = Path(config_data['pca_file'])
    else:
        # Assume PCA file is in output_dir
        pca_file = Path(output_dir) / 'pca.h5'
        config_data['pca_file'] = str(pca_file)

    if not pca_file.exists():
        raise IOError(f'Could not find PCA components file {pca_file}')

    # Get path to PCA Scores
    pca_file_scores = Path(config_data.get('pca_file_scores', Path(output_dir) / 'pca_scores.h5'))
    config_data['pca_file_scores'] = str(pca_file_scores)

    return config_data, pca_file, pca_file_scores

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
