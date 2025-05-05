"""
Helper functions for reading and loading PCA data.
"""

import h5py
import ruamel.yaml as yaml
from os.path import join, exists, splitext
from moseq2_pca.util import read_yaml
from moseq2_pca.helpers.parameters import DataProcessingParams, ChangepointParams, MaskParams, ProcessingConfig

def get_pca_paths(config_data, output_dir):
    """
    Helper function for changepoints_wrapper to perform data-path existence checks.
    Returns paths to saved pre-trained PCA components and PCA Scores files.

    Args:
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data

    Returns:
    config_data (dict): updated config_data dict with the pc component and pc score paths
    pca_file_components (str): path to trained pca file
    pca_file_scores (str): path to pca_scores file
    """

    # Check if there is PCA file from config_data
    if config_data.get('pca_file', None) is not None:
        pca_file = config_data['pca_file']
    else:
        # Assume PCA file is in output_dir
        pca_file = join(output_dir, 'pca.h5')
        config_data['pca_file'] = pca_file

    if not exists(pca_file):
        raise IOError(f'Could not find PCA components file {pca_file}')

    # Get path to PCA Scores
    pca_file_scores = config_data.get('pca_file_scores', join(output_dir, 'pca_scores.h5'))
    config_data['pca_file_scores'] = pca_file_scores

    return config_data, pca_file, pca_file_scores

def load_pcs_for_cp(pca_file, config_data):
    """
    Load computed Principal Components for Model-free Changepoint Analysis.

    Args:
    pca_file (str): path to pca h5 file to read PCs
    config_data (dict): config parameters

    Returns:
    pca_file (str): path to pca components
    changepoint_params (dict): dict of relevant changepoint parameters
    missing_data (bool): Indicates whether to use mask_params for missing data pca
    mask_params (dict): Mask parameters to use when computing CPs
    """

    print(f'Loading PCs from {pca_file}')
    with h5py.File(pca_file, 'r') as f:
        pca_components = f[config_data['pca_path']][()]

    # get the yaml for pca, check parameters, if we used fft, be sure to turn on here...
    pca_yaml = splitext(pca_file)[0] + '.yaml'

    if exists(pca_yaml):
        with open(pca_yaml, 'r') as f:
            pca_config = yaml.safe_load(f.read())

            missing_data = pca_config.get('missing_data', False)
            if missing_data:
                print('Detected missing data...')
                mask_params = {
                    'mask_height_threshold': pca_config['mask_height_threshold'],
                    'mask_threshold': pca_config['mask_threshold']
                }
            else:
                mask_params = None

            if missing_data and not exists(config_data['pca_file_scores']):
                raise RuntimeError("Need PCA scores to impute missing data, run apply pca first")

    # Pack changepoint parameters
    changepoint_params = {
        'k': config_data['klags'],
        'sigma': config_data['sigma'],
        'peak_height': config_data['threshold'],
        'peak_neighbors': config_data['neighbors'],
    }

    return pca_components, changepoint_params, missing_data, mask_params

def get_pca_yaml_data(pca_yaml):
    """
    Reads PCA yaml file and returns enclosed metadata.

    Args:
    pca_yaml (str): path to pca.yaml

    Returns:
    DataProcessingParams: dataclass containing image filtering parameters
    MaskParams: dataclass containing mask parameters
    ProcessingConfig: dataclass containing processing configuration flags
    """
    if exists(pca_yaml):
        # Load pca metadata file
        pca_config = read_yaml(pca_yaml)

        use_fft = pca_config.get('use_fft', False)
        missing_data = pca_config.get('missing_data', False)
        if use_fft:
            print('Will use FFT...')
        if missing_data:
            print('Detected missing data...')

        # Create dataclass instances
        processing_config = ProcessingConfig(
            use_fft=use_fft,
            missing_data=missing_data
        )

        data_params = DataProcessingParams(
            min_height=pca_config['min_height'],
            max_height=pca_config['max_height'],
            gaussfilter_space=pca_config['gaussfilter_space'],
            tailfilter_size=pca_config['tailfilter_size'],
            medfilter_space=pca_config.get('medfilter_space'),
            medfilter_time=pca_config.get('medfilter_time'),
            gaussfilter_time=pca_config.get('gaussfilter_time', 0.0),
            tailfilter_shape=pca_config.get('tailfilter_shape', 'ellipse')
        )

        mask_params = MaskParams(
            mask_height_threshold=pca_config.get('mask_height_threshold', 5.0),
            mask_threshold=pca_config.get('mask_threshold', -16.0)
        )

        return data_params, mask_params, processing_config
    else:
        raise IOError(f'Could not find {pca_yaml}')
