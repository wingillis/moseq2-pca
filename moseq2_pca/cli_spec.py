import click
from pathlib import Path

COMMON_PCA_OPTIONS = [
    (
        ['--input-dir', '-i'],
        {'type': click.Path(), 'default': Path.cwd(), 'help': 'Directory to find extracted h5 files'}
    ),
    (
        ['--output-dir', '-o'],
        {'type': click.Path(), 'default': Path.cwd() / '_pca', 'help': 'Directory to store PCA results'}
    ),
    (
        ['--config-file'],
        {'type': click.Path(), 'help': "Path to configuration file"}
    ),
    (
        ['--h5-path'],
        {'default': '/frames', 'type': str, 'help': 'Path to data in h5 files'}
    ),
    (
        ['--h5-mask-path'],
        {'default': '/frames_mask', 'type': str, 'help': "Path to log-likelihood mask in h5 files"}
    ),
    (
        ['--chunk-size'],
        {'default': 4000, 'type': int, 'help': 'Number of frames per chunk'}
    )
]

DASK_PARAMETERS = [
    (
        ['-d', '--dask-cache-path'],
        {'type': click.Path(), 'default': Path.cwd() / '_pca', 'help': 'Path to spill data to disk for dask'}
    ),
    (
        ['--dask-port'],
        {'default': '8787', 'type': str, 'help': "Port to access dask dashboard"}
    ),
    (
        ['-q', '--queue'],
        {'default': 'debug', 'type': str, 'help': "Cluster queue/partition for submitting jobs"}
    ),
    (
        ['-n', '--nworkers'],
        {'default': 1, 'type': int, 'help': "Number of workers"}
    ),
    (
        ['-c', '--cores'],
        {'default': 1, 'type': int, 'help': "Number of cores per worker"}
    ),
    (
        ['-p', '--processes'],
        {'default': 1, 'type': int, 'help': "Number of processes to run on each worker"}
    ),
    (
        ['-m', '--memory'],
        {'default': "15GB", 'type': str, 'help': "Total RAM usage per worker"}
    ),
    (
        ['-w', '--wall-time'],
        {'default': "06:00:00", 'type': str, 'help': "Wall time (compute time) for workers"}
    ),
    (
        ['--timeout'],
        {'default': 5, 'type': float, 'help': "Time to wait for workers to initialize before proceeding (minutes)"}
    )
]

PCA_TRAIN_OPTIONS = [
    (
        ['--gaussfilter-space'],
        {'default': (1.5, 1), 'type': (float, float), 'help': "x, y sigma for kernel in Spatial filter for data (Gaussian)"}
    ),
    (
        ['--gaussfilter-time'],
        {'default': 0, 'type': float, 'help': "sigma for temporal filter for data (Gaussian)"}
    ),
    (
        ['--medfilter-space'],
        {'default': [0], 'type': int, 'help': "kernel size for median spatial filter", 'multiple': True}
    ),
    (
        ['--medfilter-time'],
        {'default': [0], 'type': int, 'help': "kernel size for median temporal filter", 'multiple': True}
    ),
    (
        ['--missing-data'],
        {'is_flag': True, 'help': "Use missing data PCA; will be automatically set to True if cable-filter-iters > 1 from the extract step."}
    ),
    (
        ["--missing-data-iters"],
        {'default': 10, 'type': int, 'help': "number of missing data PCA iterations"}
    ),
    (
        ['--mask-threshold'],
        {'default': -16, 'type': float, 'help': "Threshold for mask (missing data PCA only)"}
    ),
    (
        ['--mask-height-threshold'],
        {'default': 5, 'type': float, 'help': "Threshold for mask based on floor height"}
    ),
    (
        ['--min-height'],
        {'default': 10, 'type': int, 'help': 'Min mouse height from floor (mm)'}
    ),
    (
        ['--max-height'],
        {'default': 120, 'type': int, 'help': 'Max mouse height from floor (mm)'}
    ),
    (
        ['--tailfilter-size'],
        {'default': (9, 9), 'type': (int, int), 'help': 'Tail filter size'}
    ),
    (
        ['--tailfilter-shape'],
        {'default': 'ellipse', 'type': str, 'help': 'Tail filter shape'}
    ),
    (
        ['--use-fft'],
        {'is_flag': True, 'help': 'Use 2D fft'}
    ),
    (
        ['--train-on-subset'],
        {'default': 1, 'type': float, 'help': "The fraction of the total frames the PCA is trained on; default PCA is trained on all frames"}
    ),
    (
        ['--recon-pcs'],
        {'default': 10, 'type': int, 'help': 'Number of PCs to use for missing data reconstruction'}
    ),
    (
        ["--rank"],
        {'default': 25, 'type': int, 'help': 'Rank for compressed SVD'}
    ),
    
    
]

def option_spec(spec_list: list[tuple]):
    """Build a decorator that applies all click.option entries from the given list."""
    
    def decorator(fn):
        for args, kwargs in reversed(spec_list):
            fn = click.option(*args, **kwargs)(fn)
        return fn
    
    return decorator
