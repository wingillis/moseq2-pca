"""
Utility and helper functions for finding and reading files, filtering operations, Dask initialization, and changepoint helper functions.
"""

import os
import cv2
import h5py
import time
import dask
import click
import psutil
import warnings
import platform
import subprocess
import numpy as np
import scipy.signal
from glob import glob
from copy import deepcopy
from ruamel.yaml import YAML
from tqdm.auto import tqdm
from functools import partial
from dask.distributed import Client
from toolz import dissoc, merge
from dask_jobqueue import SLURMCluster
from moseq2_pca.helpers.parameters import MouseProcessingParams, DaskConfig
from os.path import join, exists, abspath, expanduser


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    """
    Recursively find h5 files, along with yaml files with the same basename

    Args:
    root_dir (str): path to base directory to begin recursive search in.
    ext (str): extension to search for
    yaml_string (str): string for filename formatting when saving data

    Returns:
    h5s (list): list of found h5 files
    dicts (list): list of found metadata files
    yamls (list): list of found yaml files
    """

    if not ext.startswith('.'):
        ext = '.' + ext

    def has_frames(f):
        try:
            with h5py.File(f, 'r') as h5f:
                return 'frames' in h5f
        except OSError:
            warnings.warn(f'Error reading {f}, skipping...')
            return False

    h5s = glob(join(abspath(root_dir), '**', f'*{ext}'), recursive=True)
    h5s = filter(lambda f: exists(yaml_string.format(f.replace(ext, ''))), h5s)
    h5s = list(filter(has_frames, h5s))
    yamls = list(map(lambda f: yaml_string.format(f.replace(ext, '')), h5s))
    dicts = list(map(read_yaml, yamls))

    return h5s, dicts, yamls


def gauss_smooth(signal, win_length=None, sig=1.5, kernel=None):
    """
    Perform Gaussian Smoothing on a 1D signal.

    Args:
    signal (1d numpy array): signal to perform smoothing
    win_length (int): window size for gaussian kernel filter
    sig (float): variance of 1d gaussian kernel.
    kernel (tuple): kernel size to use for smoothing

    Returns:
    result (1d numpy array): smoothed signal
    """
    if kernel is None:
        kernel = gaussian_kernel1d(n=win_length, sig=sig)

    result = scipy.signal.convolve(signal, kernel, mode='same', method='direct')

    return result


def gaussian_kernel1d(n=None, sig=3):
    """
    Get 1D gaussian kernel.

    Args:
    n (int): window size.
    sig (int): variance of kernel to use.

    Returns:
    kernel (1d array): 1D numpy kernel.
    """

    if n is None:
        n = np.ceil(sig * 4)

    points = np.arange(-n, n)

    kernel = np.exp(-(points**2.0) / (2.0 * sig**2.0))
    kernel /= np.sum(kernel)

    return kernel


def clean_frames(frames: np.ndarray, mouse_proc_params: MouseProcessingParams, detrend_time: int = None):
    """
    Filter spatial/temporal noise from frames using Median and Gaussian filters,
    given kernel sizes for each respective requested filter.

    Args:
    frames (numpy.ndarray): frames to filter.
    mouse_proc_params (MouseProcessingParams): parameters for mouse processing.
    detrend_time (int): number of frames to use for estimating a trend.

    Returns:
    out (numpy.ndarray): filtered frames.
    """

    out = np.copy(frames)

    if mouse_proc_params.tailfilter is not None:
        for i in range(frames.shape[0]):
            mask = (
                cv2.morphologyEx(out[i], cv2.MORPH_OPEN, mouse_proc_params.tailfilter)
                > mouse_proc_params.tail_threshold
            )
            out[i] = out[i] * mask.astype(frames.dtype)

    if mouse_proc_params.medfilter_space is not None and np.all(
        np.array(mouse_proc_params.medfilter_space) > 0
    ):
        for i in range(frames.shape[0]):
            for medfilt in mouse_proc_params.medfilter_space:
                out[i] = cv2.medianBlur(out[i], medfilt)

    if mouse_proc_params.gaussfilter_space is not None and np.all(
        np.array(mouse_proc_params.gaussfilter_space) > 0
    ):
        for i in range(frames.shape[0]):
            out[i] = cv2.GaussianBlur(
                out[i],
                (21, 21),
                *mouse_proc_params.gaussfilter_space,
            )

    if mouse_proc_params.medfilter_time is not None and np.all(
        np.array(mouse_proc_params.medfilter_time) > 0
    ):
        for idx, i in np.ndenumerate(frames[0]):
            for medfilt in mouse_proc_params.medfilter_time:
                out[:, idx[0], idx[1]] = scipy.signal.medfilt(out[:, idx[0], idx[1]], medfilt)

    if mouse_proc_params.gaussfilter_time is not None and mouse_proc_params.gaussfilter_time > 0:
        kernel = gaussian_kernel1d(sig=mouse_proc_params.gaussfilter_time)
        for idx, i in np.ndenumerate(frames[0]):
            out[:, idx[0], idx[1]] = np.convolve(out[:, idx[0], idx[1]], kernel, mode="same")

    if detrend_time is not None and detrend_time > 0:
        kernel = gaussian_kernel1d(sig=detrend_time)
        for idx, i in np.ndenumerate(frames[0]):
            out[:, idx[0], idx[1]] = out[:, idx[0], idx[1]] - gauss_smooth(
                out[:, idx[0], idx[1]], kernel=kernel
            )

    return out


def insert_nans(timestamps, data, fps=30):
    """
    Fill NaN values with 0 in given 1D timestamps array. Used to handle dropped frames from the video acquisition.

    Args:
    timestamps (numpy.array): timestamp values
    data (numpy.array): additional data to fill with NaN values - can be PC scores
    fps (int): frames per second

    Returns:
    filled_data (numpy.array): filled missing timestamp values.
    data_idx (numpy.array): indices of inserted 0s
    filled_timestamps (numpy.array): filled timestamp-strs
    """

    df_timestamps = np.diff(np.insert(timestamps, 0, timestamps[0] - 1.0 / fps))
    missing_frames = np.floor(df_timestamps / (1.0 / fps))

    fill_idx = np.where(missing_frames > 1)[0]
    data_idx = np.arange(len(timestamps)).astype('float64')

    filled_data = deepcopy(data)
    filled_timestamps = deepcopy(timestamps)

    if filled_data.ndim == 1:
        isvec = True
        filled_data = filled_data[:, None]
    else:
        isvec = False

    _, nfeatures = filled_data.shape

    for idx in fill_idx[::-1]:
        if idx < len(missing_frames): # ensures ninserts value remains an int
            ninserts = int(missing_frames[idx] - 1)
            data_idx = np.insert(data_idx, idx, [np.nan] * ninserts)
            insert_timestamps = timestamps[idx - 1] + \
                np.cumsum(np.ones(ninserts,) * 1.0 / fps)
            filled_data = np.insert(filled_data, idx,
                                    np.ones((ninserts, nfeatures)) * np.nan, axis=0)
            filled_timestamps = np.insert(
                filled_timestamps, idx, insert_timestamps)

    if isvec:
        filled_data = np.squeeze(filled_data)

    return filled_data, data_idx, filled_timestamps


def read_yaml(yaml_file):
    """
    Read yaml file and return dictionary representation of file contents.

    Args:
    yaml_file (str): path to yaml file

    Returns:
    return_dict (dict): dict of yaml file contents
    """
    yaml = YAML(typ='safe')

    try:
        with open(yaml_file, 'r') as f:
            return_dict = yaml.load(f)
    except IOError:
        click.echo(f'Error reading {yaml_file}. Returning empty dict.')
        return_dict = {}

    return return_dict


def check_timestamps(h5s):
    """
    Helper function to determine whether timestamps and/or metadata is missing from
    extracted files. Function will emit a warning if either pieces of data are missing.

    Args:
    h5s (list): List of paths to all extracted h5 files.
    """

    for h5 in h5s:
        missing_data = []
        
        try:
            get_timestamp_path(h5)
        except KeyError:
            missing_data.append('timestamps')
        except Exception as e:
            warnings.warn(f'Error loading timestamps from {h5}: {str(e)}')
            missing_data.append('timestamps')

        try:
            get_metadata_path(h5)
        except KeyError:
            missing_data.append('metadata')
        except Exception as e:
            warnings.warn(f'Error loading metadata from {h5}: {str(e)}')
            missing_data.append('metadata')

        if missing_data:
            warnings.warn(f'Could not locate {", ".join(missing_data)} in {h5}. '
                        'This may cause issues if PCA has been trained on missing data.')


def get_timestamp_path(h5file):
    """
    Return path within h5 file that contains the kinect timestamps

    Args:
    h5file (str): path to h5 file.

    Returns:
    (str): path to metadata timestamps within h5 file
    """

    with h5py.File(h5file, 'r') as f:
        if '/timestamps' in f:
            return '/timestamps'
        elif '/metadata/timestamps' in f:
            return '/metadata/timestamps'
        else:
            raise KeyError('timestamp key not found')


def get_metadata_path(h5file):
    """
    Return path within h5 file that contains the kinect extraction metadata.

    Args:
    h5file (str): path to h5 file.

    Returns:
    (str): path to acquistion metadata within h5 file.
    """

    with h5py.File(h5file, 'r') as f:
        if '/metadata/acquisition' in f:
            return '/metadata/acquisition'
        elif '/metadata/extraction' in f:
            return '/metadata/extraction'
        else:
            raise KeyError('acquisition metadata not found')


def h5_to_dict(h5file: str | h5py.File, path: str) -> dict:
    """
    Read all contents from h5 and returns them in a nested dict object.

    Args:
    h5file (str | h5py.File): path to h5 file
    path (str): path to group within h5 file

    Returns:
    ans (dict): dictionary of all h5 group contents
    """

    ans = {}

    if isinstance(h5file, str):
        with h5py.File(h5file, 'r') as f:
            ans = h5_to_dict(f, path)
            return ans

    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py.Group):
            ans[key] = h5_to_dict(h5file, path + key + '/')
    return ans


def set_dask_config(memory: dict = {'target': 0.85, 'spill': False, 'pause': False, 'terminate': 0.95}):
    """
    Set initial dask configuration parameters

    Args:
    memory (dict): dictionary containing default dask configuration variables to ensure safe amount of resource usage.
    """

    memory = {f'distributed.worker.memory.{k}': v for k, v in memory.items()}
    dask.config.set(memory)
    dask.config.set({'optimization.fuse.ave-width': 5})


def get_env_cpu_and_mem():
    """
    Read current system environment and return the amount of available memory and CPUs to allocate to the created cluster.

    Returns:
    mem (float): Optimal number of memory (in bytes) to allocate to initialized dask cluster
    cpu (int): Optimal number of CPUs to allocate to dask
    """

    if is_slurm := os.environ.get('SLURM_JOBID', False):
        click.echo('Detected slurm environment, using "sacct" to detect cpu and memory requirements')
        cmd = f'sacct -j {is_slurm} --format AllocCPUS,ReqMem -X -n -p'
        output = subprocess.check_output(cmd.split(' '))
        output = output.decode('utf-8').strip().split('|')
        cpu, mem, _ = output
        cpu = max(1, int(cpu)-1)

        if 'G' in mem:
            # account for additional processes that needs memory
            mem = float(mem[:mem.index('G')]) * 1e9 * 0.8
        elif 'M' in mem:
            # account for additional processes that needs memory
            mem = float(mem[:mem.index('M')]) * 1e6 * 0.8
    else:
        mem = psutil.virtual_memory().available * 0.8
        cpu = max(1, psutil.cpu_count() - 1)

    return mem, cpu


def calculate_worker_resources(dask_config: DaskConfig, data_size: float = None) -> tuple[int, float]:
    """
    Calculate optimal number of workers and memory limits for dask cluster.

    Args:
        dask_config (DaskConfig): Configuration object containing worker settings
        data_size (float, optional): Size of dataset in bytes

    Returns:
        tuple: (nworkers, mem_limit) where nworkers is the number of workers to use
               and mem_limit is the memory limit per worker in bytes
    """
    max_mem, max_cpu = get_env_cpu_and_mem()
    overhead = 0.8e9  # memory overhead for each worker; approximate
    
    # allocating 0.4 of the maximum memory to account for overhead per worker
    allowed = max_mem * 0.4 
    max_workers = allowed // overhead

    # set number of workers to maximum workers, or total number of CPUs
    if dask_config.nworkers > max_workers:
        click.echo(f'Reducing number of workers to {min(max_workers, max_cpu)} to account for worker base memory and the number of CPUs')
    nworkers = int(min(max(1, dask_config.nworkers), max_workers, max_cpu))

    # compute mem limit per worker
    try:
        mem_limit = max(1, max_mem / dask_config.nworkers)
    except:
        mem_limit = 1

    # display some diagnostic info
    if data_size is not None:
        click.echo(f'Dataset size: {data_size / 1e9:.2f}GB')

    click.echo(f'Setting number of workers to: {nworkers}')
    click.echo(f'Overriding memory per worker to {mem_limit / 1e9:.2f}GB')

    return nworkers, mem_limit


def initialize_dask(dask_config: DaskConfig, data_size: float = None):
    """
    Initialize dask client, cluster, workers, etc.

    Args:
    nworkers (int): number of dask workers to initialize
    processes (int): number of processes per worker
    memory (str): amount of memory to allocate to dask cluster
    cores (int): number of cores to use.
    wall_time (str): amount of time to allow program to run
    queue (str): logging mode
    local_processes (bool): flag to use processes or threads when using a local cluster
    cluster_type (str): indicate what cluster to use (local or slurm)
    timeout (int): how many minutes to wait for workers to initialize
    cache_path (str or Pathlike): path to store cached data
    dashboard_port (str): port number to find dask statistics
    data_size (float): size of the dask array in number of bytes.
    kwargs: extra keyward arguments

    Returns:
    client (dask Client): initialized Client
    cluster (dask Cluster): initialized Cluster
    workers (dask Workers): intialized workers
    """

    click.echo(f'Access dask dashboard at http://localhost:{dask_config.dashboard_port}')

    if dask_config.cluster_type == 'local':
        nworkers, mem_limit = calculate_worker_resources(dask_config, data_size)

        client = Client(processes=dask_config.local_processes,
                        threads_per_worker=1,
                        memory_limit=mem_limit,
                        n_workers=nworkers,
                        dashboard_address=dask_config.dashboard_port,
                        local_directory=dask_config.cache_path)
        cluster = client.cluster

    elif dask_config.cluster_type == 'slurm':

        cluster = SLURMCluster(processes=dask_config.processes,
                               n_workers=dask_config.nworkers,
                               cores=dask_config.cores,
                               memory=dask_config.memory,
                               queue=dask_config.queue,
                               walltime=dask_config.wall_time,
                               local_directory=dask_config.cache_path,
                               scheduler_options={'dashboard_address': dask_config.dashboard_port})
        client = Client(cluster)
    else:
        raise NotImplementedError('Specified cluster not supported. Supported types are: "slurm", "local"')

    if client is not None:

        client_info = client.scheduler_info()
        if 'services' in client_info.keys() and 'bokeh' in client_info['services'].keys():
            ip = client_info['address'].split('://')[1].split(':')[0]
            port = client_info['services']['bokeh']
            hostname = platform.node()
            click.echo(f'Web UI served at {ip}:{port} (if port forwarding use internal IP not localhost)')
            click.echo(f'Tunnel command:\n ssh -NL {port}:{ip}:{port} {hostname}')
            click.echo(f'Tunnel command (gcloud):\n gcloud compute ssh {hostname} -- -NL {port}:{ip}:{port}')

    if dask_config.cluster_type == 'slurm':

        active_workers = len(client.scheduler_info()['workers'])
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pbar = tqdm(total=nworkers, desc="Intializing workers")

            elapsed_time = (time.time() - start_time) / 60

            while active_workers < nworkers and elapsed_time < dask_config.timeout:
                tmp = len(client.scheduler_info()['workers'])
                if tmp - active_workers > 0:
                    pbar.update(tmp - active_workers)
                active_workers = tmp
                time.sleep(1)
                elapsed_time = (time.time() - start_time) / 60

            pbar.close()

    workers = cluster.workers

    return client, cluster, workers


def close_dask(client, cluster, timeout):
    """
    Shut down the Dask client and cluster, and dump all cache data.

    Args:
    client (Dask Client): Client object
    cluster (dask Cluster): initialized Cluster
    timeout (int): Time to wait for client to close gracefully (minutes)

    Returns:
    """

    if client is not None:
        try:
            client.close(timeout=timeout)
            cluster.close(timeout=timeout)
        except Exception as e:
            print('Error:', e)
            print('Could not shutdown dask client')


def get_rps(frames, rps: int= 600, normalize: bool = True):
    """
    Get random projections of frames.

    Args:
    frames (numpy.array): Frames to get dimensions from.
    rps (int): Number of random projections.
    normalize (bool): indicates whether to normalize the random projections.

    Returns:
    rproj (2D or 3D numpy array): Computed random projections with same shape as frames
    """
    rng = np.random.default_rng(0)

    if frames.ndim == 3:
        frames = frames.reshape(len(frames), -1)

    rproj = frames.dot(rng.standard_normal((frames.shape[1], rps), dtype='float32'))

    if normalize:
        rproj = scipy.stats.zscore(scipy.stats.zscore(rproj).T)

    return rproj


def get_changepoints(scores, k=5, sigma=3, peak_height=.5, peak_neighbors=1,
                     baseline=True, timestamps=None):
    """
    Compute changepoints and its corresponding distribution. Changepoints describe
    the magnitude of frame-to-frame changes of mouse pose.

    Args:
    scores (numpy.ndarray): nframes * rows * columns
    k (int): klags - Lag to use for derivative calculation.
    sigma (int): Standard deviation of gaussian smoothing filter.
    peak_height (float): user-defined peak Changepoint length.
    peak_neighbors (int): number of peaks in the CP curve.
    baseline (bool): normalize data.
    timestamps (numpy.array): loaded timestamps.

    Returns:
    cps (numpy.ndarray): array of changepoint values
    normed_df (numpy.array): array of values for bar plot
    """

    k = int(k)
    peak_neighbors = int(peak_neighbors)

    nanidx = np.isnan(scores)
    smooth_scores = np.nan_to_num(scores)

    if sigma is not None and sigma > 0:
        smooth = partial(gauss_smooth, sig=sigma)
        smooth_scores = np.apply_along_axis(smooth, 1, smooth_scores)

    smooth_scores[:, k // 2:-k // 2] = np.square(smooth_scores[:, k:] - smooth_scores[:, :-k])
    smooth_scores[nanidx] = np.nan

    if sigma is not None and sigma > 0:
        smooth_scores[:, :int(6 * sigma)] = np.nan
        smooth_scores[:, -int(6 * sigma):] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        smooth_scores = np.nanmean(smooth_scores, axis=0)

        if baseline: smooth_scores -= np.nanmin(smooth_scores)

        if timestamps is not None:
            smooth_scores, _, _ = insert_nans(
                timestamps, smooth_scores, fps=np.round(1 / np.median(np.diff(timestamps))).astype('int'))

        smooth_scores = np.squeeze(smooth_scores)
        cps = scipy.signal.argrelextrema(
            smooth_scores, np.greater, order=peak_neighbors)[0]
        cps = cps[np.argwhere(smooth_scores[cps] > peak_height)]

    return cps, smooth_scores


def combine_new_config(config_file, config_data):
    """
    Read config file and combine new config params with it

    Args:
        config_file (str): path to config.yaml
        config_data (dict): dictionary of config data
    """
    # open the config file
    temp_config = read_yaml(config_file)
    # combining config data with the existing config file
    config_data = merge(temp_config, config_data)
    # ensure output_file and output_dir are not in config_data or reusing config for extraction will fail
    config_data = dissoc(config_data, 'output_dir', 'output_file')

    yaml = YAML(typ='safe')
    with open(config_file, 'w') as f:
        yaml.safe_dump(config_data, f)
