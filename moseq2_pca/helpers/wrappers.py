"""
Wrapper functions for PCA.
"""

import h5py
import click
import logging
import datetime
import warnings
import traceback
import numpy as np
import dask.array as da
from pathlib import Path
from tqdm.auto import tqdm
from toolz import dissoc, keyfilter
from moseq2_pca.viz import plot_pca_results, changepoint_dist
from moseq2_pca.helpers.data import get_pca_paths, get_pca_yaml_data, load_pcs_for_cp
from moseq2_pca.helpers.parameters import MouseProcessingParams, SVDConfig, DaskConfig
from moseq2_pca.pca.util import (
    apply_pca_dask,
    apply_pca_local,
    train_pca_dask,
    get_changepoints_dask,
)
from moseq2_pca.util import (
    recursive_find_h5s,
    initialize_dask,
    set_dask_config,
    close_dask,
    h5_to_dict,
    check_timestamps,
    write_yaml,
)

def create_dataclass_from_dict(dataclass_type, data_dict):
    """
    Create a dataclass from a dictionary. Remove any keys that are not in the dataclass.
    Return the dataclass and filtered dictionary.
    """
    # get the fields of the dataclass
    fields = dataclass_type.__dataclass_fields__
    # filter the dictionary to only include the fields of the dataclass
    dataclass_out = dataclass_type(**keyfilter(lambda k: k in fields, data_dict))
    # filter the dictionary to only include the fields of the dataclass
    filtered_dict = dissoc(data_dict, *fields)

    return dataclass_out, filtered_dict


def load_and_check_data(input_dir, output_dir, config_data):
    """
    Load relevant h5 and yaml files found in given input directory, then check for timestamps and warn the user if they are missing.

    Args:
    input_dir (str): input directory containing extracted h5 files to find
    output_dir (str): directory name to save pca results

    Returns:
    output_dir (str): output directory path
    h5s (list): list of found h5 files
    yamls (list): list of corresponding yaml files
    dicts (list): list of corresponding metadata.json files
    """
    input_dir = Path(input_dir).resolve()
    # dynamically change the set_dask_config memory setting
    if config_data["cluster_type"] == "local":
        set_dask_config(
            memory={"target": 0.85, "spill": True, "pause": False, "terminate": False}
        )
    else:
        set_dask_config()

    # Set up output directory
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # find directories with .dat files tchat either have incomplete or no extractions
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    check_timestamps(h5s)  # function to check whether timestamp files are found

    return output_dir, h5s, dicts, yamls


def can_overwrite(config_data: dict, save_file: Path) -> bool:
    """
    Handle user input for overwriting PCA files.

    Args:
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    save_file (Path): path to save PCA file

    Returns:
    bool: True if user wants to overwrite, False otherwise
    """
    save_file = save_file.with_suffix(".h5")
    # check flags for overwriting in either the train or apply steps
    if (
        not config_data.get("overwrite_pca_train", False)
        or not config_data.get("overwrite_pca_apply", False)
    ) and save_file.exists():
        ow = input(
            f"The file {save_file} already exists.\nWould you like to overwrite it? [y -> yes, n -> no]: "
        )
        if ow.lower() != "y":
            return False
    return True


def train_pca_wrapper(input_dir, config_data, output_dir, output_file):
    """
    Wrapper function to train PCA.

    Args:
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename

    Returns:
    config_data (dict): updated config_data variable to write back in GUI API
    """

    # gather parameters for mouse processing
    mouse_proc_params, config_data = create_dataclass_from_dict(MouseProcessingParams, config_data)

    # gather parameters for SVD
    svd_config, config_data = create_dataclass_from_dict(SVDConfig, config_data)

    # replace initialize_dask config_data with dask_config
    dask_config, config_data = create_dataclass_from_dict(DaskConfig, config_data)

    # Get training data
    output_dir, h5s, dicts, yamls = load_and_check_data(
        input_dir, output_dir, config_data
    )

    # Setting path to PCA config file
    save_file = output_dir / output_file

    # Edge Case: Handling pre-existing PCA file
    if not can_overwrite(config_data, save_file):
        return config_data

    logging.basicConfig(filename=str(output_dir / "train.log"), level=logging.ERROR)

    # Load all open h5 file references
    h5ps = [h5py.File(h5, mode="r") for h5 in h5s]

    # Subset extracted frames, then read them into chunked Dask arrays
    arrays = []
    for fp in tqdm(h5ps):
        temp_extracted = fp[config_data["h5_path"]]
        num_frames = int(len(temp_extracted) * config_data.get("train_on_subset", 1))
        arrays.append(
            da.from_array(temp_extracted, chunks=config_data["chunk_size"])[
                np.sort(
                    np.random.choice(len(temp_extracted), num_frames, replace=False)
                )
            ]
        )

    # To extracted frames, then read them into chunked Dask arrays
    stacked_array = da.concatenate(arrays, axis=0)

    # Filter out depth value extreme values; Generally same values used during extraction
    stacked_array = da.where(
        da.logical_or(
            stacked_array < mouse_proc_params.min_height,
            stacked_array > mouse_proc_params.max_height,
        ),
        0,
        stacked_array,
    )

    data_size = stacked_array.nbytes

    # Initialize Dask client
    client, cluster, workers = initialize_dask(dask_config, data_size=data_size)

    click.echo(f"Processing {len(stacked_array)} total frames")

    # Optionally read corresponding frame masks if for recording sessions that contain inscopix,
    # photometry, or ephys cables. These sessions in particular include frame-by-frame masks
    # to explicitly tell PCA where the mouse is, removing any noise or obstructions.
    # Note: timestamps for all files are required in order for this operation to work.
    if svd_config.missing_data or config_data.get("cable_filter_iters", 0) >= 1:
        svd_config.missing_data = True  # in case cable filter iterations >= 1
        mask_dsets = [h[config_data["h5_mask_path"]] for h in h5ps]
        mask_arrays = [
            da.from_array(dset, chunks=config_data["chunk_size"]) for dset in mask_dsets
        ]
        stacked_array_mask = da.concatenate(mask_arrays, axis=0).astype("float32")
        stacked_array_mask = da.logical_and(
            stacked_array_mask < config_data["mask_threshold"],
            stacked_array > config_data["mask_height_threshold"],
        )
        click.echo("Loaded mask for missing data")

    else:
        stacked_array_mask = None

    params = config_data
    params["start_time"] = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
    params["inputs"] = h5s

    # TODO: modify this to save the h5 file paths in another place
    # Update PCA config yaml file
    config_store = save_file.with_suffix('.yaml')
    write_yaml(config_store, params)

    # Compute Principal Components
    try:
        output_dict = train_pca_dask(
            dask_array=stacked_array,
            mask=stacked_array_mask,
            mouse_proc_params=mouse_proc_params,
            dask_config=dask_config,
            svd_config=svd_config,
            client=client,
        )
    except Exception as e:
        logging.error(e)
        logging.error(e.__traceback__)
        click.echo(
            "Training interrupted. Closing Dask Client. You may find logs of the error here:"
        )
        click.echo("---- ", output_dir / "train.log")
    finally:
        # After Success or failure: Shutting down Dask client and clearing any residual data
        close_dask(client, cluster, config_data["timeout"])

        # close all open h5 files
        [fp.close() for fp in h5ps]

    try:
        # Plotting training results
        plot_pca_results(output_dict, save_file, output_dir)

        # Saving PCA to h5 file
        with h5py.File(save_file.with_suffix('.h5'), "w") as f:
            for k, v in output_dict.items():
                f.create_dataset(k, data=v, compression="gzip", dtype="float32")

        config_data["pca_file"] = str(save_file.with_suffix('.h5'))
    except Exception as e:
        click.echo(f"Could not save PCA since the training was interrupted: {e}")
        pass

    return config_data


def apply_pca_wrapper(input_dir, config_data, output_dir, output_file):
    """
    Wrapper function to obtain PCA Scores.

    Args:
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename

    Returns:
    config_data (dict): updated config_data variable to write back in GUI API
    success (bool): flag to indicate whether the PCA scores were computed successfully
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # gather parameters for dask
    dask_config, config_data = create_dataclass_from_dict(DaskConfig, config_data)

    # Set up data
    output_dir, h5s, dicts, yamls = load_and_check_data(
        input_dir, output_dir, config_data
    )

    # Set path to PCA Scores file
    save_file = output_dir / output_file

    # Handling pre-existing PCA file
    # no intended pca overwrite
    if not can_overwrite(config_data, save_file):
        return config_data, False

    # Get path to trained PCA file to load PCs from
    config_data, pca_file, pca_file_scores = get_pca_paths(config_data, output_dir)

    print("Loading PCs from", pca_file)
    with h5py.File(config_data["pca_file"], "r") as f:
        pca_components = f[config_data["pca_path"]][()]

    # Get the yaml for pca, check parameters
    pca_yaml = Path(pca_file).with_suffix('.yaml')

    # Get filtering parameters and optional PCA reconstruction parameters (if missing_data == True)
    # TODO: figure this out - I think it merges extraction config with this config
    # use_fft, clean_params, mask_params, missing_data = get_pca_yaml_data(pca_yaml)

    with warnings.catch_warnings():
        # Compute PCA Scores locally (without dask)
        if config_data["cluster_type"] == "nodask":
            apply_pca_local(
                pca_components=pca_components,
                h5s=h5s,
                yamls=yamls,
                clean_params=clean_params,
                save_file=save_file,
                chunk_size=config_data["chunk_size"],
                mask_params=mask_params,
                fps=config_data["fps"],
                missing_data=missing_data,
                h5_path=config_data["h5_path"],
                h5_mask_path=config_data["h5_mask_path"],
                verbose=config_data["verbose"],
            )

        else:
            # Initialize Dask client
            client, cluster, workers = initialize_dask(dask_config)

            logging.basicConfig(
                filename=f"{output_dir}/scores.log", level=logging.ERROR
            )

            # Compute PCA Scores
            try:
                apply_pca_dask(
                    pca_components=pca_components,
                    h5s=h5s,
                    yamls=yamls,
                    clean_params=clean_params,
                    save_file=save_file,
                    chunk_size=config_data["chunk_size"],
                    fps=config_data["fps"],
                    client=client,
                    missing_data=missing_data,
                    mask_params=mask_params,
                    h5_path=config_data["h5_path"],
                    h5_mask_path=config_data["h5_mask_path"],
                    verbose=config_data["verbose"],
                )
            except Exception as e:
                # Clearing all data from Dask client in case of interrupted PCA
                traceback.print_exc()
                click.echo("Operation interrupted. Closing Dask Client.")
            finally:
                # After Success or failure: Shutting down Dask client and clearing any residual data
                close_dask(client, cluster, config_data["timeout"])

    config_data["pca_file_scores"] = str(save_file.with_suffix('.h5'))
    return config_data, True


def compute_changepoints_wrapper(input_dir, config_data, output_dir, output_file):
    """
    Wrapper function to compute model-free Changepoints.

    Args:
    input_dir (int): path to directory containing all h5+yaml files
    config_data (dict): dict of relevant PCA parameters (image filtering etc.)
    output_dir (str): path to directory to store PCA data
    output_file (str): pca model filename

    Returns:
    config_data (dict): updated config_data variable to write back in GUI API
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    dask_config, config_data = create_dataclass_from_dict(DaskConfig, config_data)

    # Get loaded h5s and yamls
    output_dir, h5s, dicts, yamls = load_and_check_data(
        input_dir, output_dir, config_data
    )

    # Set path to changepoints
    save_file = (Path(output_dir) / output_file).with_suffix('.h5')

    # Get paths to PCA, PCA Scores file
    config_data, pca_file, pca_file_scores = get_pca_paths(config_data, output_dir)

    # Load Principal components, set up changepoint parameter dict, and optionally load reconstructed PCs.
    pca_components, changepoint_params, missing_data, mask_params = load_pcs_for_cp(
        pca_file, config_data
    )

    # Initialize Dask client
    client, cluster, workers = initialize_dask(dask_config)

    # Compute Changepoints
    try:
        get_changepoints_dask(
            pca_components=pca_components,
            pca_scores=pca_file_scores,
            h5s=h5s,
            yamls=yamls,
            changepoint_params=changepoint_params,
            save_file=save_file,
            chunk_size=config_data["chunk_size"],
            fps=config_data["fps"],
            client=client,
            missing_data=missing_data,
            mask_params=mask_params,
            h5_path=config_data["h5_path"],
            h5_mask_path=config_data["h5_mask_path"],
            verbose=config_data["verbose"],
            n_rps=config_data["dims"],
        )
    except Exception as e:
        print(e)
        click.echo("Operation interrupted. Closing Dask Client.")
    finally:
        # After Success: Shutting down Dask client and clearing any residual data
        close_dask(client, cluster, config_data["timeout"])

    # Read Changepoints from saved file
    with h5py.File(save_file, "r") as f:
        cps = h5_to_dict(f, "cps")

    # add change point path to config file
    config_data["changepoint_file"] = str(save_file)
    # Plot and save Changepoint PDF histogram
    block_durs = np.concatenate([np.diff(cp, axis=0) for k, cp in cps.items()])
    out = changepoint_dist(block_durs, headless=True)
    if out:
        fig_path = save_file.with_name(save_file.stem + "_dist")
        fig, _ = out
        fig.savefig(f"{fig_path}.png")
        fig.savefig(f"{fig_path}.pdf")
        fig.close("all")

    return config_data
