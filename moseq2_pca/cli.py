"""
CLI for PCA and model-free changepoint analysis.
"""

import click
from moseq2_pca.util import combine_new_config, read_yaml
from moseq2_pca.helpers.wrappers import (train_pca_wrapper, apply_pca_wrapper,
                                         compute_changepoints_wrapper)
from moseq2_pca.cli_spec import (COMMON_PCA_OPTIONS, DASK_PARAMETERS, 
                               PCA_TRAIN_OPTIONS, option_spec)
from pathlib import Path


common_pca_options = option_spec(COMMON_PCA_OPTIONS)
common_dask_parameters = option_spec(DASK_PARAMETERS)
common_pca_train_options = option_spec(PCA_TRAIN_OPTIONS)


def load_config(ctx, param, value):
    """Callback to load configuration from a YAML file and set defaults."""
    if not value or not Path(value).exists():
        return None  # No config file specified or found

    try:
        config = read_yaml(value)
        # Extract the [pca] section if it exists
        pca_config = config.get("pca", {})
        if not isinstance(pca_config, dict):
            raise click.BadParameter("Config [pca] section must be a dictionary.")

        # Set the default map for the context if it doesn't exist
        ctx.default_map = ctx.default_map or {}
        # Add default map to each subcommand
        _maps = {}
        for command in ctx.command.commands.values():
            _maps[command.name] = pca_config
        ctx.default_map.update(_maps)

    except Exception as e:
        raise click.BadParameter(f"Error parsing config file {value}: {e}")

    return value  # Return the path itself


@click.group(context_settings=dict(show_default=True, default_map={}))
@click.version_option()
@click.option(
    "--config-file",
    type=click.Path(dir_okay=False),
    help="Path to a YAML configuration file. Options defined here are overridden by CLI arguments.",
    callback=load_config,
    is_eager=True,
)
@click.pass_context
def cli(ctx, config_file):
    """MoSeq2 PCA: PCA and model-free changepoint analysis."""
    ctx.ensure_object(dict).update({"config_path": config_file})

@cli.command(
    name="generate-config",
    help="Generates a configuration file (config.yaml) that holds editable options for pca parameters.",
)
@click.option("--output-file", "-o", type=click.Path(), default="pca-config.yaml")
@click.option(
    "--camera-type",
    default="k2",
    type=click.Choice(["k2", "azure"]),
    help="specify the camera type (k2 or azure), default is k2",
)
def generate_config(output_file, camera_type):
    """Copy default config and patch selected fields via sed to keep comments/structure."""
    import shutil

    script_path = Path(__file__).parent
    default_path = script_path / "default-config.yaml"
    shutil.copy(default_path, output_file)

    if camera_type == "azure":
        replacements = [
            ("gaussfilter_space", "[2.25, 1.5]"),
            ("tail_filter_size", "[15, 15]"),
            ("camera_type", '"azure"'),
        ]

        with open(output_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            for key, val in replacements:
                if key in line:
                    line = f"  {key}: {val}\n"
                    break
            new_lines.append(line)

        with open(output_file, "w") as f:
            f.writelines(new_lines)

    print(f"Successfully generated config file at {output_file}.")

@cli.command(name='train-pca', help='Train PCA on all extracted results (h5 files) in input directory')
@common_pca_options
@common_dask_parameters
@common_pca_train_options
@click.option('--output-file', default='pca', type=str, help='Name of h5 file for storing pca results')
@click.option('--local-processes', default=False, type=bool, help='Used with a local cluster. If True: use processes, If False: use threads')
@click.option('--overwrite-pca-train', default=False, type=bool, help='Used to bypass the pca overwrite question. If True: skip question, run automatically')
@click.option('--camera-type', default='k2', type=str, help='specify the camera type (k2 or azure), default is k2')
@click.pass_obj
def train_pca(ctx_obj, input_dir, output_dir, output_file, **cli_args):
    # function writes output pca path to config_data
    if cli_args.get('camera_type') == 'azure':
        # check if parameters are set to k2 default, change to azure default
        click.echo('Updating parameters for Azure Kinect camera...')
        if cli_args['gaussfilter_space'] == (1.5, 1):
            cli_args['gaussfilter_space'] = (2.25, 1.5)
        if cli_args['tailfilter_size'] == (9, 9):
            cli_args['tailfilter_size'] = (15, 15)

    config_data = train_pca_wrapper(input_dir, cli_args, output_dir, output_file)
    # write config_data to config_file if there is one
    if ctx_obj.get('config_path'):
        # combine new config with old config to add output pca path to config.yaml
        combine_new_config(ctx_obj.get('config_path'), config_data)
    

@cli.command(name='apply-pca', help='Compute PCA Scores of extraction data given a pre-trained PCA')
@common_pca_options
@common_dask_parameters
@click.option('--output-file', default='pca_scores', type=str, help='Name of h5 file for storing pca results')
@click.option('--pca-path', default='/components', type=str, help='Path to pca components in h5 file')
@click.option('--pca-file', default=None, type=click.Path(), help='Path to PCA results')
@click.option('--fill-gaps', default=True, type=bool, help='Fill dropped frames with nans')
@click.option('--fps', default=30, type=int, help='Frames per second (frame rate)')
@click.option('--detrend-window', default=0, type=float, help="Length of detrend window (in seconds, 0 for no detrending)")
@click.option('--overwrite-pca-apply', default=False, type=bool, help='Used to bypass the pca overwrite question. If True: skip question, run automatically')
@click.pass_obj
def apply_pca(ctx_obj, input_dir, output_dir, output_file, **cli_args):
    # function writes output pc score path to config_data
    config_data, _ = apply_pca_wrapper(input_dir, cli_args, output_dir, output_file)
    # write config_data to config_file if there is one
    if ctx_obj.get('config_path'):
        # combine new config with old config to add output pc score path to config.yaml
        combine_new_config(ctx_obj.get('config_path'), config_data)
        

@cli.command('compute-changepoints', help='Compute the Model-Free Syllable Changepoints based on the PCA/PCA_Scores')
@common_pca_options
@common_dask_parameters
@click.option('--output-file', default='changepoints', type=str, help='Name of h5 file for storing pca results')
@click.option('--pca-file-components', type=click.Path(), default=None, help="Path to PCA components")
@click.option('--pca-file-scores', type=click.Path(), default=None, help='Path to PCA results')
@click.option('--pca-path', default='/components', type=str, help='Path to pca components')
@click.option('--neighbors', type=int, default=1, help="Neighbors to use for peak identification")
@click.option('--threshold', type=float, default=.5, help="Peak threshold to use for changepoints")
@click.option('-k', '--klags', type=int, default=6, help="Lag to use for derivative calculation")
@click.option('-s', '--sigma', type=float, default=3.5, help="Standard deviation of gaussian smoothing filter")
@click.option('-d', '--dims', type=int, default=300, help="Number of random projections to use")
@click.option('--fps', default=30, type=int, help="Frames per second (frame rate)")
@click.pass_obj
def compute_changepoints(ctx_obj, input_dir, output_dir, output_file, **cli_args):
    # function writes output changepoint path to config_data
    config_data = compute_changepoints_wrapper(input_dir, cli_args, output_dir, output_file)
    # write config_data to config_file if there is one
    if ctx_obj.get('config_path'):
        # combine new config with old config to add output pc score path to config.yaml
        combine_new_config(ctx_obj.get('config_path'), config_data)
    

if __name__ == '__main__':
    cli()
