"""
Visualization operations for plotting computed PCs, a Scree Plot, and the Changepoint PDF histogram.
"""

import click
import logging
import warnings
import traceback
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mode


def plot_pca_results(output_dict, save_file: Path, output_dir: Path):
    """
    Plot and save trained PCA results.

    Args:
    output_dict (dict): Dict object containing PCA training results
    save_file (str): Path to save the plots to.
    output_dir (str): Directory containing logger
    """

    try:
        # Plotting PCA Components
        fig, _ = display_components(output_dict["components"], headless=True)
        for ext in ["png", "pdf"]:
            fig.savefig(save_file.with_name(f"pca_components.{ext}"))
        plt.close()
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        click.echo("could not plot components")
        click.echo(f'You may find error logs here: {output_dir / "train.log"}')

    try:
        # Plotting Scree Plot
        fig = scree_plot(output_dict["explained_variance_ratio"], headless=True)
        for ext in ["png", "pdf"]:
            fig.savefig(save_file.with_name(f"pca_scree.{ext}"))
        plt.close()
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        click.echo("could not plot scree")
        click.echo(f'You may find error logs here: {output_dir / "train.log"}')


def display_components(components, cmap="gray", headless=False):
    """
    Plot computed Principal Components.

    Args:
    components (numpy.ndarray): components to plot
    cmap (str): color map to use; default is 'gray'.
    headless (bool): bool flag to run in headless environment

    Returns:
    plt (plt.figure): figure to save
    ax (plt.ax): figure axis variable
    """

    # Get square image size
    im_size = int(np.sqrt(components.shape[1]))
    components = components.reshape((-1, im_size, im_size))

    ntiles_row = ntiles_col = int(np.ceil(np.sqrt(len(components))))
    plotv = np.full((ntiles_row * im_size, ntiles_col * im_size), np.mean(components))
    for i in range(ntiles_row):
        row_slice = slice(i * im_size, (i + 1) * im_size)
        for j in range(ntiles_col):
            img_idx = i * ntiles_col + j
            if img_idx < len(components):
                col_slice = slice(j * im_size, (j + 1) * im_size)
                plotv[row_slice, col_slice] = components[img_idx]

    if headless:
        plt.switch_backend("agg")

    # Plot PCs
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(plotv, cmap=cmap)
    ax.set(
        xticks=[],
        yticks=[],
    )

    return fig, ax


def scree_plot(explained_variance_ratio, headless=False):
    """
    Plot a scree plot describing principal components.

    Args:
    explained_variance_ratio (numpy.array): explained variance ratio of each principal component
    headless (bool): bool flag to run in headless environment

    Returns:
    plt (plt.figure): figure to save
    """

    csum = np.cumsum(explained_variance_ratio) * 100

    if headless:
        plt.switch_backend("agg")

    sns.set_style("ticks")
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(csum)

    ax.set(
        ylim=(0, 100),
        xlim=(0, len(explained_variance_ratio)),
        xlabel="nPCs",
        ylabel="Variance explained (percent)",
    )

    idx = np.where(csum >= 90)[0]
    if len(idx) > 0:
        idx = idx[0]
        ax.axvline(x=idx, color="k", linestyle="--")
        ax.axhline(y=csum[idx], color="k", linestyle="--")
        ax.set_title(f"{csum[idx]:0.2f}% in {idx + 1} pcs")

    sns.despine()

    return fig


def changepoint_dist(cps, headless=False):
    """
    Creates bar plot describing computed Changepoint Distribution.

    Args:
    cps (numpy.ndarray): changepoints to graph
    headless (bool): bool flag to run in headless environment

    Returns:
    plt (plt.figure): figure to save
    ax (plt.ax): figure axis variable
    """

    if cps.size > 0:

        if headless:
            plt.switch_backend("agg")

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.set_style("ticks")

        mode_val = mode(cps).mode
        s = f"Mean {np.mean(cps):.2f}s, median {np.median(cps):.2f}s, mode {mode_val.squeeze():.2f}s"

        print(cps.shape)

        ax = sns.histplot(
            cps, bins=np.linspace(0, 10, 100), stat="density", kde=True, bw_adjust=0.5
        )
        ax.set(
            xlim=(0, 3),
            xticks=np.linspace(0, 3, 11),
            title=s,
            ylabel="Probability density",
            xlabel="Block duration (s)",
        )

        sns.despine()

        return fig, ax
    else:
        warnings.warn("No changepoints detected - check if mouse is present or moving")
