"""Module for multomics plotting functions."""

import muon as mu
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import math
import scanpy as sc

import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.axes import Axes
from matplotlib import colors
import plotly.graph_objects as go

from beartype import beartype
from beartype.typing import Optional, Tuple, List
from numpy.typing import NDArray

from sctoolbox.plotting.general import _save_figure, plot_heatmap
from sctoolbox import settings
logger = settings.logger


@beartype
def check_modality_is_valid(mdata: mu.MuData,
                            modality: str) -> None:
    """
    Check if a given modality is a valid key for modalities saved in the mudata object.

    Parameters
    ----------
    mdata : mu.MuData
        Muon object containing both modalities with clustering information.
    modality : String
        Value with the name of a modality.

    Raises
    ------
    KeyError
        If modality is not in mdata.mod.keys().
    """
    if modality not in mdata.mod.keys():
        raise KeyError(f"\'{modality}\' is not a valid key for modalities saved in the mdata object.")


@beartype
def visualize_cluster_comparison(mdata: mu.MuData,  # noqa: C901
                                 clusters_mod1: str,
                                 clusters_mod2: str,
                                 embedding: str = "X_umap",
                                 save: Optional[str] = None,
                                 title: str = "Cluster comparison",
                                 title_size: int = 16) -> Tuple[matplotlib.figure.Figure, NDArray[Axes]]:
    """
    Create multiple UMAP plots.

    - One for modality 1 clusters on modality 1 UMAP.
    - One for modality 2 clusters on modality 2 UMAP.
    - One for modality 1 data with cells marked according to which modality 2 cluster they belong to.
    - One for modality 2 data with cells marked according to which modality 1 cluster they belong to.
    - One modality 1 plot per modality 2 cluster where only cells from that specific modality 2 cluster are marked,
        all other cells are grayed out.
    - One modality 2 plot per modality 1 cluster where only cells from that specific modality 1 cluster are marked,
        all other cells are grayed out.

    Parameters
    ----------
    mdata : mu.MuData
        Muon object modalities both combined and separate, each with clustering information.
        Value containing the name of the second modality. Must be a valid key in mdata.mod.keys().
    clusters_mod1 : str
        Value with the name of the modality 1 clustering column in the modality 1 matrix or in the joint matrix.
    clusters_mod2 : str
        Value with the name of the modality 2 clustering column in the modality 2 matrix or in the joint matrix.
    embedding : str, default="X_umap"
        Value with embedding method to use for plotting. Must be a key in mdata.mod.obsm
    save: Optional[str], default None
        If given, save the figure to this path.
    title : str, default="Multiomics Cluster comparison"
        Value containing the optional title of the figure as a whole.
    title_size : int, default=20
        Font size of the figure title.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, NDArray[Axes]]
        Figure object and array of axes objects containing the plots.

    Raises
    ------
    ValueError
        If muon has more or less than 2 modalities.
    """
    mod_list = list(mdata.mod.keys())
    if len(mod_list) != 2:
        raise ValueError("Muon object contains more or less than 2 modalities. This is currently not supported.")
    modality_1, modality_2 = list(mdata.mod.keys())

    # Assure that values for parameters clusters_mod1 and clusters_mod2 have the correct prefixes for mdata.obs
    clusters_mod1 = ":".join([modality_1, clusters_mod1]) if not clusters_mod1.startswith(modality_1) else clusters_mod1
    clusters_mod2 = ":".join([modality_2, clusters_mod2]) if not clusters_mod2.startswith(modality_2) else clusters_mod2

    # Add additional columns with clusternames modified to be <modality>:<cluster name> for both modalities
    mdata.obs[f"{modality_1}:tmp_cluster_names"] = mdata.obs[clusters_mod1].apply(lambda x: ":".join([modality_1, x]))
    mdata.obs[f"{modality_2}:tmp_cluster_names"] = mdata.obs[clusters_mod2].apply(lambda x: ":".join([modality_2, x]))

    # Set variables containing cluster names for modality 1 and for modality 2 clusters
    cluster_names_mod1 = mdata.obs[clusters_mod1].cat.categories
    cluster_names_mod2 = mdata.obs[clusters_mod2].cat.categories

    fig_adj = len(cluster_names_mod1) + len(cluster_names_mod2)

    # Set variables containing embedding for modality 1 and for modality 2
    embedding_mod1 = ":".join([modality_1, embedding])
    embedding_mod2 = ":".join([modality_2, embedding])

    # Determine number of rows for figure object by choosing the max number of clusters between modality 1 and modality 2
    # + 2 rows for the first set of plots
    n_rows = max(len(mdata.obs[clusters_mod2].unique()), len(mdata.obs[clusters_mod1].unique())) + 2

    # Generate figure and axes objects
    fig, axes = plt.subplots(n_rows, 2, figsize=(12 + fig_adj / 10, 4 * n_rows))
    fig.suptitle(title, size=title_size)

    # Generate plots
    for i in range(0, n_rows):
        for j in range(0, 2):
            ax = axes[i][j]

            # Modality 1 clusters on modality 1 UMAP and modality 2 clusters on modality 2 UMAP
            if i == 0:
                if j == 0:
                    mu.pl.embedding(mdata, embedding_mod1, color=[f"{modality_1}:tmp_cluster_names"],
                                    title=f"{modality_1} clusters on {modality_1} {embedding}", ax=ax, show=False)
                elif j == 1:
                    mu.pl.embedding(mdata, embedding_mod2, color=[f"{modality_2}:tmp_cluster_names"],
                                    title=f"{modality_2} clusters on {modality_2} {embedding}", ax=ax, show=False)

            # Modality 2 clusters on modality 1 UMAP and modality 1 clusters on modality 2 UMAP
            elif i == 1:
                if j == 0:
                    mu.pl.embedding(mdata, embedding_mod1, color=[f"{modality_2}:tmp_cluster_names"],
                                    title=f"{modality_2} clusters on {modality_1} {embedding}", ax=ax, show=False)
                elif j == 1:
                    mu.pl.embedding(mdata, embedding_mod2, color=[f"{modality_1}:tmp_cluster_names"],
                                    title=f"{modality_1} clusters on {modality_2} {embedding}", ax=ax, show=False)

            # One plot per single modaltiy 2 cluster on modality 1 UMAP
            elif j == 0 and i - 2 <= len(cluster_names_mod2) - 1:
                cluster = cluster_names_mod2[i - 2]
                mu.pl.embedding(mdata, embedding_mod1, color=[f"{modality_2}:tmp_cluster_names"],
                                groups=":".join([modality_2, cluster]),
                                title=f"{modality_2} cluster \"{cluster}\" on {modality_1} {embedding}", ax=ax, show=False)

            # One plot per single modality 1 cluster on modality 2 UMAP
            elif j == 1 and i - 2 <= len(cluster_names_mod1) - 1:
                cluster = cluster_names_mod1[i - 2]
                mu.pl.embedding(mdata, embedding_mod2, color=[f"{modality_1}:tmp_cluster_names"],
                                groups=":".join([modality_1, cluster]),
                                title=f"{modality_1} cluster \"{cluster}\" on {modality_2} {embedding}", ax=ax, show=False)

            # Hide plots that are not filled in
            elif i - 2 > len(cluster_names_mod1) - 1 or i - 2 > len(cluster_names_mod2) - 1:
                ax.axis('off')

            # Set axis labels so that y-axis label is only displayed for plots on the left
            if j == 0:
                ax.set_ylabel(f"{embedding}_2")
            elif j == 1:
                ax.set_ylabel("")

            # Set axis labels so that x-axis label is only displayed for plots on the bottom
            if (j == 0 and i - 2 < len(cluster_names_mod2) - 1) or (j == 1 and i - 2 < len(cluster_names_mod1) - 1):
                ax.set_xlabel("")
            else:
                ax.set_xlabel(f"{embedding}_1")

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Remove temporary clustering columns
    mdata.obs.drop(columns=[f"{modality_1}:tmp_cluster_names", f"{modality_2}:tmp_cluster_names"], inplace=True)

    _save_figure(save)

    return fig, axes


@beartype
def compare_cluster_heatmap(mdata: mu.MuData,
                            dfs_heatmaps: NDArray[pd.DataFrame],
                            save: Optional[str] = None,
                            title_heatmaps: str = "Cluster comparison heatmap",
                            title_size: int = 16,
                            cmap: str = "Greys") -> Tuple[matplotlib.figure.Figure, NDArray[Axes]]:
    """
    Plot one heatmap for each modality to show percentage overlap of cells per cluster.

    Parameters
    ----------
    mdata : mu.MuData
        Muon object containing both modalities with clustering information.
    dfs_heatmaps : NDArray[pd.DataFrame]
        Array of cluster comparison scores for each modality.
    save: Optional[str], default None
        If given, save the figure to this path.
    title_heatmaps : str, default 'Cluster comparison heatmap'
        Title for the heatmap plot.
    title_size : int, default 20
        Font size of the heatmap figure title.
    cmap : str, default="Greys"
        Color map for the heatmap plot.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, NDArray[Axes]]
        Figure and axes of the heatmap plots showing best matches for both modalities
        and the combined match score.

    Raises
    ------
    ValueError
        If muon has more or less than 2 modalities.
    """
    mod_list = list(mdata.mod.keys())
    if len(mod_list) != 2:
        raise ValueError("Muon object contains more or less than 2 modalities. This is currently not supported.")
    modality_1, modality_2 = list(mdata.mod.keys())

    # Generate figure and axes objects for heatmaps and set figure title
    fig_heatmaps, axes_heatmaps = plt.subplots(1, 2, figsize=(12, 4))
    fig_heatmaps.suptitle(title_heatmaps, size=title_size)

    # Plot heatmaps for clutster comparison
    for i in range(0, len(dfs_heatmaps)):
        # reset index of data frame
        dfs_heatmaps[i] = dfs_heatmaps[i].reset_index()
        title_heatmap = f"{modality_1 if i == 0 else modality_2} clusters matched against {modality_2 if i == 0 else modality_1} clusters."
        plot_heatmap(dfs_heatmaps[i], axes_heatmaps[i], index=dfs_heatmaps[i].columns[0], columns=dfs_heatmaps[i].columns[1],
                     values="Cells_per_cluster_pct", cmap=cmap, title=title_heatmap, title_size=title_size,
                     flip=False if i == 0 else True, vmin=0.00, vmax=1.00)

    plt.tight_layout()
    fig_heatmaps.show()

    _save_figure(save)

    return fig_heatmaps, axes_heatmaps


@beartype
def plot_sankey(data_frame: pd.DataFrame,
                modalities: List[str],
                clustercols: List[str],
                title: str = "Cluster Overlap Sankey Diagram",
                title_size: int = 16,
                cmap: str = "tab20",
                save: Optional[str] = None) -> go.Figure:
    """
    Plot a Sankey diagram to show connections between modality 1 clusters and modality 2 clusters.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        Data frame containing the numbers of cells of each modality 1 cluster that overlap with each modality 2 cluster.
    modalities : List of Strings
        List of modality names.
    clustercols : List of Strings
        List of clustering column names.
    title : String, default="Cluster Overlap Sankey Diagram"
        Title of the Sankey diagram.
    title_size : Integer, default=16
        Font size of the figure title.
    cmap : String, default="tab20"
        Name of the matplotlib colormap to use for the diagram.
    save: Optional[str], default None
        If given, save the figure to this path.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with the sankey plot.
    """

    clust1 = f"{modalities[0]}:{clustercols[0]}"
    clust2 = f"{modalities[1]}:{clustercols[1]}"

    # Generate labels for diagram
    labels_mod1 = [":".join([modalities[0], cluster]) for cluster in data_frame[clust1].unique()]
    labels_mod2 = [":".join([modalities[1], cluster]) for cluster in data_frame[clust2].unique()]
    labels = labels_mod1 + labels_mod2
    labels.sort()

    # Generate colors from colormap, one color per label
    colormap = plt.get_cmap(cmap)
    color = [colors.to_hex(colormap(i / len(labels))) for i in range(len(labels))]

    # Generate source, target and value lists
    source = [labels.index(":".join([modalities[0], item])) for item in data_frame[clust1]]
    target = [labels.index(":".join([modalities[1], item])) for item in data_frame[clust2]]
    value = list(data_frame["Cells_per_cluster"])

    # Generate color_lines list based on source colors
    color_lines = [color[src] for src in source]

    # Plot sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=color
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            label=labels,
            color=color_lines
        ))])

    fig.update_layout(title_text=title, font_size=title_size)

    # Save sankey diagram
    if save:
        save_path = Path(settings.figure_dir) / save
        fig.write_html(save_path)

    return fig


@beartype
def visualize_modality_grid(mdata: mu.MuData,
                            clustercols: List[str],
                            embedding: str = "X_umap",
                            use_mod_1: bool = True,
                            title: str = "",
                            title_size: int = 16,
                            save: Optional[str] = None
                            ) -> Tuple[matplotlib.figure.Figure, NDArray[Axes]]:
    """
    Plot grid umaps highlighting best fitting cluster pair.

    Function plots a grid of modality 1 UMAPs that shows each modality 1 cluster together with each modality 2 cluster
    with the shared cells marked. For each modality 1 cluster the plot with the best match modality 2 cluster
    is highlighted.

    Parameters
    ----------
    mdata : mu.MuData
        Mudata object conainting all information in regards to both modalities.
    clustercols : List[str]
        List of cluster column names for the modalities. Must match the order
        of modality names in 'modalities' parameter and be valid columns for mdata.obs.
    embedding : str, default="X_umap"
        Value with the name of the embedding option for plotting.
    use_mod_1 : bool, default=True
        If True uses modality 1 else uses modality 2.
    title : str, default=""
        Title of the plot grid figure. The default value will plot no title.
    title_size : int, default=20
        Font size of the figure title.
    save: Optional[str], default None
        If given, save the figure to this path.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, NDArray[Axes]]
        Figure and axes of the UMAP plots.

    Raises
    ------
    ValueError
        If muon has more or less than 2 modalities.
    """
    # Setup correct keys
    mod_list = list(mdata.mod.keys())
    if len(mod_list) != 2:
        raise ValueError("Muon object contains more or less than 2 modalities. This is currently not supported.")
    modality_1, modality_2 = list(mdata.mod.keys())

    best_match = modality_1 if use_mod_1 else modality_2
    embedding = f"{best_match}:{embedding}"
    clust_1 = f"{modality_1}:{clustercols[0]}"
    clust_2 = f"{modality_2}:{clustercols[1]}"

    cell_list = mdata.obs.index

    cluster_list_1 = mdata.obs[clust_1].cat.categories  # x-axis / columns
    cluster_list_2 = mdata.obs[clust_2].cat.categories  # y-axis / rows

    cols = len(cluster_list_2)
    rows = len(cluster_list_1)

    # Get best matches data frame from mdata.uns
    best_matches_df = mdata.uns["cluster_comparison_best_matches"]

    # Get best matches index list
    best_matches = list(best_matches_df.idxmax(axis=1 if use_mod_1 else 0))

    # Get a copy of the mdata for plotting
    mdata_copy = mdata.copy()

    # Create grid
    fig, axes = plt.subplots(ncols=cols, nrows=rows, figsize=(cols * 6, rows * 4))
    fig.suptitle(title, size=title_size)

    # Plot
    for col_i, mod2_cluster in enumerate(cluster_list_2):  # columns
        for row_i, mod1_cluster in enumerate(cluster_list_1):  # rows
            # Find individual and common cells
            mod1_cells = mdata.obs.index[mdata.obs[clust_1] == mod1_cluster].tolist()
            mod2_cells = mdata.obs.index[mdata.obs[clust_2] == mod2_cluster].tolist()
            common_cells = list(set(mod1_cells).intersection(set(mod2_cells)))

            # Select colors
            conditions = [[cell in common_cells for cell in cell_list],
                          [cell in mod1_cells and cell not in common_cells for cell in cell_list],
                          [cell in mod2_cells and cell not in common_cells for cell in cell_list]]

            choices = ["shared", f"{modality_1}:{mod1_cluster}", f"{modality_2}:{mod2_cluster}"]
            color_list = np.select(conditions, choices, "No assignment")

            # Add plotting column to mudata
            mdata_copy.obs[f"{best_match}:cells color"] = color_list

            # Plot embedding scatter
            ax = axes[row_i, col_i]
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)

                mu.pl.embedding(mdata_copy, embedding, color=[f"{best_match}:cells color"],
                                palette={"shared": "purple",
                                         f"{modality_1}:{mod1_cluster}": "darksalmon",
                                         f"{modality_2}:{mod2_cluster}": "skyblue",
                                         "No assignment": "lightgrey"},
                                ax=ax, show=False)

            # Remove plotting column from mudata
            mdata_copy.obs = mdata_copy.obs.drop(labels=f"{best_match}:cells color", axis=1)

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            # Mark best match
            if best_match == modality_1 and not row_i >= len(best_matches):
                if best_matches[row_i] == mod2_cluster and best_matches[row_i] != 'nan':
                    ax.patch.set_edgecolor('black')
                    ax.patch.set_linewidth(3)
            elif best_match == modality_2 and not col_i >= len(best_matches):
                if best_matches[col_i] == mod1_cluster and best_matches[col_i] != 'nan':
                    ax.patch.set_edgecolor('black')
                    ax.patch.set_linewidth(3)

            if col_i == 0:
                ax.set_ylabel(":".join([modality_1, mod1_cluster]), fontsize=9)
            if row_i == 0:
                ax.set_title(":".join([modality_2, mod2_cluster]), fontsize=9)

    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.98])

    # Save sankey diagram if output path for figures is given
    _save_figure(save)

    return fig, axes


@beartype
def umap_parameter_sweep(mdata: mu.MuData,
                         min_dist_range: Tuple[float, float, float],
                         spread_range: Tuple[float, float, float],
                         n_cols: int = 3
                         ) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    """
    Iterate over a range of UMAP parameters and plot the results in a grid.

    Parameters
    ----------
    mdata : mu.MuData
        MuData object to run UMAP on.
    min_dist_range : Tuple[float, float, float]
        Range of min_dist values to iterate over as (start, stop, stepsize).
    spread_range : Tuple[float, float, float]
        Range of spread values to iterate over as (start, stop, stepsize).
    n_cols : int, default=3
        Number of columns in the plot grid.


    Returns
    -------
    Tuple[matplotlib.figure.Figure, np.ndarray]
        Figure and axes of the UMAP plots.
    """
    # Generate ranges from tuples
    min_dist_values = np.arange(*min_dist_range)
    spread_values = np.arange(*spread_range)

    # Generate all valid parameter combinations
    # min_dist must be less than or equal to spread
    combinations = [(min_dist, spread)
                    for min_dist in min_dist_values
                    for spread in spread_values
                    if min_dist <= spread]

    # Calculate grid dimensions
    n_plots = len(combinations)
    n_rows = math.ceil(n_plots / n_cols)

    # Create figure and axes grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for ax, (min_dist, spread) in zip(axes, combinations):
        logger.info(f"Running UMAP with min_dist={min_dist:.2f}, spread={spread:.2f}")

        # Run UMAP with current parameters
        sc.tl.umap(mdata, min_dist=min_dist, spread=spread)

        # Plot UMAP onto the current axis
        sc.pl.umap(mdata, ax=ax, show=False,
                   title=f"min_dist={min_dist:.2f}, spread={spread:.2f}")

    # Hide any unused axes
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    plt.tight_layout()

    return fig, axes
