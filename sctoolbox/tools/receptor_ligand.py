"""Tools for a receptor-ligand analysis."""
import math
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations_with_replacement
import scipy
from sklearn.preprocessing import minmax_scale
from pathlib import Path
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Patch
import matplotlib.lines as lines
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import seaborn as sns
import igraph as ig
import pycirclize
from tqdm import tqdm
import warnings
import logging
import liana.resource as liana_res
import networkx as nx
from beartype.typing import Optional, Tuple, List, Dict
import numpy.typing as npt
from beartype import beartype

import itertools

import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings

from sctoolbox.utils.adata import add_uns_info, in_uns, get_uns
from sctoolbox.utils.bioutils import pseudobulk_table
from sctoolbox.plotting.general import _save_figure, plot_table

logger = settings.logger

# -------------------------------------------------- setup functions -------------------------------------------------- #


@deco.log_anndata
@beartype
def download_db(adata: sc.AnnData,
                db_path: str,
                ligand_column: str,
                receptor_column: str,
                sep: str = "\t",
                inplace: bool = False,
                overwrite: bool = False,
                remove_duplicates: bool = True,
                report: Optional[Tuple[str, str]] = None) -> Optional[sc.AnnData]:
    r"""
    Download table of receptor-ligand interactions and store in adata.

    Parameters
    ----------
    adata : sc.AnnData
        Analysis object the database will be added to.
    db_path : str
        A valid database needs a column with receptor gene ids/symbols and ligand gene ids/symbols.
        Either a path to a database table e.g.:
        - Human: http://tcm.zju.edu.cn/celltalkdb/download/processed_data/human_lr_pair.txt
        - Mouse: http://tcm.zju.edu.cn/celltalkdb/download/processed_data/mouse_lr_pair.txt
        or the name of a database available in the LIANA package:
        - https://liana-py.readthedocs.io/en/latest/notebooks/prior_knowledge.html#Ligand-Receptor-Interactions
    ligand_column : str
        Name of the column with ligand gene names.
        Use 'ligand_gene_symbol' for the urls provided above. For LIANA databases use 'ligand'.
    receptor_column : str
        Name of column with receptor gene names.
        Use 'receptor_gene_symbol' for the urls provided above. For LIANA databases use 'receptor'.
    sep : str, default '\t'
        Separator of database table.
    inplace : bool, default False
        Whether to copy `adata` or modify it inplace.
    overwrite : bool, default False
        If True will overwrite existing database.
    remove_duplicates : bool, default True
        If True, removes duplicate receptor-ligand combinations.
    report : Optional[Tuple[str, str]]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.

    Notes
    -----
    This will remove all information stored in adata.uns['receptor-ligand']

    Returns
    -------
    Optional[sc.AnnData]
        If not inplace, return copy of adata with added database path and
        database table to adata.uns['receptor-ligand']

    Raises
    ------
    ValueError
        1: If ligand_column is not in database.
        2: If receptor_column is not in database.
        3: If db_path is neither a file nor a LIANA resource.
    """

    # datbase already existing?
    if not overwrite and "receptor-ligand" in adata.uns and "database" in adata.uns["receptor-ligand"]:
        warnings.warn("Database already exists! Skipping. Set `overwrite=True` to replace.")

        if inplace:
            return
        else:
            return adata

    try:
        database = pd.read_csv(db_path, sep=sep)
        liana = False
    except FileNotFoundError:
        # Check if a LIANA resource
        if db_path in liana_res.show_resources():
            # get LIANA db
            database = liana_res.select_resource(db_path)
            # explode protein complexes interactions into single protein interactions
            database = liana_res.explode_complexes(database)
            liana = True
        else:
            raise ValueError(f"{db_path} is neither a valid file nor on of the available LIANA resources ({liana_res.show_resources()}).")

    # check column names in table
    if ligand_column not in database.columns:
        raise ValueError(f"Ligand column '{ligand_column}' not found in database! Available columns: {database.columns}")
    if receptor_column not in database.columns:
        raise ValueError(f"Receptor column '{receptor_column}' not found in database! Available columns: {database.columns}")

    # Remove duplicates if requested
    if remove_duplicates:
        # Get mask of duplicated receptor-ligand combinations
        duplicate_mask = database.duplicated(subset=[ligand_column, receptor_column], keep='first')
        # Remove duplicates
        database = database[~duplicate_mask]

    modified_adata = adata if inplace else adata.copy()

    # setup dict to store information old data will be overwriten!
    modified_adata.uns['receptor-ligand'] = dict()

    modified_adata.uns['receptor-ligand']['database_path'] = db_path
    modified_adata.uns['receptor-ligand']['database'] = database
    modified_adata.uns['receptor-ligand']['ligand_column'] = ligand_column
    modified_adata.uns['receptor-ligand']['receptor_column'] = receptor_column

    # report
    if settings.report_dir and report:
        with open(Path(settings.report_dir) / report[0], "w") as file:
            file.write('Used database: ' + (f"LIANA - {db_path}" if liana else db_path))
        plot_table(table=database, report=report[1], crop=10)

    if not inplace:
        return modified_adata


@deco.log_anndata
@beartype
def calculate_interaction_table(adata: sc.AnnData,
                                cluster_column: str,
                                gene_index: Optional[str] = None,
                                normalize: Optional[int] = None,
                                weight_by_ep: Optional[bool] = True,
                                inplace: bool = False,
                                overwrite: bool = False,
                                layer: Optional[str] = None) -> Optional[sc.AnnData]:
    """
    Calculate an interaction table of the clusters defined in adata.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object that holds the expression values and clustering
    cluster_column : str
        Name of the cluster column in adata.obs.
    gene_index : Optional[str], default None
        Column in adata.var that holds gene symbols/ ids.
        Uses index when None.
    normalize : Optional[int], default None
        Correct clusters to given size. If None, max clustersize is used.
    weight_by_ep : Optional[bool], default True
        Whether to weight the expression Z-Score by the expression proportion.
    inplace : bool, default False
        Whether to copy `adata` or modify it inplace.
    overwrite : bool, default False
        If True will overwrite existing interaction table.
    layer : Optional[str], default None
        The layer used for score computation. None to use `adata.X`. It is recommended to use raw or normalized data for statistical analysis.

    Returns
    -------
    Optional[sc.AnnData]
        If not inplace, return copy of adata with added interactions table to adata.uns['receptor-ligand']['interactions']

    Raises
    ------
    ValueError
        1: If receptor-ligand database cannot be found.
        2: Id database genes do not match adata genes.
        3: If the adata layer does not exist.
    Exception
        If not interactions were found.
    """

    if "receptor-ligand" not in adata.uns.keys():
        raise ValueError("Could not find receptor-ligand database. Please setup database with `download_db(   )` before running this function.")

    if layer and layer not in adata.layers:
        raise ValueError(f"Layer {layer} not found in adata.layers")

    # interaction table already exists?
    if not overwrite and "receptor-ligand" in adata.uns and "interactions" in adata.uns["receptor-ligand"]:
        warnings.warn("Interaction table already exists! Skipping. Set `overwrite=True` to replace.")

        if inplace:
            return
        else:
            return adata

    r_col, l_col = adata.uns["receptor-ligand"]["receptor_column"], adata.uns["receptor-ligand"]["ligand_column"]
    index = adata.var[gene_index] if gene_index else adata.var.index

    # test if database gene columns overlap with adata.var genes
    if (not set(adata.uns["receptor-ligand"]["database"][r_col]) & set(index)
            or not set(adata.uns["receptor-ligand"]["database"][l_col]) & set(index)):
        raise ValueError(f"Database columns '{r_col}', '{l_col}' don't match adata.var['{gene_index}']. Please make sure to select gene ids or symbols in all columns.")

    # ----- compute cluster means and expression percentage for each gene -----
    # gene mean expression per cluster
    cl_mean_expression = pseudobulk_table(adata, groupby=cluster_column, layer=layer, gene_index=gene_index)
    # percent cells in cluster expressing gene
    cl_percent_expression = pd.DataFrame(index=index)
    # number of cells for each cluster
    clust_sizes = {}

    # fill above tables
    for cluster in tqdm(set(adata.obs[cluster_column]), desc="computing cluster gene scores"):
        # filter adata to a specific cluster
        cluster_adata = adata[adata.obs[cluster_column] == cluster]
        clust_sizes[cluster] = len(cluster_adata)

        # select the data layer
        cluster_layer = cluster_adata.layers[layer] if layer else cluster_adata.X

        # -- compute expression percentage --
        # get nonzero expression count for all genes
        _, cols = cluster_layer.nonzero()
        gene_occurence = Counter(cols)

        cl_percent_expression[cluster] = 0
        cl_percent_expression.iloc[list(gene_occurence.keys()), cl_percent_expression.columns.get_loc(cluster)] = list(gene_occurence.values())
        cl_percent_expression[cluster] = cl_percent_expression[cluster] / len(cluster_adata.obs) * 100

    # combine duplicated genes through mean (can happen due to mapping between organisms)
    if len(set(cl_mean_expression.index)) != len(cl_mean_expression):
        cl_mean_expression = cl_mean_expression.groupby(cl_mean_expression.index).mean()
        cl_percent_expression = cl_percent_expression.groupby(cl_percent_expression.index).mean()

    # if user does not provide normalization factor - use max clustersize
    max_clust_size = np.max(list(clust_sizes.values()))
    max_clust_size = int(max_clust_size)
    if normalize is None:
        normalize = max_clust_size
    elif normalize < max_clust_size:
        warnings.warn("Value of normalize parameter is smaller then max clustersize. \nClusters with size larger then normalize will be overproportionally scaled.")

    # cluster scaling factor for cluster size correction
    scaling_factor = {k: v / normalize for k, v in clust_sizes.items()}

    # ----- Scale data before computing zscores -----
    # Apply scaling to mean expression data
    scaled_mean_expression = cl_mean_expression.copy()
    for cluster in scaled_mean_expression.columns:
        scaled_mean_expression[cluster] = scaled_mean_expression[cluster] * scaling_factor[cluster]

    # ----- compute zscore of cluster means for each gene -----
    # create pandas functions that show progress bar
    tqdm.pandas(desc="computing Z-scores")

    zscores = cl_mean_expression.progress_apply(lambda x: pd.Series(scipy.stats.zscore(x, nan_policy='omit'), index=cl_mean_expression.columns), axis=1)

    # nan to 0 to interpret static genes aka all with the same expression as "inactive"
    zscores.fillna(value=0, inplace=True)

    interactions = {"receptor_cluster": [],
                    "ligand_cluster": [],
                    "receptor_gene": [],
                    "ligand_gene": [],
                    "receptor_score": [],
                    "ligand_score": [],
                    "receptor_percent": [],
                    "ligand_percent": [],
                    "receptor_scale_factor": [],
                    "ligand_scale_factor": [],
                    "receptor_cluster_size": [],
                    "ligand_cluster_size": []}

    # ----- create interaction table -----
    for _, (receptor, ligand) in tqdm(adata.uns["receptor-ligand"]["database"][[r_col, l_col]].iterrows(),
                                      total=len(adata.uns["receptor-ligand"]["database"]),
                                      desc="finding receptor-ligand interactions"):

        if receptor is np.nan or ligand is np.nan:
            continue

        if receptor not in zscores.index or ligand not in zscores.index:
            continue

        # add interactions to dict
        for receptor_cluster in zscores.columns:
            for ligand_cluster in zscores.columns:
                interactions["receptor_gene"].append(receptor)
                interactions["ligand_gene"].append(ligand)
                interactions["receptor_cluster"].append(receptor_cluster)
                interactions["ligand_cluster"].append(ligand_cluster)
                interactions["receptor_score"].append(zscores.loc[receptor, receptor_cluster])
                interactions["ligand_score"].append(zscores.loc[ligand, ligand_cluster])
                interactions["receptor_percent"].append(cl_percent_expression.loc[receptor, receptor_cluster])
                interactions["ligand_percent"].append(cl_percent_expression.loc[ligand, ligand_cluster])
                interactions["receptor_scale_factor"].append(scaling_factor[receptor_cluster])
                interactions["ligand_scale_factor"].append(scaling_factor[ligand_cluster])
                interactions["receptor_cluster_size"].append(clust_sizes[receptor_cluster])
                interactions["ligand_cluster_size"].append(clust_sizes[ligand_cluster])

    interactions = pd.DataFrame(interactions)

    # compute interaction score
    if weight_by_ep:
        interactions["receptor_score"] = interactions["receptor_score"] * (interactions["receptor_percent"] / 100)
        interactions["ligand_score"] = interactions["ligand_score"] * (interactions["ligand_percent"] / 100)

    interactions["interaction_score"] = interactions["receptor_score"] + interactions["ligand_score"]

    # clean up columns
    interactions.drop(columns=["receptor_scale_factor", "ligand_scale_factor"], inplace=True)

    # no interactions found error
    if not len(interactions):
        raise Exception("Failed to find any receptor-ligand interactions. Consider using a different database.")

    # add to adata
    modified_adata = adata if inplace else adata.copy()

    modified_adata.uns['receptor-ligand']['interactions'] = interactions

    if not inplace:
        return modified_adata

# -------------------------------------------------- plotting functions -------------------------------------------------- #


@deco.log_anndata
@beartype
def interaction_violin_plot(adata: sc.AnnData,
                            min_perc: int | float,
                            save: Optional[str] = None,
                            figsize: Tuple[int | float, int | float] = (5, 20),
                            dpi: Optional[int | float] = None) -> npt.ArrayLike:
    """
    Generate violin plot of pairwise cluster interactions.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object
    min_perc : int | float
        Minimum percentage of cells in a cluster that express the respective gene. A value from 0-100.
    save : str, default None
        Output filename. Uses the internal 'sctoolbox.settings.figure_dir'.
    figsize : Tuple[int | float, int | float], default (5, 20)
        Figure size
    dpi : Optional[int | float], default None
        The resolution of the figure in dots-per-inch. Overwrites `sctoolbox.settings.dpi`.

    Returns
    -------
    npt.ArrayLike
        Object containing all plots. As returned by matplotlib.pyplot.subplots
    """
    if dpi is None:
        dpi = settings.dpi

    # check if data is available
    _check_interactions(adata)

    interactions = get_interactions(adata)

    rows = len(set(interactions["receptor_cluster"]))

    fig, axs = plt.subplots(ncols=1, nrows=rows, figsize=figsize, dpi=dpi, tight_layout={'rect': (0, 0, 1, 0.95)})  # prevent label clipping; leave space for title
    flat_axs = axs.flatten()

    # generate violins of one cluster vs rest in each iteration
    for i, cluster in enumerate(sorted(set(interactions["receptor_cluster"].tolist() + interactions["ligand_cluster"].tolist()))):
        cluster_interactions = get_interactions(adata, min_perc=min_perc, group_a=[cluster])

        # get column of not main clusters
        cluster_interactions["Cluster"] = cluster_interactions.apply(lambda x: x.iloc[1] if x.iloc[0] == cluster else x.iloc[0], axis=1).tolist()

        try:
            # temporarily change logging because of message see link below
            # https://discourse.matplotlib.org/t/why-am-i-getting-this-matplotlib-error-for-plotting-a-categorical-variable/21758
            former_level = logging.getLogger("matplotlib").getEffectiveLevel()
            logging.getLogger("matplotlib").setLevel('WARNING')

            plot = sns.violinplot(x=cluster_interactions["Cluster"],
                                  y=cluster_interactions["interaction_score"],
                                  ax=flat_axs[i])
        finally:
            logging.getLogger("matplotlib").setLevel(former_level)

        plot.set_xticks(plot.get_xticks())  # https://stackoverflow.com/a/68794383/19870975
        plot.set_xticklabels(plot.get_xticklabels(), rotation=90)

        flat_axs[i].set_title(f"Cluster {cluster}")

    # save plot
    _save_figure(save, dpi=dpi)

    return axs


@deco.log_anndata
@beartype
def hairball(adata: sc.AnnData,
             min_perc: int | float,
             interaction_score: float | int = 0,
             interaction_perc: Optional[int | float] = None,
             save: Optional[str] = None,
             title: Optional[str] = "Network",
             color_min: float | int = 0,
             color_max: Optional[float | int] = None,
             cbar_label: str = "Interaction count",
             show_count: bool = False,
             restrict_to: Optional[list[str]] = None,
             additional_nodes: Optional[list[str]] = None,
             hide_edges: Optional[list[Tuple[str, str]]] = None,
             node_size: int | float = 10,
             node_label_size: int | float = 12) -> npt.ArrayLike:
    """
    Generate network graph of interactions between clusters. See cyclone plot for alternative.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object
    min_perc : int | float
        Minimum percentage of cells in a cluster that express the respective gene. A value from 0-100.
    interaction_score : float | int, default 0
        Interaction score must be above this threshold for the interaction to be counted in the graph.
    interaction_perc : Optional[int | float], default None
        Select interaction scores above or equal to the given percentile. Will overwrite parameter interaction_score. A value from 0-100.
    save : str, default None
        Output filename. Uses the internal 'sctoolbox.settings.figure_dir'.
    title : str, default 'Network'
        The plots title.
    color_min : float, default 0
        Min value for color range.
    color_max : Optional[float | int], default None
        Max value for color range.
    cbar_label : str, default 'Interaction count'
        Label above the colorbar.
    show_count : bool, default False
        Show the interaction count in the hairball.
    restrict_to : Optional[list[str]], default None
        Only show given clusters provided in list.
    additional_nodes : Optional[list[str]], default None
        List of additional node names displayed in the hairball.
    hide_edges : Optional[list[Tuple[str, str]]], default None
        List of tuples with node names that should not have an edge shown. Order doesn't matter. E.g. `[("a", "b")]` to omit the edge between node a and b.
    node_size : int | float, default 10
        Set the size of the nodes.
    node_label_size : int | float, default 12
        Set the font size of the node labels.

    Returns
    -------
    npt.ArrayLike
        Object containing all plots. As returned by matplotlib.pyplot.subplots

    Raises
    ------
    ValueError
        If restrict_to contains invalid clusters.
    """

    # check if data is available
    _check_interactions(adata)

    interactions = get_interactions(adata)

    # any invalid cluster names
    if restrict_to:
        valid_clusters = set.union(set(interactions["ligand_cluster"]), set(interactions["receptor_cluster"]), set(additional_nodes) if additional_nodes else set())
        invalid_clusters = set(restrict_to) - valid_clusters
        if invalid_clusters:
            raise ValueError(f"Invalid cluster in `restrict_to`: {invalid_clusters}")

    # ----- create igraph -----
    graph = ig.Graph()

    # --- set nodes ---
    if restrict_to:
        clusters = restrict_to
    else:
        clusters = list(set(list(interactions["receptor_cluster"]) + list(interactions["ligand_cluster"])))

        # add additional nodes
        if additional_nodes:
            clusters += additional_nodes

    graph.add_vertices(clusters)
    graph.vs['label'] = clusters
    graph.vs['size'] = node_size  # node size
    graph.vs['label_size'] = node_label_size  # label size
    graph.vs['label_dist'] = 2  # distance of label to node # not working
    graph.vs['label_angle'] = 1.5708  # rad = 90 degree # not working

    # --- set edges ---
    for (a, b) in combinations_with_replacement(clusters, 2):
        if hide_edges and ((a, b) in hide_edges or (b, a) in hide_edges):
            continue

        subset = get_interactions(adata, min_perc=min_perc, interaction_score=interaction_score, interaction_perc=interaction_perc, group_a=[a], group_b=[b])

        graph.add_edge(a, b, weight=len(subset))

    # set edge colors/ width based on weight
    colormap = matplotlib.cm.get_cmap('viridis', len(graph.es))
    max_weight = np.max(np.array(graph.es['weight'])) if color_max is None else color_max
    for e in graph.es:
        e["color"] = colormap(e["weight"] / max_weight, e["weight"] / max_weight)
        e["width"] = (e["weight"] / max_weight)  # * 10
        # show weights in plot
        if show_count and e["weight"] > 0:
            e["label"] = e["weight"]
            e["label_size"] = 25

    # ----- plotting -----
    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [20, 1]})
    fig.suptitle(title, fontsize=12)

    ig.plot(obj=graph, layout=graph.layout_circle(order=sorted(clusters)), target=axes[0])

    # add colorbar
    cb = matplotlib.colorbar.ColorbarBase(axes[1],
                                          orientation='vertical',
                                          cmap=colormap,
                                          norm=matplotlib.colors.Normalize(0 if color_min is None else color_min,
                                                                           max_weight)
                                          )

    cb.ax.tick_params(labelsize=10)
    cb.ax.set_title(cbar_label, fontsize=10)

    # prevent label clipping out of picture
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)

    if save:
        fig.savefig(f"{settings.figure_dir}/{save}")

    return axes


@deco.log_anndata
@beartype
def cyclone(
    adata: sc.AnnData,
    min_perc: int | float,
    interaction_score: float | int = 0,
    interaction_perc: Optional[int | float] = None,
    title: Optional[str] = "Network",
    color_min: float | int = 0,
    color_max: Optional[float | int] = None,
    cbar_label: str = 'Interaction count',
    colormap: str = 'viridis',
    sector_text_size: int | float = 10,
    directional: bool = False,
    sector_size_is_cluster_size: bool = False,
    show_genes: bool = True,
    gene_amount: int = 5,
    figsize: Tuple[int | float, int | float] = (10, 10),
    dpi: Optional[int | float] = None,
    save: Optional[str] = None,
    report: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Generate network graph of interactions between clusters. See the hairball plot as an alternative.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object
    min_perc : int | float
        Minimum percentage of cells in a cluster that express the respective gene. A value from 0-100.
    interaction_score : float | int, default 0
        Interaction score must be above this threshold for the interaction to be counted in the graph.
    interaction_perc : Optional[int | float], default None
        Select interaction scores above or equal to the given percentile. Will overwrite parameter interaction_score. A value from 0-100.
    title : str, default 'Network'
        The plots title.
    color_min : float, default 0
        Min value for color range.
    color_max : Optional[float | int], default None
        Max value for color range.
    cbar_label : str, default 'Interaction count'
        Label above the colorbar.
    colormap : str, default "viridis"
        The colormap to be used when plotting.
    sector_text_size: int | float, default 10
        The text size for the sector name.
    directional: bool, defalut False
        Determines whether to display the interactions as arrows (Ligand -> Receptor).
    sector_size_is_cluster_size: bool, default False
        Determines whether the sector's size is equivalent to the coresponding cluster's number of cells.
    show_genes: bool, default True
        Determines whether to display the top genes as an additional track.
    gene_amount: int = 5
        The amount of genes per receptor and ligand to display on the outer track (displayed genes per sector = gene_amount *2).
    figsize : Tuple[int | float, int | float], default (10, 10)
        Figure size
    dpi : Optional[int | float], default None
        The resolution of the figure in dots-per-inch. Overwrites `sctoolbox.settings.dpi` and `sctoolbox.settings.report_dpi`.
    save : str, default None
        Output filename. Uses the internal 'sctoolbox.settings.figure_dir'.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib figure object containing the plot.
    """

    # check if data is available
    _check_interactions(adata)

    # commonly used list
    cluster_col_list = ["receptor_cluster", "ligand_cluster"]

    # --------------------- filtering data ---------------------------------
    filtered = get_interactions(anndata=adata,
                                min_perc=min_perc,
                                interaction_score=interaction_score,
                                interaction_perc=interaction_perc)

    # sort data by interaction score for top gene selection
    if show_genes:
        filtered.sort_values(by="interaction_score", ascending=False, inplace=True)

    # ------------------- getting important values ----------------------------

    # saves a table of how many times each receptor cluster interacted with each ligand cluster
    interactions_directed = pd.DataFrame(
        filtered[cluster_col_list].value_counts(),
        columns=["count"]
    )

    interactions_directed.reset_index(inplace=True)

    # Adds the number of interactions for each pair of receptor and ligand
    # if they should be displayed non directional.

    # e.g.: ligand A interacts with receptor B five times
    #       and ligand B interacts with receptor A ten times
    #       therefore A and B interact fifteen times non directional
    if not directional:
        interactions_undirected = interactions_directed.copy()

        # OpenAI GPT-4 supported
        # First, make sure each edge is represented in the same way, by sorting the two ends
        interactions_undirected[cluster_col_list] = np.sort(
            interactions_undirected[cluster_col_list].values,
            axis=1
        )

        # Then, you can group by the two columns and sum the connections
        interactions_undirected = interactions_undirected.groupby(cluster_col_list)["count"].sum().reset_index()
        # <<< OpenAI GPT-4 supported

        interactions = interactions_undirected
    else:
        interactions = interactions_directed

    # gets the size of the clusters and saves it in cluster_to_size
    receptor_cluster_to_size = filtered[["receptor_cluster", "receptor_cluster_size"]]
    ligand_cluster_to_size = filtered[["ligand_cluster", "ligand_cluster_size"]]

    receptor_cluster_to_size = receptor_cluster_to_size.drop_duplicates()
    ligand_cluster_to_size = ligand_cluster_to_size.drop_duplicates()

    receptor_cluster_to_size.rename(
        columns={"receptor_cluster": "cluster", "receptor_cluster_size": "cluster_size"},
        inplace=True
    )
    ligand_cluster_to_size.rename(
        columns={"ligand_cluster": "cluster", "ligand_cluster_size": "cluster_size"},
        inplace=True
    )

    # create a table of cluster sizes
    cluster_to_size = [receptor_cluster_to_size, ligand_cluster_to_size]

    cluster_to_size = pd.concat(cluster_to_size)

    cluster_to_size.drop_duplicates(inplace=True)

    # get a list of the available clusters
    avail_clusters = sorted(list(set(filtered["receptor_cluster"].unique()).union(set(filtered["ligand_cluster"].unique()))))

    # set up for colormapping
    colormap_ = matplotlib.colormaps[colormap]

    norm = matplotlib.colors.Normalize(interactions["count"].min(), interactions["count"].max())

    # --------------------------- setting up for the plot -----------------------------

    # saving the cluster types in the sectors dictionary to use in the plot
    # their size is set to 100 unless their size should be displayed depending on the cluster size

    sectors = {}

    if not sector_size_is_cluster_size:
        for index in avail_clusters:
            sectors[index] = 100
    else:
        for index in avail_clusters:
            sectors[index] = cluster_to_size.set_index("cluster").at[index, "cluster_size"]

    # ------------------------ plotting ---------------------------------

    # create initial object with sectors (cell types)
    circos = pycirclize.Circos(sectors, space=5)

    # creating a from-to-table for a matrix
    interactions_for_matrix = interactions.copy()

    # set link width to 1 so it can be scaled up later
    interactions_for_matrix["count"] = 1

    # TODO remove when this is fixed: https://github.com/moshi4/pyCirclize/issues/75
    interactions_for_matrix[cluster_col_list] = interactions_for_matrix[cluster_col_list] + "_"

    # create the links in pycirclize format
    matrix = pycirclize.parser.Matrix.parse_fromto_table(interactions_for_matrix)

    # TODO remove when this is fixed: https://github.com/moshi4/pyCirclize/issues/75
    matrix._row_names = [r[:-1] for r in matrix.row_names]
    matrix._col_names = [c[:-1] for c in matrix.col_names]
    matrix._matrix.columns = [c[:-1] for c in matrix._matrix.columns]
    matrix._matrix.index = [r[:-1] for r in matrix._matrix.index]
    matrix._name2size = {k[:-1]: v for k, v in matrix._name2size.items()}
    matrix._links = [((l1[0][:-1], *l1[1:]), (l2[0][:-1], *l2[1:])) for l1, l2 in matrix._links]
    interactions_for_matrix[cluster_col_list] = interactions_for_matrix[cluster_col_list].map(lambda x: x[:-1])

    link_list = matrix.to_links()

    # calculates the amount of times sector.name is in the table interactions
    links_per_sector = {sector.name: (interactions == sector.name).sum().sum() for sector in circos.sectors}

    # adjust the start and end width of each link
    for _ in range(len(link_list)):
        # divide sector size by number of links in a sector to get the same width for all links within a sector
        start_link_width = circos.get_sector(link_list[0][0][0]).size / links_per_sector[link_list[0][0][0]]
        end_link_width = circos.get_sector(link_list[0][1][0]).size / links_per_sector[link_list[0][1][0]]

        # link structure:
        # ((start_sector, left_side_of_link, right_side_of_link),
        #  (end_sector, left_side_of_link, right_side_of_link))
        # multiply link width (set to 1) with the respective multiplier from above
        temp = (
            (link_list[0][0][0],
             math.floor(link_list[0][0][1] * start_link_width),
             math.floor(link_list[0][0][2] * start_link_width)),
            (link_list[0][1][0],
             math.floor(link_list[0][1][1] * end_link_width),
             math.floor(link_list[0][1][2] * end_link_width))
        )

        # remove the first link and add the updated version at the end
        link_list.pop(0)
        link_list.append(temp)

    # size of the area where the links will be displayed
    if show_genes:
        link_radius = 65
    else:
        link_radius = 95

    # add the calculated links to the circos object
    for link in matrix.to_links():
        circos.link(
            *link,
            color=colormap_(norm(interactions.set_index(cluster_col_list).loc[(link[0][0], link[1][0]), "count"])),
            r1=link_radius,
            r2=link_radius,
            direction=-1 if directional else 0
        )

    # add the tracks (layers) to the sectors
    for sector in circos.sectors:
        # first track (grey bars connected with the links)
        track = sector.add_track((65, 70)) if show_genes else sector.add_track((95, 100))
        track.axis(fc="grey")
        track.text(sector.name, color="black", size=sector_text_size, r=105)

        # shows cluster/ sector size in the center of the sector
        track.text(f"{cluster_to_size.set_index('cluster').at[sector.name, 'cluster_size']}",
                   r=67 if show_genes else 97,
                   size=9,
                   color="white",
                   adjust_rotation=True)

        # add top receptor/ ligand gene track
        # shows the top x genes (interaction score) with the respective cluster as ligand
        # and the same for receptor
        if show_genes:
            track2 = sector.add_track((73, 77))

            # get first x unique receptor/ ligand genes
            # https://stackoverflow.com/a/17016257/19870975
            sector_r_interactions = filtered[filtered["receptor_cluster"] == sector.name]
            top_receptors = list(dict.fromkeys(sector_r_interactions["receptor_gene"]))[:gene_amount]
            sector_l_interactions = filtered[filtered["ligand_cluster"] == sector.name]
            top_ligands = list(dict.fromkeys(sector_l_interactions["ligand_gene"]))[:gene_amount]

            # generates dummy data for the heatmap function to give it a red and a blue half
            dummy_data = [1] * len(top_receptors) + [2] * len(top_ligands)
            track2.heatmap(
                data=dummy_data,
                rect_kws=dict(ec="black", lw=1, alpha=0.3)
            )

            # compute x tick positions
            # half step to add ticks to the middle of each cell
            half_step = sector.size / len(dummy_data) / 2
            x_tick_pos = np.linspace(half_step,
                                     sector.size - half_step,
                                     len(dummy_data))

            # add gene ticks
            track2.xticks(
                x=x_tick_pos,
                tick_length=2,
                labels=top_receptors + top_ligands,
                label_margin=1,
                label_size=8,
                label_orientation="vertical"
            )

    circos.text(title, r=120, deg=0, size=15)

    # legend
    # legend title
    circos.text(cbar_label, deg=69, r=137, fontsize=10)

    circos.colorbar(
        bounds=(1.1, 0.2, 0.02, 0.5),
        vmin=0 if color_min is None else color_min,
        vmax=color_max if color_max else interactions.max()["count"],
        cmap=colormap_
    )

    patch_handles = [Patch(color="grey", label="Number of cells\nper cluster")]

    # add gene (heatmap) legend
    if show_genes:
        patch_handles.extend([Patch(color=(1, 0, 0, 0.3), label="Top ligand genes\nby interaction score"),
                              Patch(color=(0, 0, 1, 0.3), label="Top receptor genes\nby interaction score")])

    # add directional legend
    if directional:
        patch_handles.append(
            Patch(color=(0, 0, 0, 0), label="Ligand → Receptor")
        )

    # needed to access ax
    fig = circos.plotfig(dpi=int(dpi if dpi else settings.dpi), figsize=figsize)

    # add custom legend to plot
    patch_legend = circos.ax.legend(
        handles=patch_handles,
        bbox_to_anchor=(1, 1),
        fontsize=10,
        handlelength=1
    )
    circos.ax.add_artist(patch_legend)

    _save_figure(save, dpi=dpi)

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True, dpi=dpi)

    return fig


@beartype
def progress_violins(datalist: list[pd.DataFrame],
                     datalabel: list[str],
                     cluster_a: str,
                     cluster_b: str,
                     min_perc: float | int,
                     save: str,
                     figsize: Tuple[int | float, int | float] = (12, 6)) -> str:
    """
    Show cluster interactions over timepoints.

    CURRENTLY NOT FUNCTIONAL!

    TODO Implement function

    Parameters
    ----------
    datalist : list[pd.DataFrame]
        List of interaction DataFrames. Each DataFrame represents a timepoint.
    datalabel : list[str]
        List of strings. Used to label the violins.
    cluster_a : str
        Name of the first interacting cluster.
    cluster_b : str
        Name of the second interacting cluster.
    min_perc : float | int
        Minimum percentage of cells in a cluster each gene must be expressed in.
    save : str
        Path to output file.
    figsize : Tuple[int, int], default (12, 6)
        Tuple of plot (width, height).

    Returns
    -------
    str
    """

    return "Function to be implemented"

    fig, axs = plt.subplots(1, len(datalist), figsize=figsize)
    fig.suptitle(f"{cluster_a} - {cluster_b}")

    flat_axs = axs.flatten()
    for i, (table, label) in enumerate(zip(datalist, datalabel)):
        # filter data
        subset = table[((table["cluster_a"] == cluster_a) & (table["cluster_b"] == cluster_b)
                       | (table["cluster_a"] == cluster_b) & (table["cluster_b"] == cluster_a))
                       & (table["percentage_a"] >= min_perc)
                       & (table["percentage_b"] >= min_perc)]

        v = sns.violinplot(data=subset, y="interaction_score", ax=flat_axs[i])
        v.set_xticklabels([label])

    plt.tight_layout()

    if save is not None:
        fig.savefig(save)


@beartype
def interaction_progress(datalist: list[sc.AnnData],
                         datalabel: list[str],
                         receptor: str,
                         ligand: str,
                         receptor_cluster: str,
                         ligand_cluster: str,
                         figsize: Tuple[int | float, int | float] = (4, 4),
                         dpi: Optional[int | float] = None,
                         save: Optional[str] = None,
                         report: Optional[str] = None) -> matplotlib.axes.Axes:
    """
    Barplot that shows the interaction score of a single interaction between two given clusters over multiple datasets.

    TODO add checks & error messages

    Parameters
    ----------
    datalist : list[sc.AnnData]
        List of anndata objects.
    datalabel : list[str]
        List of labels for the given datalist.
    receptor : str
        Name of the receptor gene.
    ligand : str
        Name of the ligand gene.
    receptor_cluster : str
        Name of the receptor cluster.
    ligand_cluster : str
        Name of the ligand cluster.
    figsize : Tuple[int | float, int | float], default (4, 4)
        Figure size in inch.
    dpi : Optional[int | float], default None
        Dots per inch. Overwrites `sctoolbox.settings.dpi` and `sctoolbox.settings.report_dpi`.
    save : Optional[str], default None
        Output filename. Uses the internal 'sctoolbox.settings.figure_dir'.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.

    Returns
    -------
    matplotlib.axes.Axes
        The plotting object.
    """

    table = []

    for data, label in zip(datalist, datalabel):
        # interactions
        inter = data.uns["receptor-ligand"]["interactions"]

        # select interaction
        inter = inter[
            (inter["receptor_cluster"] == receptor_cluster)
            & (inter["ligand_cluster"] == ligand_cluster)
            & (inter["receptor_gene"] == receptor)
            & (inter["ligand_gene"] == ligand)
        ].copy()

        # add datalabel
        inter["name"] = label

        table.append(inter)

    table = pd.concat(table)

    # plot
    with plt.rc_context({"figure.figsize": figsize, "figure.dpi": dpi if dpi else settings.dpi}):
        plot = sns.barplot(
            data=table,
            x="name",
            y="interaction_score"
        )

        plot.set(
            title=f"{receptor} - {ligand}\n{receptor_cluster} - {ligand_cluster}",
            ylabel="Interaction Score",
            xlabel=""
        )

        plot.set_xticklabels(
            plot.get_xticklabels(),
            rotation=90,
            horizontalalignment='right'
        )

    plt.tight_layout()

    if save:
        _save_figure(save, dpi=dpi)

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True, dpi=dpi)

    return plot


@deco.log_anndata
@beartype
def connectionPlot(adata: sc.AnnData,
                   restrict_to: Optional[list[str]] = None,
                   figsize: Tuple[int | float, int | float] = (10, 15),
                   dpi: Optional[int | float] = None,
                   connection_alpha: Optional[str] = "interaction_score",
                   save: Optional[str] = None,
                   title: Optional[str] = None,
                   # receptor params
                   receptor_cluster_col: str = "receptor_cluster",
                   receptor_col: str = "receptor_gene",
                   receptor_hue: str = "receptor_score",
                   receptor_size: str = "receptor_percent",
                   receptor_genes: Optional[list[str]] = None,
                   # ligand params
                   ligand_cluster_col: str = "ligand_cluster",
                   ligand_col: str = "ligand_gene",
                   ligand_hue: str = "ligand_score",
                   ligand_size: str = "ligand_percent",
                   ligand_genes: Optional[list[str]] = None,
                   # additional plot params
                   filter: Optional[str] = None,
                   lw_multiplier: int | float = 2,
                   dot_size: Tuple[int | float, int | float] = (10, 100),
                   wspace: float = 0.4,
                   line_colors: Optional[str] = "rainbow",
                   dot_colors: str = "flare",
                   xlabel_order: Optional[list[str]] = None,
                   alpha_range: Optional[Tuple[int | float, int | float]] = None,
                   report: Optional[str] = None) -> npt.ArrayLike:
    """
    Show specific receptor-ligand connections between clusters.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object
    restrict_to : Optional[list[str]], default None
        Restrict plot to given cluster names.
    figsize : Tuple[int | float, int | float], default (10, 15)
        Figure size
    dpi : Optional[int | float], default None
        The resolution of the figure in dots-per-inch. Overwrites `sctoolbox.settings.dpi` and `sctoolbox.settings.report_dpi`.
    connection_alpha : str, default 'interaction_score'
        Name of column that sets alpha value of lines between plots. None to disable.
    save : Optional[str], default None
        Output filename. Uses the internal 'sctoolbox.settings.figure_dir'.
    title : Optional[str], default None
        Title of the plot
    receptor_cluster_col : str, default 'receptor_cluster'
        Name of column containing cluster names of receptors. Shown on x-axis.
    receptor_col : str, default 'receptor_gene'
        Name of column containing gene names of receptors. Shown on y-axis.
    receptor_hue : str, default 'receptor_score'
        Name of column containing receptor scores. Shown as point color.
    receptor_size : str, default 'receptor_percent'
        Name of column containing receptor expression percentage. Shown as point size.
    receptor_genes : Optional[list[str]], default None
            Restrict receptors to given genes.
    ligand_cluster_col : str, default 'ligand_cluster'
        Name of column containing cluster names of ligands. Shown on x-axis.
    ligand_col : str, default 'ligand_gene'
        Name of column containing gene names of ligands. Shown on y-axis.
    ligand_hue : str, default 'ligand_score'
        Name of column containing ligand scores. Shown as point color.
    ligand_size : str, default 'ligand_percent'
        Name of column containing ligand expression percentage. Shown as point size.
    ligand_genes : Optional[list[str]], default None
            Restrict ligands to given genes.
    filter : Optional[str], default None
        Conditions to filter the interaction table on. E.g. 'column_name > 5 & other_column < 2'. Forwarded to pandas.DataFrame.query.
    lw_multiplier : int | float, default 2
        Linewidth multiplier.
    dot_size : Tuple[int | float, int | float], default (1, 10)
        Minimum and maximum size of the displayed dots.
    wspace : float, default 0.4
        Width between plots. Fraction of total width.
    line_colors : Optional[str], default 'rainbow'
        Name of the colormap used to color lines. All lines are black if None.
    dot_colors : str, default 'flare'
        Name of the colormap used to color the dots.
    xlabel_order : Optional[list[str]], default None
        Defines the order of data displayed on the x-axis in both plots. Leave None to order alphabetically.
    alpha_range : Optional[Tuple[int | float, int | float]], default None
        Sets the minimum and maximum value for the `connection_alpha` legend. Values outside this range will be set to the min or max value.
        Minimum is mapped to transparent (alpha=0) and maximum to opaque (alpha=1). Will use the min and max values of the data by default (None).
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.

    Returns
    -------
    npt.ArrayLike
        Object containing all plots. As returned by matplotlib.pyplot.subplots

    Raises
    ------
    Exception
        If no onteractions between clsuters are found.
    """

    # check if data is available
    _check_interactions(adata)

    data = get_interactions(adata).copy()

    # filter receptor genes
    if receptor_genes:
        data = data[data[receptor_col].isin(receptor_genes)]

    # filter ligand genes
    if ligand_genes:
        data = data[data[ligand_col].isin(ligand_genes)]

    # filter interactions
    if filter:
        data.query(filter, inplace=True)

    if xlabel_order:
        # create a custom sort function
        sorting_dict = {c: i for i, c in enumerate(xlabel_order)}

        def sort_fun(x):
            return x.map(sorting_dict)
    else:
        sort_fun = None

    # restrict interactions to certain clusters
    if restrict_to:
        data = data[data[receptor_cluster_col].isin(restrict_to) & data[ligand_cluster_col].isin(restrict_to)]
    if len(data) < 1:
        raise Exception(f"No interactions between clusters {restrict_to}")

    # setup subplot
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi if dpi else settings.dpi, gridspec_kw={'wspace': wspace})
    fig.suptitle(title)

    try:
        # temporarily change logging because of message see link below
        # https://discourse.matplotlib.org/t/why-am-i-getting-this-matplotlib-error-for-plotting-a-categorical-variable/21758
        former_level = logging.getLogger("matplotlib").getEffectiveLevel()
        logging.getLogger("matplotlib").setLevel('WARNING')

        # receptor plot
        r_plot = sns.scatterplot(data=data.sort_values(by=receptor_cluster_col, key=sort_fun),
                                 y=receptor_col,
                                 x=receptor_cluster_col,
                                 hue=receptor_hue,
                                 size=receptor_size,
                                 palette=dot_colors,
                                 sizes=dot_size,
                                 legend="brief",
                                 ax=axs[0])
    finally:
        logging.getLogger("matplotlib").setLevel(former_level)

    r_plot.set(xlabel="Cluster", ylabel=None, title="Receptor", axisbelow=True)
    axs[0].tick_params(axis='x', rotation=90)
    axs[0].grid(alpha=0.8)

    try:
        # temporarily change logging because of message see link below
        # https://discourse.matplotlib.org/t/why-am-i-getting-this-matplotlib-error-for-plotting-a-categorical-variable/21758
        former_level = logging.getLogger("matplotlib").getEffectiveLevel()
        logging.getLogger("matplotlib").setLevel('WARNING')

        # ligand plot
        l_plot = sns.scatterplot(data=data.sort_values(by=ligand_cluster_col, key=sort_fun),
                                 y=ligand_col,
                                 x=ligand_cluster_col,
                                 hue=ligand_hue,
                                 size=ligand_size,
                                 palette=dot_colors,
                                 sizes=dot_size,
                                 legend="brief",
                                 ax=axs[1])
    finally:
        logging.getLogger("matplotlib").setLevel(former_level)

    axs[1].yaxis.tick_right()
    l_plot.set(xlabel="Cluster", ylabel=None, title="Ligand", axisbelow=True)
    axs[1].tick_params(axis='x', rotation=90)
    axs[1].grid(alpha=0.8)

    # force tick labels to be populated
    # https://stackoverflow.com/questions/41122923/getting-empty-tick-labels-before-showing-a-plot-in-matplotlib
    fig.canvas.draw()

    # add receptor-ligand lines
    receptors = list(set(data[receptor_col]))

    # create colorramp
    if line_colors:
        cmap = matplotlib.cm.get_cmap(line_colors, len(receptors))
        colors = cmap(range(len(receptors)))
    else:
        colors = ["black"] * len(receptors)

    # scale connection score column between 0-1 to be used as alpha values
    if connection_alpha:
        # set custom min and max values
        if alpha_range:
            def alpha_sorter(x):
                """Set values outside of range to min or max."""
                if x < alpha_range[0]:
                    return alpha_range[0]
                elif x > alpha_range[1]:
                    return alpha_range[1]
                return x
            # add min and max in case they are not present in the data
            alpha_values = list(alpha_range) + [alpha_sorter(val) for val in data[connection_alpha]]
        else:
            alpha_values = data[connection_alpha].tolist()

        # note: minmax_scale sometimes produces values >1. Looks like a rounding error (1.000000000002).
        # end bracket removes added alpha_range if needed
        data["alpha"] = minmax_scale(alpha_values, feature_range=(0, 1))[2 if alpha_range else 0:]
        # fix values >1
        data.loc[data["alpha"] > 1, "alpha"] = 1
    else:
        data["alpha"] = 1

    # find receptor label location
    for i, label in enumerate(axs[0].get_yticklabels()):
        data.loc[data[receptor_col] == label.get_text(), "rec_index"] = i

    # find ligand label location
    for i, label in enumerate(axs[1].get_yticklabels()):
        data.loc[data[ligand_col] == label.get_text(), "lig_index"] = i

    # add receptor-ligand lines
    # draws strongest connection for each pair
    for rec, color in zip(receptors, colors):
        pairs = data.loc[data[receptor_col] == rec]

        for lig in set(pairs[ligand_col]):
            # get all connections for current pair
            connections = pairs.loc[pairs[ligand_col] == lig]

            # get max connection
            max_con = connections.loc[connections["alpha"].idxmax()]

            # stolen from https://matplotlib.org/stable/gallery/userdemo/connect_simple01.html
            # Draw a line between the different points, defined in different coordinate
            # systems.
            con = ConnectionPatch(
                # x in axes coordinates, y in data coordinates
                xyA=(1, max_con["rec_index"]), coordsA=axs[0].get_yaxis_transform(),
                # x in axes coordinates, y in data coordinates
                xyB=(0, max_con["lig_index"]), coordsB=axs[1].get_yaxis_transform(),
                arrowstyle="-",
                color=color,
                zorder=-1000,
                alpha=max_con["alpha"],
                linewidth=max_con["alpha"] * lw_multiplier
            )

            axs[1].add_artist(con)

    # ----- legends -----
    # set receptor plot legend position
    sns.move_legend(r_plot, loc='upper right', bbox_to_anchor=(-1, 1, 0, 0))

    # create legend for connection lines
    if connection_alpha:
        step_num = 5
        s_steps, a_steps = np.linspace(min(alpha_values), max(alpha_values), step_num), np.linspace(0, 1, step_num)

        # create proxy actors https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#proxy-legend-handles
        line_list = [lines.Line2D([], [], color="black", alpha=a, linewidth=a * lw_multiplier, label=f"{np.round(s, 2)}") for a, s in zip(a_steps, s_steps)]
        line_list.insert(0, lines.Line2D([], [], alpha=0, label=connection_alpha))

        # add to current legend
        handles, _ = axs[1].get_legend_handles_labels()
        axs[1].legend(handles=handles + line_list,
                      bbox_to_anchor=(2, 1, 0, 0),
                      loc='upper left',
                      title=receptor_hue if receptor_hue == receptor_size else None)  # fix missing legend label
    else:
        # set ligand plot legend position
        axs[1].legend(bbox_to_anchor=(2, 1, 0, 0),
                      loc='upper left',
                      title=receptor_hue if receptor_hue == receptor_size else None)  # fix missing legend label

    _save_figure(save, dpi=dpi)

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True, dpi=dpi)

    return axs


# -------------------------------------------------- helper functions -------------------------------------------------- #


@deco.log_anndata
@beartype
def get_interactions(anndata: sc.AnnData,
                     min_perc: Optional[float | int] = None,
                     interaction_score: Optional[float | int] = None,
                     interaction_perc: Optional[float | int] = None,
                     group_a: Optional[list[str]] = None,
                     group_b: Optional[list[str]] = None,
                     save: Optional[str] = None) -> pd.DataFrame:
    """
    Get interaction table from anndata and apply filters.

    Parameters
    ----------
    anndata : sc.AnnData
        Anndata object to pull interaction table from.
    min_perc : Optional[float | int], default None
        Minimum percent of cells in a cluster that express the ligand/ receptor gene. Value from 0-100.
    interaction_score : Optional[float | int], default None
        Filter receptor-ligand interactions below given score. Ignored if `interaction_perc` is set.
    interaction_perc : Optional[float | int], default None
        Filter receptor-ligand interactions below the given percentile. Overwrite `interaction_score`. Value from 0-100.
    group_a : Optional[list[str]], default None
        List of cluster names that must be present in any given receptor-ligand interaction.
    group_b : Optional[list[str]], default None
        List of cluster names that must be present in any given receptor-ligand interaction.
    save : Optional[str], default None
        Output filename. Uses the internal 'sctoolbox.settings.table_dir'.

    Returns
    -------
    pd.DataFrame
        Table that contains interactions. Columns:

            - receptor_cluster      = name of the receptor cluster
            - ligand_cluster        = name of the ligand cluster
            - receptor_gene         = name of the receptor gene
            - ligand_gene           = name of the ligand gene
            - receptor_score        = zscore of receptor gene cluster mean expression (scaled by cluster size)
            - ligand_score          = zscore of ligand gene cluster mean expression (scaled by cluster size)
            - receptor_percent      = percent of cells in cluster expressing receptor gene
            - ligand_percent        = percent of cells in cluster expressing ligand gene
            - receptor_cluster_size = number of cells in receptor cluster
            - ligand_cluster_size   = number of cells in ligand cluster
            - interaction_score     = sum of receptor_score and ligand_score
    """
    # check if data is available
    _check_interactions(anndata)

    table = anndata.uns["receptor-ligand"]["interactions"]

    if min_perc is None:
        min_perc = 0

    # overwrite interaction_score
    if interaction_perc:
        interaction_score = np.nanpercentile(table["interaction_score"], interaction_perc)

    subset = table[
        (True if min_perc is None else table["receptor_percent"] >= min_perc)
        & (True if min_perc is None else table["ligand_percent"] >= min_perc)
        & (True if interaction_score is None else table["interaction_score"] > interaction_score)
    ]

    if group_a and group_b:
        subset = subset[(subset["receptor_cluster"].isin(group_a) & subset["ligand_cluster"].isin(group_b))
                        | (subset["receptor_cluster"].isin(group_b) & subset["ligand_cluster"].isin(group_a))]
    elif group_a or group_b:
        group = group_a if group_a else group_b

        subset = subset[subset["receptor_cluster"].isin(group) | subset["ligand_cluster"].isin(group)]

    if save:
        subset.to_csv(f"{settings.table_dir}/{save}", sep='\t', index=False)

    return subset.copy()


@beartype
def _check_interactions(anndata: sc.AnnData):
    """Return error message if anndata object doesn't contain interaction data."""

    # is interaction table available?
    if "receptor-ligand" not in anndata.uns.keys() or "interactions" not in anndata.uns["receptor-ligand"].keys():
        raise ValueError("Could not find interaction data! Please setup with `calculate_interaction_table(   )` before running this function.")

# --------- DIFFERENCE CALCULATION ----------#


# Receptor-Ligand Difference Analysis

# This section provides functions for receptor-ligand interaction analysis
# between different cell populations across experimental conditions.
# The analysis pipeline:

# 1. Filters the AnnData object based on conditions, clusters, and genes
# 2. Calculates interaction tables between cell clusters
# 3. Compares interaction scores between different conditions
# 4. Processes combinations of conditions for differential analysis
# 5. Stores results in the AnnData object

# The results can be visualized with the plotting functions
# of the subsequent section.


# Function to create a filtered AnnData object
@beartype
def _filter_anndata(
    adata: sc.AnnData,
    condition_values: List[str] | npt.ArrayLike,
    condition_columns: List[str],
    cluster_column: str,
    cluster_filter: Optional[List[str] | npt.ArrayLike] = None,
    gene_column: Optional[str] = None,
    gene_filter: Optional[List[str] | npt.ArrayLike] = None
) -> Optional[sc.AnnData]:
    """Create a filtered AnnData object based on specified conditions.

    Parameters
    ----------
    adata : sc.AnnData
        Input annotated data matrix.
    condition_values : List[str] | npt.ArrayLike
        Values to filter on for each corresponding condition column.
    condition_columns : List[str]
        Column names in adata.obs to filter on.
    cluster_column : str
        Column name containing cluster information.
    cluster_filter : Optional[List[str] | npt.ArrayLike], default None
        Specific clusters to include (includes all if None).
    gene_column : Optional[str], default None
        Column containing gene identifiers (uses index if None).
    gene_filter : Optional[List[str] | npt.ArrayLike], default None
        Specific genes to include (includes all if None).

    Returns
    -------
    Optional[sc.AnnData]
        Filtered AnnData object or None if no matches found.

    Raises
    ------
    ValueError
        If condition lengths don't match, required columns are missing,
        or specified clusters/genes are not found.

    Examples
    --------
    # Returns an adata with T-/B-cells belonging to the treatment condition.
    filtered_data = _filter_anndata(
       adata=adata,
       condition_values=['treatment'],
       condition_columns=['condition'],
       cluster_column='leiden',
       cluster_filter=['T-cell', 'B-cell']
    )
    """
    # Validate and normalize inputs
    if len(condition_values) != len(condition_columns):
        raise ValueError(f"Expected {len(condition_columns)} condition values, got {len(condition_values)}")

    # Create pairs of condition columns and values for filtering
    # Example: If condition_columns=['batch', 'treatment'] and condition_values=['1', 'control'],
    # This creates: [('batch', '1'), ('treatment', 'control')]
    condition_pairs = list(zip(
        condition_columns if condition_columns is not None else [],
        condition_values if condition_values is not None else []
    ))

    # Filter out pairs where value is None
    # Example: If condition_pairs=[('batch', '1'), ('treatment', None)],
    # keeps only: [('batch', '1')]
    valid_pairs = [(col, val) for col, val in condition_pairs if val is not None]
    if not valid_pairs:
        return None

    # Construct and apply query for cell filtering
    # Example: If valid_pairs=[('batch', '1'), ('treatment', 'control')],
    # This creates query: "batch == '1' & treatment == 'control'"
    query = " & ".join([f"{col} == '{val}'" for col, val in valid_pairs])

    # Boolean mask
    cell_mask = adata.obs.eval(query)

    # Check if mask contains only False
    if cell_mask.sum() == 0:
        warnings.warn(f"No cells found for conditions {dict(valid_pairs)}.")
        return None

    filtered = adata[cell_mask].copy()

    # Apply cluster filter if provided
    # Example: If cluster_column='cell_type' and cluster_filter=['T-cell', 'B-cell'],
    # This will keep only cells where cell_type is either 'T-cell' or 'B-cell'
    if cluster_filter is not None:
        cluster_filter = np.unique(cluster_filter)
        valid_clusters = set(cluster_filter).intersection(set(filtered.obs[cluster_column]))

        if not valid_clusters:
            warnings.warn("No valid clusters for the specified conditions.")
            return None

        cluster_mask = filtered.obs[cluster_column].isin(valid_clusters)
        if cluster_mask.sum() == 0:
            warnings.warn("No cells match the cluster filter.")
            return None

        filtered = filtered[cluster_mask].copy()

    # Apply gene filter if provided
    # Example: If gene_filter=['CD4', 'CD8A', 'IL2RA'] and gene_column=None,
    # This will keep only genes with those IDs in the index
    if gene_filter is not None:
        gene_filter = np.unique(gene_filter)

        if gene_column is not None and gene_column not in filtered.var.columns:
            raise ValueError(f"gene_column: {gene_column} not available in adata.var.columns")

        # Get gene source (index or column)
        gene_source = filtered.var.index if gene_column is None else filtered.var[gene_column]
        valid_genes = set(gene_filter).intersection(set(gene_source))

        if not valid_genes:
            warnings.warn("No valid genes for the specified conditions.")
            return None

        # Apply gene mask
        gene_mask = (filtered.var.index if gene_column is None else filtered.var[gene_column]).isin(valid_genes)

        if gene_mask.sum() == 0:
            warnings.warn("No genes match the gene filter.")
            return None

        filtered = filtered[:, gene_mask].copy()

    return filtered


# Function to calculate differences between two AnnData objects
@beartype
def _calculate_condition_difference(
    adata_a: sc.AnnData,
    adata_b: sc.AnnData,
    condition_a_name: str,
    condition_b_name: str,
    min_perc: Optional[int | float] = None,
    interaction_score: Optional[float | int] = None,
    interaction_perc: Optional[float | int] = None
) -> pd.DataFrame:
    """Calculate quantile rank differences between two conditions' interaction scores.

    Parameters
    ----------
    adata_a : sc.AnnData
        AnnData object with cells treated as condition A.
    adata_b : sc.AnnData
        AnnData object with cells treated as condition B.
    condition_a_name : str
        Names of the conditions for column labeling.
    condition_b_name : str
        Names of the conditions for column labeling.
    min_perc : Optional[int | float]
        Minimum percentage of cells expressing the gene (0-100).
    interaction_score : Optional[float | int]
        Minimum interaction score threshold (ignored if interaction_perc is set).
    interaction_perc : Optional[float | int]
        Minimum interaction percentile threshold (0-100, overrides interaction_score).

    Returns
    -------
    pd.DataFrame
        Quantile-ranked interaction differences between conditions.

    Notes
    -----
    Filter parameters (min_perc, interaction_score, interaction_perc) might
    introduce bias in the difference calculation,
    so they are recommended to be disabled (default).

    Examples
    --------
    diff_result = _calculate_condition_difference(
        adata_a=adata_treatment,
        adata_b=adata_control,
        condition_a_name='treatment',
        condition_b_name='control',
        min_perc=10
    )
    diff_result.sort_values('abs_diff_control_vs_treatment', ascending=False).head()
    """

    """
    Get interaction tables and add condition labels
    This extracts receptor-ligand interactions from each condition's AnnData object
    """
    interactions = {
        condition_a_name: get_interactions(
            adata_a, min_perc=min_perc,
            interaction_score=interaction_score,
            interaction_perc=interaction_perc
        ),
        condition_b_name: get_interactions(
            adata_b, min_perc=min_perc,
            interaction_score=interaction_score,
            interaction_perc=interaction_perc
        )
    }

    # Add condition labels and combine for ranking
    # This tags each interaction with its source condition for later identification
    for cond, df in interactions.items():
        df['condition'] = cond

    # Combine datasets for uniform ranking across conditions
    # This is critical for fair comparison - we want to rank all interactions together
    # rather than separately within each condition so that there are no biases due to
    # differences in scale
    combined = pd.concat(interactions.values(), ignore_index=True)

    # Calculate quantile rank (0-1) of interaction scores across all conditions
    # If records have the same rank, they will be ranked by the average rank of
    # this group. See here for more on that:
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html
    combined['quantile_rank'] = combined['interaction_score'].rank(method='average', pct=True)

    # Key columns for merging - used to identify unique receptor-ligand pairs
    key_cols = ['receptor_gene', 'ligand_gene', 'receptor_cluster', 'ligand_cluster']

    # Extract ranks and merge back to original data
    for cond in [condition_a_name, condition_b_name]:
        cond_ranks = combined[combined['condition'] == cond][key_cols + ['quantile_rank']]
        interactions[cond] = pd.merge(interactions[cond], cond_ranks, on=key_cols)

    # Merge the interaction tables with descriptive suffixes
    a_suffix, b_suffix = f"_{condition_a_name}", f"_{condition_b_name}"
    result = pd.merge(
        interactions[condition_a_name],
        interactions[condition_b_name],
        on=key_cols,
        suffixes=(a_suffix, b_suffix)
    )

    # Calculate difference between condition ranks
    # Example: For treatment vs control, this would calculate:
    # 'rank_diff_control_vs_treatment' = quantile_rank_control - quantile_rank_treatment
    diff_col = f'rank_diff_{condition_b_name}_vs_{condition_a_name}'
    abs_diff_col = f'abs_diff_{condition_b_name}_vs_{condition_a_name}'

    result[diff_col] = result[f'quantile_rank{b_suffix}'] - result[f'quantile_rank{a_suffix}']
    result[abs_diff_col] = result[diff_col].abs()

    return result


@beartype
def _process_condition_combinations(
    adata: sc.AnnData,
    condition_columns: List[str],
    cluster_column: str,
    condition_values_dict: Dict[str, List[str] | npt.ArrayLike] = None,
    min_perc: Optional[int | float] = None,
    interaction_score: Optional[float | int] = None,
    interaction_perc: Optional[float | int] = None,
    cluster_filter: Optional[List[str] | npt.ArrayLike] = None,
    gene_column: Optional[str] = None,
    gene_filter: Optional[List[str] | npt.ArrayLike] = None,
    normalize: Optional[int] = None,
    weight_by_ep: bool = True,
    save_diff: bool = False,
    sequential_time_analysis: bool = True,
    layer: Optional[str] = None
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Process and compare combinations of conditions for differential analysis.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated single-cell data matrix.
    condition_columns : List[str]
        Columns in adata.obs for filtering.
    cluster_column : str
        Column containing cluster information.
    condition_values_dict : Dict[str, List[str] | npt.ArrayLike], default None
        Possible values for each condition column.
    min_perc : Optional[int | float], default None
        Minimum percentage of cells in a cluster expressing a gene (0-100).
    interaction_score : Optional[float | int], default None
        Threshold for receptor-ligand interaction scores.
    interaction_perc : Optional[float | int], default None
        Percentile threshold for interactions (0-100, overrides interaction_score).
    cluster_filter : Optional[List[str] | npt.ArrayLike], default None
        Clusters to include in analysis (includes all if None).
    gene_column : Optional[str], default None
        Column for gene IDs (uses index if None).
    gene_filter : Optional[List[str] | npt.ArrayLike], default None
        Genes to include in analysis (includes all if None).
    normalize : Optional[int], default None
        Size for cluster normalization.
    weight_by_ep : bool, default True
        Weight expression by proportion.
    save_diff : bool, default False
        Save difference tables to disk.
    sequential_time_analysis : bool, default True
        If True, only compare sequential timepoints (t₁ vs t₀, t₂ vs t₁, etc.).
    layer : Optional[str], default None
        The layer used for score computation. None to use `adata.X`. It is recommended to use raw or normalized data for statistical analysis.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Comparison results dictionary.

    Examples
    --------
    condition_values = {'treatment': ['A', 'B', 'C'], 'batch': ['1', '2']}
    results = _process_condition_combinations(
        adata=adata,
        condition_columns=['treatment', 'batch'],
        condition_values_dict=condition_values,
        cluster_column='leiden'
    )
    """
    # Check if condition_values_dict is None or empty
    if condition_values_dict is None or not condition_values_dict:
        warnings.warn("Empty condition_values_dict provided. Cannot perform comparisons.")
        return {}

    # The first column is considered the "target" column that we want to compare values from
    # Example: If condition_columns=['treatment', 'batch', 'patient_id'],
    # then 'treatment' is the target condition we're comparing across batches and patient IDs
    target_condition = condition_columns[0]
    target_values = condition_values_dict[target_condition]

    # Check if we have enough values to compare
    if len(target_values) < 2:
        warnings.warn(f"Need at least 2 values for {target_condition} to compare")
        return {}

    # Generate all possible combinations of control conditions
    # Control conditions are all columns after the first (target) column
    # Example: If condition_columns=['treatment', 'batch', 'patient_id'],
    # control_conditions would be ['batch', 'patient_id']
    control_conditions = condition_columns[1:]

    if not control_conditions:
        # If only one condition column, no additional controls
        control_combinations = [{}]
    else:
        # Generate all combinations of values for control conditions
        # Example: If control_conditions=['batch', 'patient_id'] with values {'batch': ['1', '2'], 'patient_id': ['A', 'B']},
        # This creates combinations: [{'batch': '1', 'patient_id': 'A'}, {'batch': '1', 'patient_id': 'B'},
        #                             {'batch': '2', 'patient_id': 'A'}, {'batch': '2', 'patient_id': 'B'}]
        control_values = [condition_values_dict[col] for col in control_conditions]
        control_combinations = [
            dict(zip(control_conditions, combo))
            for combo in itertools.product(*control_values)
        ]

    # Store results
    all_results = {}

    # Process each control combination
    for control_dict in control_combinations:
        # Create a descriptive string for this control combination
        # Example: If control_dict={'batch': '1', 'patient_id': 'A'},
        # This creates "batch=1_patient_id=A
        control_desc = "_".join([f"{col}={val}" for col, val in control_dict.items()])
        filtered_datasets = {}

        # Filter data for each target value
        # This creates a filtered dataset for each target value within the current control combination
        for target_value in target_values:
            # Combine target and control values for filtering
            # Example: If target_value='treatment_A' and control_dict={'batch': '1', 'patient_id': 'A'},
            # filter_values would be ['treatment_A', '1', 'A']
            filter_values = [target_value] + [control_dict.get(col) for col in condition_columns[1:]]
            logger.info(f"Filtering for {list(zip(condition_columns, filter_values))}")

            # Filter the AnnData object
            # This creates a subset of the data matching the current combination of conditions
            filtered = _filter_anndata(
                adata=adata,
                condition_values=filter_values,
                condition_columns=condition_columns,
                cluster_column=cluster_column,
                cluster_filter=cluster_filter,
                gene_column=gene_column,
                gene_filter=gene_filter,
            )

            if filtered is not None:

                # Calculate interaction table for the filtered data
                calculate_interaction_table(
                    adata=filtered,
                    cluster_column=cluster_column,
                    gene_index=gene_column,
                    normalize=normalize,
                    weight_by_ep=weight_by_ep,
                    inplace=True,
                    overwrite=True,
                    layer=layer
                )

                filtered_datasets[target_value] = filtered
                logger.info(f"Filtered {target_condition}={target_value}: {filtered.n_obs} cells")

        # Get valid targets and determine what pairs to compare
        valid_targets = list(filtered_datasets.keys())

        # Determine which pairs to compare
        if sequential_time_analysis and len(valid_targets) > 1:
            # Sort by the order in the original condition_values_dict to respect user-defined order
            ordered_targets = [t for t in condition_values_dict[target_condition] if t in valid_targets]

            # Create pairs of sequential timepoints only
            # Example: For timepoints ['day0', 'day3', 'day7', 'day14'],
            # This creates comparison pairs: [('day0', 'day3'), ('day3', 'day7'), ('day7', 'day14')]
            comparison_pairs = [(ordered_targets[i], ordered_targets[i + 1])
                                for i in range(len(ordered_targets) - 1)]
        else:
            # Compare all combinations (default behavior)
            # Example: For targets ['control', 'treatment_A', 'treatment_B'],
            # This creates comparison pairs: [('control', 'treatment_A'), ('control', 'treatment_B'), ('treatment_A', 'treatment_B')]
            comparison_pairs = list(itertools.combinations(valid_targets, 2))

        # Process each comparison pair
        for value_a, value_b in comparison_pairs:
            # Calculate differences between the two conditions
            # This computes how receptor-ligand interactions differ between the two conditions
            # by calculating the quantile rank differences
            diff_result = _calculate_condition_difference(
                adata_a=filtered_datasets[value_a],
                adata_b=filtered_datasets[value_b],
                condition_a_name=str(value_a),
                condition_b_name=str(value_b),
                min_perc=min_perc,
                interaction_score=interaction_score,
                interaction_perc=interaction_perc
            )

            if not diff_result.empty:
                # Create key and store results
                comp_key = f"{control_desc}_{value_b}_vs_{value_a}" if control_desc else f"{value_b}_vs_{value_a}"

                # Add control info to results
                # This allows tracking what control values were used for this comparison
                for col, val in control_dict.items():
                    diff_result[f"control_{col}"] = val

                # Store and optionally save results
                all_results[comp_key] = {'differences': diff_result}

                if save_diff:
                    diff_result.to_csv(f"{settings.table_dir}/{comp_key}_differences.csv", sep='\t', index=False)

    return all_results


@deco.log_anndata
@beartype
def calculate_condition_differences(
    adata: sc.AnnData,
    condition_columns: List[str],
    cluster_column: str,
    min_perc: Optional[int | float] = None,
    interaction_score: Optional[float | int] = None,
    interaction_perc: Optional[int | float] = None,
    condition_filters: Optional[Dict[str, List[str] | npt.ArrayLike]] = None,
    time_column: Optional[str] = None,
    time_order: Optional[List[str] | npt.ArrayLike] = None,
    gene_column: Optional[str] = None,
    cluster_filter: Optional[List[str] | npt.ArrayLike] = None,
    gene_filter: Optional[List[str] | npt.ArrayLike] = None,
    normalize: Optional[int] = None,
    weight_by_ep: Optional[bool] = True,
    inplace: bool = False,
    overwrite: bool = False,
    save_diff: bool = False,
    layer: Optional[str] = None
) -> Optional[sc.AnnData]:
    """
    Calculate interaction quantile rank differences between conditions.

    Compares values of the first condition within each combination of subsequent conditions.
    Can also analyze conditions over specific timepoints in user-defined order.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix with expression values and metadata.
    condition_columns : List[str]
        Columns in adata.obs for hierarchical filtering, ordered sequentially.
        First column contains values to be compared within combinations of other columns.
        The interaction score differences will be calculated between the values of the first
        column given all combinations of the subsequent condition columns.
    cluster_column : str
        Name of the cluster column in adata.obs.
    min_perc : Optional[int | float], optional
        Minimum percentage of cells in a cluster expressing a gene (0-100).
        Default is None.
    interaction_score : Optional[float | int], optional
        Threshold for filtering receptor-ligand interactions by score.
        Ignored if interaction_perc is set. Default is None.
    interaction_perc : Optional[int | float], optional
        Percentile threshold for filtering receptor-ligand interactions (0-100).
        Overrides interaction_score. Default is None.
    condition_filters : Optional[Dict[str, List[str] | npt.ArrayLike]], optional
        Mapping of condition column names to values to include.
        If None or column not specified, all unique values are used. Default is None.
    time_column : Optional[str], optional
        Column in adata.obs containing timepoint information.
        If provided, analysis will respect time ordering. Default is None.
    time_order : Optional[List[str] | npt.ArrayLike], optional
        Order of timepoints to use in analysis.
        Provide the unique timepoints in the correct order as a list.
        Required if time_column is specified. Default is None.
    gene_column : Optional[str], optional
        Column in adata.var containing gene symbols/IDs.
        Uses index if None. Default is None.
    cluster_filter : Optional[List[str] | npt.ArrayLike], optional
        Clusters to include in analysis.
        If None, all clusters are included. Default is None.
    gene_filter : Optional[List[str] | npt.ArrayLike], optional
        Genes to include in analysis.
        If None, all genes are included. Default is None.
    normalize : Optional[int], optional
        Correct clusters to specified size.
        If None, maximum cluster size is used. Default is None.
    weight_by_ep : Optional[bool], default True
        Whether to weight expression Z-Score by expression proportion.
    inplace : bool, default False
        If True, modifies adata in-place.
        If False, returns a copy of adata with results.
    overwrite : bool, default False
        If True, overwrites existing interaction table.
    save_diff : bool, default False
        Whether to save the differences table.
    layer : Optional[str], default None
        The layer used for score computation. None to use `adata.X`. It is recommended to use raw or normalized data for statistical analysis.

    Returns
    -------
    Optional[sc.AnnData]
        - If inplace=False: Returns a copy of AnnData with condition differences
        - If inplace=True: Returns None, modifies original AnnData

        In both cases, the differences are stored in:
        adata.uns['sctoolbox']['receptor-ligand']['condition-differences']

    Raises
    ------
    ValueError
        - If invalid keys are provided in condition_filters
        - If no valid values exist for a condition after filtering
        - If fewer than one condition column is provided
        - If no valid values match any provided filters

    Examples
    --------
    # Calculate differences between treatment conditions across batches
    diff_adata = calculate_condition_differences(
        adata=adata,
        condition_columns=['treatment', 'batch'],
        cluster_column='leiden',
        min_perc=10,
        condition_filters={'treatment': ['control', 'treatment_A', 'treatment_B']},
        inplace=False
    )

    # Access the results
    results = diff_adata.uns['sctoolbox']['receptor-ligand']['condition-differences']

    # Temporal analysis with timepoints in specific order
    time_series_adata = calculate_condition_differences(
        adata=adata,
        condition_columns=['timepoint', 'treatment'],
        cluster_column='leiden',
        time_column='timepoint',
        time_order=['day0', 'day3', 'day7', 'day14', 'day28'],
        condition_filters={'treatment': ['control', 'drug_A']},
        inplace=False
    )

    # Analyzing a condition over timepoints
    # Note: 'timepoint' as first column with time_column specified
    # will trigger sequential analysis (day3 vs day0, day7 vs day3, etc.)
    time_course_adata = calculate_condition_differences(
        adata=adata,
        condition_columns=['timepoint', 'treatment'],
        cluster_column='leiden',
        time_column='timepoint',
        time_order=['day0', 'day3', 'day7', 'day14', 'day28'],
        condition_filters={'treatment': ['drug_A']},
        inplace=False
    )
    """

    # Create modified_adata at the beginning
    modified_adata = adata if inplace else adata.copy()

    # Check if condition differences already exist
    if not overwrite and in_uns(modified_adata, ['sctoolbox', 'receptor-ligand', 'condition-differences']):
        warnings.warn("Condition differences already exists! Skipping. Set `overwrite=True` to replace.")

        if inplace:
            return None
        else:
            return modified_adata

    # Validate time information if provided
    if time_column is not None:
        if time_column not in modified_adata.obs.columns:
            raise ValueError(f"time_column '{time_column}' not found in adata.obs columns")

        if time_order is None:
            raise ValueError("time_order must be provided when time_column is specified")

        # Check if all provided time points exist in the data
        time_values = set(modified_adata.obs[time_column])
        missing_times = set(time_order) - time_values
        if missing_times:
            raise ValueError(f"The following time points in time_order were not found in the data: {missing_times}")

    # Validate condition filters
    if condition_filters is not None:
        invalid_keys = set(condition_filters.keys()) - set(condition_columns)
        if invalid_keys:
            raise ValueError(f"Invalid keys in condition_filters: {invalid_keys}. Valid keys are: {condition_columns}")

    # Ensure we have at least one condition column
    if len(condition_columns) < 1:
        raise ValueError(f"Need at least one condition column, got {len(condition_columns)}")

    # Convert inputs to numpy arrays for consistent handling
    cluster_filter = list(set(cluster_filter)) if cluster_filter is not None else None
    gene_filter = list(set(gene_filter)) if gene_filter is not None else None

    # Get all possible values for each condition
    condition_values_dict = {}
    for col in condition_columns:
        if (
            condition_filters is not None
            and col in condition_filters
            and condition_filters[col] is not None
            and len(condition_filters[col]) > 0
        ):
            # Get all possible values from the data
            available_values = set(modified_adata.obs[col])
            # Use intersection to get values that both exist in the data AND are in the filter
            valid_values = list(set(condition_filters[col]).intersection(available_values))

            if not valid_values:
                raise ValueError(f"No valid values for condition '{col}'. Available: {available_values}")

            condition_values_dict[col] = valid_values
        else:
            # Use all values
            condition_values_dict[col] = sorted(list(modified_adata.obs[col].unique()))

    # If time_column is provided and it's one of the condition columns, sort the values according to user-provided time_order
    if time_column is not None and time_column in condition_columns:
        # Filter time_order to include only available values in the selected condition
        available_times = set(condition_values_dict[time_column])
        ordered_times = [t for t in time_order if t in available_times]
        # Replace the default ordering with the user-provided order
        condition_values_dict[time_column] = ordered_times

    # Start processing
    logger.info("Starting comparison processing   ")

    # The first column is the target condition to compare
    # Example: If condition_columns=['treatment', 'batch', 'patient_id'],
    # target_condition is 'treatment'
    target_condition = condition_columns[0]

    # Check if this is a time series analysis
    # This is true when the time_column is the first column in condition_columns
    # Example: condition_columns=['timepoint', 'treatment'] with time_column='timepoint'
    is_time_series = time_column is not None and time_column == target_condition

    # Process all condition combinations
    # We pass the condition_values_dict containing the possible values for each condition
    # This will create filtered datasets for each combination and calculate differences
    all_results = {target_condition: _process_condition_combinations(
        adata=modified_adata,
        condition_columns=condition_columns,
        condition_values_dict=condition_values_dict,
        cluster_column=cluster_column,
        min_perc=min_perc,
        interaction_score=interaction_score,
        interaction_perc=interaction_perc,
        cluster_filter=cluster_filter,
        gene_column=gene_column,
        gene_filter=gene_filter,
        normalize=normalize,
        weight_by_ep=weight_by_ep,
        save_diff=save_diff,
        sequential_time_analysis=is_time_series,
        layer=layer
    )}

    # If time series analysis, add time metadata to the results
    if time_column is not None and time_order is not None:
        # Store the time ordering information with the results
        for _, comparison_data in all_results[target_condition].items():
            for _, result_df in comparison_data.items():
                result_df['time_column'] = time_column
                result_df['time_order'] = ','.join(time_order)

    # Check if any comparisons were found
    if len(all_results[target_condition]) == 0:
        warnings.warn("No comparisons found. Check your condition values and filters.")
    else:
        logger.info(f"Completed processing with {len(all_results[target_condition])} comparisons")
        if time_column is not None:
            logger.info(f"Temporal analysis using '{time_column}' with order: {time_order}")

    # Store results in the AnnData object
    add_uns_info(
        modified_adata,
        key=['receptor-ligand', 'condition-differences'],
        value=all_results,
        how="overwrite"
    )

    if inplace:
        return None
    else:
        return modified_adata

# ---------- DIFFERENCE PLOTTING -----------#


# Receptor-Ligand Difference Visualization

# This section provides functions for visualizing receptor-ligand interaction differences
# between different cell populations across experimental conditions and timepoints.


def _identify_hub_networks(
    G: nx.DiGraph,
    hub_threshold: int = 4
) -> tuple[dict[str, nx.DiGraph], list[tuple[str, str]]]:
    """Identify hub nodes and create hub-centric subgraphs.

    Parameters
    ----------
    G : nx.DiGraph
        The input directed graph.
    hub_threshold : int, default 4
        Minimum number of connections for a node to be considered a hub.

    Returns
    -------
    tuple[dict[str, nx.DiGraph], list[tuple[str, str]]]
        A tuple containing:
        - hub_networks: Dictionary of hub-centered subgraphs keyed by hub node.
        - non_hub_edges: List of edges between non-hub nodes.

    Examples
    --------
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), ('B', 'C')])
    hub_networks, non_hub_edges = _identify_hub_networks(G, hub_threshold=3)
    """
    # Use degree function to get node connections
    node_degrees = dict(G.degree())

    # Identify hub nodes (nodes with connections meeting or exceeding the threshold)
    hub_nodes = {node for node, degree in node_degrees.items() if degree >= hub_threshold}

    """
    Create hub-centered subgraphs
    For each hub, extract its neighborhood to visualize its distinct interaction network
    """
    hub_networks = {}
    for hub in hub_nodes:
        # Get all neighbors of the hub
        neighbors = set(G.successors(hub)).union(set(G.predecessors(hub)))
        # Create a subgraph with the hub and its neighbors
        nodes_in_subgraph = {hub}.union(neighbors)
        subgraph = G.subgraph(nodes_in_subgraph).copy()
        hub_networks[hub] = subgraph

    # Identify edges between non-hub nodes
    non_hub_edges = [(u, v) for u, v in G.edges()
                     if u not in hub_nodes and v not in hub_nodes]

    return hub_networks, non_hub_edges


def _draw_network(graph: nx.DiGraph,
                  ax: plt.Axes,
                  title: Optional[str] = None,
                  cell_colors: Optional[dict[str, str]] = None,
                  cell_shapes: Optional[dict[str, str]] = None,
                  colormap: Optional[plt.cm.ScalarMappable] = None,
                  norm: Optional[plt.Normalize] = None,
                  all_cell_types: Optional[set[str]] = None) -> None:
    """
    Draw a network on the given axis.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph to draw
    ax : matplotlib.axes.Axes
        The axis to draw on
    title : str, optional
        Title for the plot
    cell_colors : dict
        Mapping of cell type to color
    cell_shapes : dict
        Mapping of cell type to node shape
    colormap : matplotlib.colors.Colormap
        Colormap for edge colors
    norm : matplotlib.colors.Normalize
        Normalization for edge colors
    all_cell_types : set
        Set of all cell types to ensure consistent drawing

    Returns
    -------
    None
        This function modifies the provided axis in-place

    Examples
    --------
    # Create simple test graph
    G = nx.DiGraph()
    G.add_node('A_cluster1', cell_type='cluster1', gene='A')
    G.add_node('B_cluster2', cell_type='cluster2', gene='B')
    G.add_edge('A_cluster1', 'B_cluster2', diff=0.5)

    # Setup for drawing
    cell_colors = {'cluster1': 'blue', 'cluster2': 'red'}
    cell_shapes = {'cluster1': 'o', 'cluster2': 's'}
    colormap = plt.cm.Reds
    norm = Normalize(vmin=0, vmax=1)

    # Create figure and draw
    fig, ax = plt.subplots()
    _draw_network(G, ax, title='Test Network',
                    cell_colors=cell_colors, cell_shapes=cell_shapes,
                    colormap=colormap, norm=norm, all_cell_types={'cluster1', 'cluster2'})
    plt.close()
    """
    # Skip if graph is empty
    if len(graph.nodes()) == 0:
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=12)
        return

    # Create layout for this network
    pos = nx.spring_layout(graph, k=2.0, iterations=333, seed=42, scale=0.9)

    # Group nodes by cell type for efficient drawing
    node_groups = {}
    for node, data in graph.nodes(data=True):
        cell_type = data.get('cell_type')
        if cell_type:
            if cell_type not in node_groups:
                node_groups[cell_type] = []
            node_groups[cell_type].append(node)

    # Draw all nodes for each cell type at once
    for cell_type in sorted(all_cell_types):
        # Get all nodes for this cluster
        ct_nodes = node_groups.get(cell_type, [])

        # Draw all nodes for this cluster with the same shape and color
        if ct_nodes:
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=ct_nodes,
                node_color=[cell_colors[cell_type]] * len(ct_nodes),
                node_size=700,
                alpha=0.8,
                edgecolors='lightgrey',
                linewidths=0.5,
                node_shape=cell_shapes[cell_type],
                ax=ax
            )

    # Create edge colors and labels for drawing
    edge_colors = []
    edge_list = []
    edge_labels = {}

    for u, v, data in graph.edges(data=True):
        edge_list.append((u, v))
        # Map the difference value to a color using the provided colormap and normalization
        edge_colors.append(colormap(norm(data['diff'])))
        # Create edge label showing the difference value with 2 decimal precision
        edge_labels[(u, v)] = f"{data['diff']:.2f}"

    # Draw all edges at once
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=edge_list,
        width=1.5,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=15,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )

    # Draw edge labels
    nx.draw_networkx_edge_labels(
        graph, pos,
        edge_labels=edge_labels,
        font_size=8,
        bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', boxstyle='round,pad=0.2'),
        ax=ax
    )

    # Extract node labels using the gene name as the label
    labels = {node: data.get('gene', node) for node, data in graph.nodes(data=True)}

    # Draw labels
    nx.draw_networkx_labels(
        graph, pos,
        labels=labels,
        font_size=10,
        font_weight='normal',
        bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', boxstyle='round,pad=0.2'),
        ax=ax
    )

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=12)

    ax.set_axis_off()


def _extract_diff_key_columns(
    diff_df: pd.DataFrame
) -> dict[str, str | list[str]]:
    """Extract standardized key column names from the differences DataFrame.

    Parameters
    ----------
    diff_df : pd.DataFrame
        Differences DataFrame from condition_differences calculation.

    Returns
    -------
    dict[str, str | list[str]]
        Dictionary with standardized column names mapping.

    Examples
    --------
    # Example with standard column names
    df1 = pd.DataFrame({
        'rank_diff_B_vs_A': [0.5],
        'abs_diff_B_vs_A': [0.5]
    })
    col_info = _extract_diff_key_columns(df1)
    # Example with time column info
    df2 = pd.DataFrame({
        'rank_diff_day7_vs_day0': [0.5],
        'abs_diff_day7_vs_day0': [0.5],
        'time_column': ['timepoint'],
        'time_order': ['day0,day3,day7,day14']
    })
    time_info = _extract_diff_key_columns(df2)
    """
    # Initialize the result dictionary
    result = {}

    # Find the rank difference column (it should start with 'rank_diff_')
    diff_cols = [col for col in diff_df.columns if col.startswith('rank_diff_')]
    abs_diff_cols = [col for col in diff_df.columns if col.startswith('abs_diff_')]

    if diff_cols and abs_diff_cols:
        diff_col = diff_cols[0]
        abs_diff_col = abs_diff_cols[0]

        # Extract condition names from the column name. Format is typically 'rank_diff_conditionB_vs_conditionA'
        parts = diff_col.split('_')
        if len(parts) >= 4 and parts[-2] == 'vs':
            result['condition_a'] = parts[-1]
            result['condition_b'] = parts[-3]
            result['diff_col'] = diff_col
            result['abs_diff_col'] = abs_diff_col

    # Check for time series information
    if 'time_column' in diff_df.columns and not diff_df['time_column'].isna().all():
        time_col = diff_df['time_column'].iloc[0]
        result['time_column'] = time_col

    if 'time_order' in diff_df.columns and not diff_df['time_order'].isna().all():
        # Convert comma-separated string to list
        time_order_str = diff_df['time_order'].iloc[0]
        if isinstance(time_order_str, str):
            result['time_order'] = time_order_str.split(',')

    return result


@beartype
def _format_control_conditions(diff_df: pd.DataFrame) -> str:
    """Extract and format control conditions from a differences DataFrame.

    Processes control columns (prefixed with 'control_') from a receptor-ligand
    differences DataFrame and formats them into a string for
    display in plot main title.

    Parameters
    ----------
    diff_df : pd.DataFrame
        Differences DataFrame from calculate_condition_differences containing
        columns prefixed with 'control_' that specify the values of control
        conditions used during the comparison. Expected columns include:
        - control_timepoint: timepoint identifier (if timepoint was a control condition)
        - Additional control_* columns for other controlled conditions

    Returns
    -------
    str
        Formatted string containing control conditions in the format:
        " | condition1: value1, condition2: value2"
        Returns empty string if no control columns are found.
    """
    control_cols = [col for col in diff_df.columns
                    if col.startswith('control_')]

    if not control_cols:
        return ""

    conditions = []
    for col in control_cols:
        value = diff_df[col].iloc[0]
        name = col.replace('control_', '')
        conditions.append(f"{name}: {value}")

    return f" | {', '.join(conditions)}"


@beartype
def condition_differences_network(
    adata: sc.AnnData,
    n_top: int = 100,
    figsize: Tuple[int | float, int | float] = (22, 16),
    dpi: Optional[int | float] = None,
    save: Optional[str] = None,
    split_by_direction: bool = True,
    hub_threshold: int = 4,
    color_palette: str = 'tab20',
    vmin: float = -1.0,
    vmax: float = 1.0,
    n_cols: int = 4,
    n_rows: Optional[int] = None,
    non_hub_cols: Optional[int] = None,
    close_figs: bool = True,
    report: Optional[Tuple[str, str]] = None
) -> List[matplotlib.figure.Figure]:
    """
    Visualize differences between conditions as a receptor-ligand network with hubs separated.

    Creates a multi-grid visualization where network hubs are displayed separately.
    Works with both standard condition differences and time series analyses.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing results from calculate_condition_differences
    n_top : int, default 100
        Number of top differential interactions to display
    figsize : Tuple[int | float, int | float], default (22, 16)
        Size of the figure
    dpi : Optional[int | float], default None
        The resolution of the figure. Overwrites `sctoolbox.settings.dpi` and `sctoolbox.settings.report_dpi`.
    save : Optional[str], default None
        Output filename base. Uses the internal 'sctoolbox.settings.figure_dir'
    split_by_direction : bool, default True
        Whether to create separate networks for positive and negative differences
    hub_threshold : int, default 4
        Minimum number of connections for a node to be considered a hub
    color_palette : str, default 'tab20'
        Color palette to use for cell types
    vmin : float, default -1.0
        Minimum value for colormap normalization
    vmax : float, default 1.0
        Maximum value for colormap normalization
    n_cols : int, default 4
        Number of columns in the grid layout
    n_rows : Optional[int], default None
        Number of rows in the grid layout, calculated automatically if None
    non_hub_cols : Optional[int], default None
        Number of columns for the non-hub network. If None, then uses all available columns (n_cols - 1)
    close_figs : bool, default True
        Whether to close figures after saving to free up memory
    report : Optional[Tuple[str, str]]
        Tuple of file prefix and filextension e.g. ('01_network', '.png') used to construct the report files.
        Will be silently skipped if `sctoolbox.settings.report_dir` is None.

    Returns
    -------
    List[matplotlib.figure.Figure]
        List of generated figures. If split_by_direction=True, returns up to two figures per comparison
        (one for positive and one for negative differences). If split_by_direction=False, returns one
        figure per comparison with all differences combined. Note: if close_figs=True, these figures
        will be closed after saving.

    Raises
    ------
    ValueError
        If no valid differences are found in the provided AnnData object.

    Examples
    --------
    .. plot::
        :context: close-figs

        # Create differential interaction network plot
        figs = rl.condition_differences_network(
            adata=adata_diff,
            n_top=30,
            figsize=(16, 12),
            hub_threshold=3,
            close_figs=False
        )
    """
    # Get condition differences from adata
    diff_results = adata.uns.get('sctoolbox', {}).get('receptor-ligand', {}).get("condition-differences", {})

    # Check if data is available
    if not diff_results:
        raise ValueError("No valid condition differences found in the AnnData object.")

    # List to store all generated figures
    figures = []

    # Collect all cell types across all comparisons
    # This ensures consistent colors and shapes across all visualizations
    all_cell_types = set()
    for dimension_results in diff_results.values():
        for comparison_data in dimension_results.values():
            # Skip non-dictionary entries or entries without differences
            if not isinstance(comparison_data, dict) or 'differences' not in comparison_data:
                continue

            diff_df = comparison_data['differences']

            # Skip empty DataFrames
            if len(diff_df) == 0:
                continue

            # Add cell types to the set
            all_cell_types.update(diff_df['receptor_cluster'].unique())
            all_cell_types.update(diff_df['ligand_cluster'].unique())

    # If no cell types were found, raise an error
    if not all_cell_types:
        raise ValueError("No valid cell types found in the differences data.")

    # Get colormap for cell types
    cmap = plt.colormaps.get_cmap(color_palette)

    # Define node shapes
    cluster_shapes = ['o', 's', 'p', '^', 'D', 'v', '<', '>', '8', 'h', 'H', 'd', 'P', 'X']

    # Create a mapping of cell type to (color, shape) pairs
    cell_colors = {}
    cell_shapes = {}

    # Create pairs of (color, shape) combinations efficiently
    sorted_cell_types = sorted(all_cell_types)
    for i, ct in enumerate(sorted_cell_types):
        color_idx = i % cmap.N
        shape_idx = i // cmap.N % len(cluster_shapes)

        cell_colors[ct] = cmap(color_idx)
        cell_shapes[ct] = cluster_shapes[shape_idx]

    # Define colormap settings based on direction
    # These settings determine how the edge colors are displayed:
    #  - positive differences: red scale (higher in condition B)
    #  - negative differences: blue scale (higher in condition A)
    #  - all differences: blue-to-red diverging scale
    colormap_settings = {
        'positive': {'vmin': 0, 'vmax': vmax, 'cmap': 'Reds'},
        'negative': {'vmin': vmin, 'vmax': 0, 'cmap': 'Blues_r'},
        'all': {'vmin': vmin, 'vmax': vmax, 'cmap': 'RdBu_r'}
    }

    # Process each condition dimension
    for i, (dimension_key, dimension_results) in enumerate(diff_results.items()):

        # Process each comparison in this dimension
        for j, (comparison_key, comparison_data) in enumerate(dimension_results.items(), start=i):
            # Skip non-dictionary entries or entries without differences
            if not isinstance(comparison_data, dict) or 'differences' not in comparison_data:
                continue

            diff_df = comparison_data['differences']

            # Skip if empty
            if len(diff_df) == 0:
                continue

            # Extract key column information
            col_info = _extract_diff_key_columns(diff_df)
            if not col_info or 'diff_col' not in col_info:
                logger.warning(f"Could not identify difference columns for {comparison_key}")
                continue

            # Use extracted column names
            diff_col = col_info.get('diff_col')
            abs_diff_col = col_info.get('abs_diff_col')
            condition_a = col_info.get('condition_a', 'Condition A')
            condition_b = col_info.get('condition_b', 'Condition B')

            # Determine directions to plot
            # This section handles whether to create separate plots for:
            #  - positive differences (higher in condition B)
            #  - negative differences (higher in condition A)
            #  - or a single plot with all differences
            directions = []
            if split_by_direction:
                # Get top positive differences (higher in condition B)
                pos_mask = diff_df[diff_col] > 0
                if pos_mask.any():
                    pos_diff = diff_df[pos_mask].sort_values(diff_col, ascending=False).head(n_top)
                    directions.append(('positive', pos_diff, f'Higher in {condition_b}'))

                # Get top negative differences (higher in condition A)
                neg_mask = diff_df[diff_col] < 0
                if neg_mask.any():
                    neg_diff = diff_df[neg_mask].sort_values(diff_col, ascending=True).head(n_top)
                    directions.append(('negative', neg_diff, f'Higher in {condition_a}'))
            else:
                # Get top absolute differences
                top_diff = diff_df.sort_values(abs_diff_col, ascending=False).head(n_top)
                if len(top_diff) > 0:
                    directions.append(('all', top_diff, 'All differences'))

            # Skip if no directions to plot
            if not directions:
                continue

            # Create visualizations for each direction
            for direction_name, top_diff, direction_label in directions:
                # Create a graph
                G = nx.DiGraph()

                # Prepare node and edge data for efficient graph creation
                receptor_nodes = {}
                ligand_nodes = {}
                edges = []

                # Use correct column names based on extracted info
                for _, row in top_diff.iterrows():
                    # Node identifiers
                    r_node = f"{row['receptor_gene']}_{row['receptor_cluster']}"
                    l_node = f"{row['ligand_gene']}_{row['ligand_cluster']}"

                    # Collect node data
                    receptor_nodes[r_node] = {
                        'cell_type': row['receptor_cluster'],
                        'gene': row['receptor_gene'],
                        'is_receptor': True
                    }

                    ligand_nodes[l_node] = {
                        'cell_type': row['ligand_cluster'],
                        'gene': row['ligand_gene'],
                        'is_receptor': False
                    }

                    # Derive interaction score columns based on condition names
                    interaction_a_col = next((col for col in row.index if col.startswith('interaction_score_')
                                             and col.endswith(f"_{condition_a}")), None)
                    interaction_b_col = next((col for col in row.index if col.startswith('interaction_score_')
                                             and col.endswith(f"_{condition_b}")), None)

                    interaction_a = row[interaction_a_col] if interaction_a_col else None
                    interaction_b = row[interaction_b_col] if interaction_b_col else None

                    # The edge represents the ligand-receptor interaction with its difference score
                    # Direction is from ligand to receptor (l_node → r_node)
                    edges.append((l_node, r_node, {
                        'weight': abs(row[diff_col]),
                        'diff': row[diff_col],
                        'interaction_a': interaction_a,
                        'interaction_b': interaction_b
                    }))

                # Skip if no edges
                if not edges:
                    continue

                # Add all nodes and edges at once
                G.add_nodes_from(receptor_nodes.items())
                G.add_nodes_from(ligand_nodes.items())
                G.add_edges_from(edges)

                # Skip if graph is empty
                if len(G.nodes()) == 0:
                    continue

                # Identify hub nodes and their networks which are genes with high number of connections
                hub_networks, non_hub_edges = _identify_hub_networks(G, hub_threshold)

                # Create non-hub subgraph if needed
                non_hub_subgraph = None
                if non_hub_edges:
                    non_hub_subgraph = G.edge_subgraph(non_hub_edges).copy()

                # Determine layout for the multi-grid visualization
                num_hubs = len(hub_networks)
                has_non_hub = non_hub_subgraph is not None and len(non_hub_subgraph.edges) > 0

                # Calculate grid layout parameters
                # The layout accommodates:
                #  1. Non-hub network (if exists) - in the first row (stretching over defined or all cols)
                #  2. Hub networks - each in its own subplot
                #  3. Cluster Legend - in the last column
                #  4. At the bottom of the plots is
                #     4.1. The colorbar for the quantile differences
                #     4.2. A legend that describes that the graphs are directed from ligand to receptor

                # Calculate number of networks for layout
                # If non-hub exists, it uses space equivalent to non_hub_cols
                # Default non_hub_cols to 3 if None (equivalent to all content columns in a 4-column grid)
                non_hub_grid = (non_hub_cols if non_hub_cols is not None else 3) if has_non_hub else 0
                number_of_networks = num_hubs + non_hub_grid

                # Calculate columns (including legend column)
                # For 0-2 networks: use number of networks + 1 columns (for legend)
                # For 3+ networks: use 4 columns total
                cols = min(number_of_networks + 1, 4)

                # Calculate rows
                if number_of_networks <= 2:
                    # For 0-2 networks: always 1 row
                    rows = 1
                elif number_of_networks == 3:
                    # For 3 networks: use 2 rows, 3 columns (2×2 + 1 legend column)
                    rows, cols = 2, 3
                else:
                    # For 4+ networks: use (cols-1) for calculating required rows
                    rows = math.ceil(number_of_networks / (cols - 1))

                # Apply user overrides
                if n_rows is not None:
                    rows = n_rows
                if n_cols is not None:
                    cols = n_cols

                # Close any existing figures to free memory if requested
                if close_figs:
                    plt.close('all')

                # Create figure with GridSpec for multi-grid layout
                fig = plt.figure(figsize=figsize, dpi=dpi if dpi else settings.dpi)

                # Create grid with the last column for legend and make it half size
                # This allocates more space for network visualizations and less for the legend
                width_ratios = [1] * (cols - 1) + [0.5]

                # Create the grid with correct spacing
                gs = gridspec.GridSpec(
                    rows, cols,
                    figure=fig,
                    width_ratios=width_ratios,
                    wspace=0.3, hspace=0.35
                )

                # Set colormap parameters based on direction
                cm_settings = colormap_settings[direction_name]
                norm = mcolors.Normalize(vmin=cm_settings['vmin'], vmax=cm_settings['vmax'])
                colormap = plt.colormaps.get_cmap(cm_settings['cmap'])

                # Prepare the list of subgraphs to plot
                plots_to_display = []

                # Add non-hub subgraph
                if non_hub_subgraph:
                    plots_to_display.append(('non_hub', non_hub_subgraph, "Non-Hub Network"))

                # Add hub subgraphs
                for hub, subgraph in hub_networks.items():
                    plots_to_display.append((hub, subgraph, f"Hub: {hub}"))

                # Tracking grid positions for dynamic allocation
                current_row, current_col = 0, 0

                # Draw each plot in its position
                # This loop places each network in the respective grid cell
                for i, (plot_id, graph, title) in enumerate(plots_to_display):
                    # 2x1 grid for non-hub
                    if plot_id == 'non_hub':
                        # Check if there's enough space to use a colsx1 block for non-hub
                        if current_col + 1 >= (cols - 1):
                            current_col = 0
                            current_row += 1

                        # Non-hub gets non_hub_cols columns according to the non hub grid
                        ax = fig.add_subplot(gs[current_row, current_col:current_col + (non_hub_grid)])

                        # Move to next position
                        current_col += non_hub_grid

                        # If this row is filled with subplots, move to next row
                        if current_col >= (cols - 1):
                            current_col = 0
                            current_row += 1

                    else:
                        # If the end of a row is reached, move to the next row
                        if current_col >= (cols - 1):
                            current_col = 0
                            current_row += 1

                        ax = fig.add_subplot(gs[current_row, current_col])
                        current_col += 1

                    # Draw the network in the current subplot
                    _draw_network(
                        graph, ax,
                        title=title,
                        cell_colors=cell_colors,
                        cell_shapes=cell_shapes,
                        colormap=colormap,
                        norm=norm,
                        all_cell_types=all_cell_types
                    )

                # Add legend to the final column (cluster legend)
                ax_legend = fig.add_subplot(gs[:, -1])
                ax_legend.axis('off')

                # Create proxy handles for legend if which each represents a cell type
                legend_elements = []
                for ct in sorted_cell_types:
                    shape = cell_shapes[ct]
                    color = cell_colors[ct]
                    legend_elements.append(
                        matplotlib.lines.Line2D(
                            [], [], marker=shape, color='w', label=ct,
                            markerfacecolor=color, markersize=10,
                            markeredgecolor='lightgrey', linestyle='None'
                        )
                    )

                # Draw the legend
                ax_legend.legend(
                    handles=legend_elements,
                    loc='center left',
                    title="Cell Types",
                    fontsize=10,
                    title_fontsize=12,
                    handlelength=1.5,
                    handletextpad=0.6,
                    labelspacing=1.0,
                    borderpad=1,
                    ncol=1
                )

                # Extract control conditions
                control_info = _format_control_conditions(diff_df)

                # Build title components
                hub_info = ""
                if hub_networks:
                    hub_info = f" | {len(hub_networks)} Hubs (≥{hub_threshold} connections)"

                # Assemble final title
                main_title = (
                    f"{condition_b} - {condition_a}"
                    f"{control_info} | {direction_label}{hub_info}"
                )

                fig.suptitle(main_title, fontsize=16, y=0.98)
                plt.subplots_adjust(bottom=0.15)

                # Set colorbar title based on direction
                if direction_name == 'positive':
                    colorbar_title = f'Quantile rank differences: Higher in {condition_b}'
                elif direction_name == 'negative':
                    colorbar_title = f'Quantile rank differences: Higher in {condition_a}'
                else:
                    colorbar_title = 'Quantile rank differences'

                # Add colorbar for edge colors at the bottom of the figure
                sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
                sm.set_array([])
                cbar_ax = fig.add_axes([0.35, 0.06, 0.3, 0.02])
                cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal", shrink=0.6, pad=0.15)
                cbar.set_label(colorbar_title, fontsize=12)
                cbar.ax.tick_params(labelsize=10)

                # Add arrow legend to explain edge direction
                arrow_ax = fig.add_axes([0.15, 0.06, 0.25, 0.02])
                arrow_ax.axis('off')

                # Draw arrow example
                arrow_ax.add_artist(lines.Line2D(
                    [0.2, 0.25], [0.5, 0.5],
                    color='black',
                    linewidth=2,
                    transform=arrow_ax.transAxes
                ))

                arrow_ax.add_artist(lines.Line2D(
                    [0.25], [0.5],
                    marker='>',
                    markersize=10,
                    color='black',
                    transform=arrow_ax.transAxes
                ))

                arrow_ax.text(
                    0.30, 0.5,
                    "Edge Direction:\nLigand → Receptor",
                    fontsize=12,
                    ha='left',
                    va='center',
                    transform=arrow_ax.transAxes
                )

                # Save if requested
                _save_figure(f"{save}_{dimension_key}_{comparison_key}_{direction_name}.png", dpi=dpi)

                # report
                if settings.report_dir and report:
                    prefix, rest = report[0].split("_")  # add number to prefix to handle multiplot output
                    _save_figure(f"{prefix}{j:02d}_{rest}_{dimension_key}_{comparison_key}_{direction_name}{report[1]}", report=True, dpi=dpi)

                # Store the figure for return
                figures.append(fig)

    # Close all figures if requested to free memory
    if close_figs:
        for fig in figures:
            plt.close(fig)

    # Return the list of figures
    return figures


@beartype
def plot_all_condition_differences(
    adata: sc.AnnData,
    n_top: int = 100,
    figsize: Tuple[int | float, int | float] = (22, 16),
    dpi: Optional[int | float] = None,
    save_prefix: Optional[str] = None,
    split_by_direction: bool = True,
    hub_threshold: int = 4,
    color_palette: str = 'tab20',
    vmin: float = -1.0,
    vmax: float = 1.0,
    n_cols: int = 4,
    n_rows: Optional[int] = None,
    show: bool = True,
    return_figures: bool = False,
    close_figs: bool = True,
    report: Optional[Tuple[str, str]] = None
) -> Optional[Dict[str, List[matplotlib.figure.Figure]]]:
    """Generate network plots for all condition difference comparisons.

    Process all dimensions in the condition differences data
    and generate visualizations for each comparison.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing condition differences data.
    n_top : int, default 100
        Number of top differential interactions to display.
    figsize : Tuple[int | float, int | float], default (22, 16)
        Size of the figure.
    dpi : Optional[int | float], default None
        Resolution of the figure. Overwrites `sctoolbox.settings.dpi` and `sctoolbox.settings.report_dpi`.
    save_prefix : Optional[str], default None
        Prefix for saved image filenames.
    split_by_direction : bool, default True
        Create separate plots for positive and negative differences.
    hub_threshold : int, default 4
        Minimum connections for a node to be considered a hub.
    color_palette : str, default 'tab20'
        Matplotlib colormap to use for cell types.
    vmin : float, default -1.0
        Minimum value for color scale limits.
    vmax : float, default 1.0
        Maximum value for color scale limits.
    n_cols : int, default 4
        Number of columns in the grid layout.
    n_rows : Optional[int], default None
        Number of rows in the grid layout (calculated if None).
    show : bool, default True
        Display the figures after generating.
    return_figures : bool, default False
        Return the generated figures.
    close_figs : bool, default True
        Close figures after processing to free memory.
    report : Optional[Tuple[str, str]]
        Tuple of file prefix and filextension e.g. ('01_network', '.png') used to construct the report files.
        Will be silently skipped if `sctoolbox.settings.report_dir` is None.

    Returns
    -------
    Optional[Dict[str, List[matplotlib.figure.Figure]]]
        Dictionary of figures by dimension key if return_figures=True. For each comparison:
        - If split_by_direction=True: Up to two figures (positive and negative differences)
        - If split_by_direction=False: One figure with all differences combined
        Returns None if return_figures=False.

    Raises
    ------
    ValueError
        If no condition differences are found in the AnnData object.

    Examples
    --------
    .. plot::
        :context: close-figs

        # Generate all network plots at once
        all_figures = rl.plot_all_condition_differences(
            adata=time_diff,
            n_top=25,
            figsize=(18, 14),
            split_by_direction=True,
            hub_threshold=3,
            close_figs=False
        )
    """
    # Get condition differences from the AnnData object
    diff_results = adata.uns.get('sctoolbox', {}).get('receptor-ligand', {}).get("condition-differences", {})
    if not diff_results:
        raise ValueError("No condition differences found in the AnnData object.")

    # Initialize dictionary to store figures if requested
    all_figures = {} if return_figures else None

    # Process each dimension separately
    for dimension_key, dimension_results in diff_results.items():
        if not dimension_results:
            continue

        # Create a temporary AnnData with just this dimension's results
        temp_adata = sc.AnnData()
        temp_adata.uns['sctoolbox'] = {
            'receptor-ligand': {'condition-differences': {dimension_key: dimension_results}}
        }

        # Generate figures for this dimension
        figures = condition_differences_network(
            adata=temp_adata,
            n_top=n_top,
            figsize=figsize,
            dpi=dpi,
            save=save_prefix,
            split_by_direction=split_by_direction,
            hub_threshold=hub_threshold,
            color_palette=color_palette,
            vmin=vmin,
            vmax=vmax,
            n_cols=n_cols,
            n_rows=n_rows,
            close_figs=close_figs and not show,
            report=report
        )

        # Store figures if requested
        if return_figures:
            all_figures[dimension_key] = figures

        # Show figures if requested
        if show:
            for fig in figures:
                plt.figure(fig.number)
                plt.show()

    return all_figures if return_figures else None


@beartype
def _get_gene_expression(
    adata: sc.AnnData,
    gene: str,
    cluster: str,
    timepoint: str,
    timepoint_col: str,
    cluster_col: str,
    layer: Optional[str] = None
) -> float:
    """
    Get mean expression of a gene in a specific cluster at a specific timepoint.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing gene expression data.
    gene : str
        Gene name or identifier to retrieve expression for.
    cluster : str
        Cluster identifier to filter cells by.
    timepoint : str
        Timepoint identifier to filter cells by.
    timepoint_col : str
        Column name in adata.obs that contains timepoint information.
    cluster_col : str
        Column name in adata.obs that contains cluster information.
    layer : Optional[str], default None
        The layer used for score computation. None to use `adata.X`. It is recommended to use raw or normalized data for statistical analysis.

    Returns
    -------
    float
        Mean expression value of the gene in the specified cluster at the specified timepoint.
        Returns 0.0 if no cells match the criteria or if the gene is not found.

    Notes
    -----
    - If no cells are found that match the timepoint and cluster, returns 0.0.
    - If the gene is not found in the dataset, returns 0.0.
    - A warning is issued when 0.0 is returned, specifying the reason.
    """
    # Find cells that match the timepoint and cluster
    mask = (adata.obs[timepoint_col] == timepoint) & (adata.obs[cluster_col] == cluster)
    if not mask.any() or gene not in adata.var_names:
        warnings.warn(f"No cells found in cluster '{cluster}' at timepoint '{timepoint}'. Returning 0.0.")
        return 0.0

    # Get gene index and cell indices
    gene_idx = adata.var_names.get_loc(gene)
    cell_indices = np.where(mask)[0]

    if len(cell_indices) == 0:
        warnings.warn(f"No cells found in cluster '{cluster}' at timepoint '{timepoint}'. Returning 0.0.")
        return 0.0

    # Get expression values
    if layer:
        expr_values = adata.layers[layer][cell_indices, gene_idx]
    else:
        expr_values = adata.X[cell_indices, gene_idx]

    if hasattr(expr_values, "toarray"):
        expr_values = expr_values.toarray().flatten()

    return float(np.mean(expr_values))


@beartype
def plot_interaction_timeline(
    adata: sc.AnnData,
    interactions: List[Tuple[str, str, str, str]],
    timepoint_column: str,
    cluster_column: str,
    time_order: List[str] | npt.ArrayLike,
    figsize: Optional[Tuple[int | float, int | float]] = None,
    dpi: Optional[int | float] = None,
    save: Optional[str] = None,
    title: Optional[str] = None,
    n_cols: int = 2,
    receptor_color: Optional[str] = None,
    ligand_color: Optional[str] = None,
    use_global_ylim: bool = False,
    layer: Optional[str] = None,
    report: Optional[str] = None
) -> matplotlib.figure.Figure:
    """
    Plot receptor-ligand interaction expression levels over time as barplots.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing the interaction table.
    interactions : List[Tuple[str, str, str, str]]
        List of (receptor_gene, receptor_cluster, ligand_gene, ligand_cluster) tuples.
    timepoint_column : str
        Column in adata.obs containing timepoints.
    cluster_column : str
        Column in adata.obs containing cluster names.
    time_order : List[str] | npt.ArrayLike
        Order of timepoints to use in analysis.
        Provide the unique timepoints in the correct order as a list.
        All timepoints must exist in the data.
    figsize : Optional[Tuple[int | float, int | float]], default None
        Figure dimensions in inches.
    dpi : Optional[int | float], default None
        Figure resolution. Overwrites `sctoolbox.settings.dpi` and `sctoolbox.settings.report_dpi`.
    save : Optional[str], default None
        Output filename.
    title : Optional[str], default None
        Overall figure title.
    n_cols : int, default 2
        Number of columns in the grid layout.
    receptor_color : Optional[str], default None
        Color for receptor bars. If None, uses the first color from seaborn's default palette.
    ligand_color : Optional[str], default None
        Color for ligand bars. If None, uses the second color from seaborn's default palette.
    use_global_ylim : bool, default False
        Whether to use the same y-limit for all subplots based on the global maximum.
    layer : Optional[str], default None
        The layer used. None to use `adata.X`. It is recommended to use raw or normalized data for statistical analysis.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Raises
    ------
    ValueError
        If the specified timepoint or cluster columns don't exist in adata.obs.
        If no receptor-ligand interaction data is found in adata.
        If none of the specified interactions are found in the data.
        If time_order is not provided.
        If any timepoint in time_order doesn't exist in the data.

    Examples
    --------
    .. plot::
        :context: close-figs

        # Plot interactions over time
        fig = rl.plot_interaction_timeline(
            adata=adata,
            interactions=interaction_pairs,
            timepoint_column='timepoint',
            cluster_column='louvain',
            time_order=['Day0', 'Day3', 'Day7'],
            figsize=(16, 6),
            title="Top Interactions Over Time"
        )
    """
    if timepoint_column not in adata.obs.columns:
        raise ValueError(f"Timepoint column '{timepoint_column}' not found in adata.obs")

    if cluster_column not in adata.obs.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found in adata.obs")

    if time_order is None:
        raise ValueError("time_order parameter is required. Please provide a list of timepoints in the desired order.")

    # Check if all timepoints exist in the data
    available_timepoints = set(adata.obs[timepoint_column].unique())
    missing_timepoints = [tp for tp in time_order if tp not in available_timepoints]

    if missing_timepoints:
        raise ValueError(
            f"The following timepoints specified in time_order do not exist in the data: {missing_timepoints}. "
            f"Available timepoints are: {sorted(available_timepoints)}"
        )

    # Check interaction database exists
    if not in_uns(adata, ["receptor-ligand", "database"]):
        raise ValueError("No receptor-ligand database found in adata")

    # Filter interactions that exist in the database
    interaction_df = get_uns(adata, ["receptor-ligand", "database"])
    receptor_col = get_uns(adata, ["receptor-ligand", "receptor_column"])
    ligand_col = get_uns(adata, ["receptor-ligand", "ligand_column"])

    valid_interactions = []

    # check if the r-l combination exists in the database
    for r_gene, r_cluster, l_gene, l_cluster in interactions:
        mask = (
            (interaction_df[receptor_col] == r_gene)
            & (interaction_df[ligand_col] == l_gene)
        )
        if mask.any():
            valid_interactions.append((r_gene, r_cluster, l_gene, l_cluster))
        else:
            logger.warning(f"Warning: Interaction {l_gene}->{r_gene} not found")

    if not valid_interactions:
        raise ValueError("None of the specified interactions were found in the data")

    # Use the provided timepoints order
    timepoints = time_order

    # Set up figure dimensions
    n_interactions = len(valid_interactions)
    n_rows = math.ceil(n_interactions / n_cols)

    # Get colors from seaborn
    sns_colors = sns.color_palette()
    if receptor_color is None:
        receptor_color = sns_colors[0]
    if ligand_color is None:
        ligand_color = sns_colors[1]

    # Set figsize with more space to prevent overlapping
    if n_interactions == 1:
        if figsize is None:
            figsize = (10, 7)
        n_cols, n_rows = 1, 1
    elif figsize is None:
        figsize = (8 * n_cols, 6 * n_rows)

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi if dpi else settings.dpi, squeeze=False)

    # Calculate all expression values first
    expr_data = []
    all_expr_values = [] if use_global_ylim else None

    for idx, (r_gene, r_cluster, l_gene, l_cluster) in enumerate(valid_interactions):
        # Get expression data for each timepoint
        r_expr = [
            _get_gene_expression(adata, r_gene, r_cluster, tp, timepoint_column, cluster_column, layer=layer)
            for tp in timepoints
        ]
        l_expr = [
            _get_gene_expression(adata, l_gene, l_cluster, tp, timepoint_column, cluster_column, layer=layer)
            for tp in timepoints
        ]

        expr_data.append((r_expr, l_expr))

        # Only collect all values if using global ylim
        if use_global_ylim:
            all_expr_values.extend(r_expr + l_expr)

    # Calculate the global maximum for consistent y-axis scaling if requested
    global_max = max(all_expr_values) if use_global_ylim and all_expr_values else None

    # Plot each interaction with the global maximum y-limit
    for idx, ((r_gene, r_cluster, l_gene, l_cluster), ax) in enumerate(zip(valid_interactions, axes.flatten())):

        # Get the pre-calculated expression data
        r_expr, l_expr = expr_data[idx]

        # Plot bars with more spacing
        x = np.arange(len(timepoints))
        width = 0.3

        # Draw bars
        r_bars = ax.bar(
            # More space between bar groups
            x - width / 2 - 0.05,
            r_expr,
            width,
            facecolor=receptor_color,
            edgecolor='black',
            linewidth=0.5,
            label=f"Receptor {r_gene} in {r_cluster}"
        )

        l_bars = ax.bar(
            # More space between bar groups
            x + width / 2 + 0.05,
            l_expr,
            width,
            facecolor=ligand_color,
            edgecolor='black',
            linewidth=0.5,
            label=f"Ligand {l_gene} in {l_cluster}"
        )

        # Add value labels with offset to prevent overlap
        for bars in [r_bars, l_bars]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    # Determine reference value for scaling offset
                    reference_max = global_max if use_global_ylim else max(r_expr + l_expr)
                    # Scale offset based on appropriate max value
                    y_offset = 0.02 * reference_max
                    ax.annotate(
                        f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        # Add extra vertical offset
                        xytext=(0, 3 + y_offset),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
                    )

        # Format axes
        ax.set_xticks(x)
        ax.set_xticklabels(timepoints, rotation=45, ha="right", fontsize=12)
        ax.set_title(f"{r_gene} ({r_cluster}) - {l_gene} ({l_cluster})", fontsize=14)
        ax.set_xlabel(timepoint_column, fontsize=12)
        ax.set_ylabel("Expression level", fontsize=12)

        # Set tick font size
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Set y-limit based on configuration
        if use_global_ylim:
            # Use consistent y-limit based on global maximum
            # Add 25% extra space for labels
            y_max = global_max * 1.25
        else:
            # Use local maximum for this subplot
            local_max = max(r_expr + l_expr) if r_expr + l_expr else 1.0
            # Add 25% extra space for labels
            y_max = local_max * 1.25

        ax.set_ylim(0, y_max)

        # Add individual legend below plot with additional space
        ax.legend(fontsize=12, loc='upper right')

    # Hide empty subplots
    for empty_idx in range(n_interactions, n_rows * n_cols):
        empty_row, empty_col = empty_idx // n_cols, empty_idx % n_cols
        axes[empty_row, empty_col].set_visible(False)

    # Set title
    if title is None:
        title = "Receptor-Ligand Interaction" if n_interactions == 1 else "Receptor-Ligand Interactions"
    fig.suptitle(title, fontsize=18, y=0.98)

    # Adjust layout with more space
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.15,
        top=0.9,
        wspace=0.4,
        hspace=0.6
    )

    # Save figure
    _save_figure(save, dpi=dpi)

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True, dpi=dpi)

    return fig
