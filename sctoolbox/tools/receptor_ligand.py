"""Tools for a receptor-ligand analysis."""
import math
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations_with_replacement
import scipy
from sklearn.preprocessing import minmax_scale
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

import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings


# -------------------------------------------------- setup functions -------------------------------------------------- #


@deco.log_anndata
@beartype
def download_db(adata: sc.AnnData,
                db_path: str,
                ligand_column: str,
                receptor_column: str,
                sep: str = "\t",
                inplace: bool = False,
                overwrite: bool = False) -> Optional[sc.AnnData]:
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
    except FileNotFoundError:
        # Check if a LIANA resource
        if db_path in liana_res.show_resources():
            # get LIANA db
            database = liana_res.select_resource(db_path)
            # explode protein complexes interactions into single protein interactions
            database = liana_res.explode_complexes(database)
        else:
            raise ValueError(f"{db_path} is neither a valid file nor on of the available LIANA resources ({liana_res.show_resources()}).")

    # check column names in table
    if ligand_column not in database.columns:
        raise ValueError(f"Ligand column '{ligand_column}' not found in database! Available columns: {database.columns}")
    if receptor_column not in database.columns:
        raise ValueError(f"Receptor column '{receptor_column}' not found in database! Available columns: {database.columns}")

    modified_adata = adata if inplace else adata.copy()

    # setup dict to store information old data will be overwriten!
    modified_adata.uns['receptor-ligand'] = dict()

    modified_adata.uns['receptor-ligand']['database_path'] = db_path
    modified_adata.uns['receptor-ligand']['database'] = database
    modified_adata.uns['receptor-ligand']['ligand_column'] = ligand_column
    modified_adata.uns['receptor-ligand']['receptor_column'] = receptor_column

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
                                overwrite: bool = False) -> Optional[sc.AnnData]:
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
        Whether to weight the expression Z-Score by the expression proprotion.
    inplace : bool, default False
        Whether to copy `adata` or modify it inplace.
    overwrite : bool, default False
        If True will overwrite existing interaction table.

    Returns
    -------
    Optional[sc.AnnData]
        If not inpalce, return copy of adata with added interactions table to adata.uns['receptor-ligand']['interactions']

    Raises
    ------
    ValueError
        1: If receptor-ligand database cannot be found.
        2: Id database genes do not match adata genes.
    Exception
        If not interactions were found.
    """

    if "receptor-ligand" not in adata.uns.keys():
        raise ValueError("Could not find receptor-ligand database. Please setup database with `download_db(...)` before running this function.")

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
    cl_mean_expression = pd.DataFrame(index=index)
    # percent cells in cluster expressing gene
    cl_percent_expression = pd.DataFrame(index=index)
    # number of cells for each cluster
    clust_sizes = {}

    # fill above tables
    for cluster in tqdm(set(adata.obs[cluster_column]), desc="computing cluster gene scores"):
        # filter adata to a specific cluster
        cluster_adata = adata[adata.obs[cluster_column] == cluster]
        clust_sizes[cluster] = len(cluster_adata)

        # -- compute cluster means --
        if gene_index is None:
            cl_mean_expression.loc[cl_mean_expression.index.isin(cluster_adata.var.index), cluster] = cluster_adata.X.mean(axis=0).reshape(-1, 1)
        else:
            cl_mean_expression.loc[cl_mean_expression.index.isin(cluster_adata.var[gene_index]), cluster] = cluster_adata.X.mean(axis=0).reshape(-1, 1)

        # -- compute expression percentage --
        # get nonzero expression count for all genes
        _, cols = cluster_adata.X.nonzero()
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
    processed_pairs = set()
    for _, (receptor, ligand) in tqdm(adata.uns["receptor-ligand"]["database"][[r_col, l_col]].iterrows(),
                                      total=len(adata.uns["receptor-ligand"]["database"]),
                                      desc="finding receptor-ligand interactions"):
        # Skip if pair has already been processed or is empty
        pair_key = f"{receptor}|{ligand}"
        if pair_key in processed_pairs or receptor is np.nan or ligand is np.nan:
            continue

        processed_pairs.add(pair_key)

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
    else:
        interactions["receptor_score"] = interactions["receptor_score"]
        interactions["ligand_score"] = interactions["ligand_score"]

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
                            figsize: Tuple[int, int] = (5, 20),
                            dpi: int = 100) -> npt.ArrayLike:
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
    figsize : int tuple, default (5, 20)
        Figure size
    dpi : float, default 100
        The resolution of the figure in dots-per-inch.

    Returns
    -------
    npt.ArrayLike
        Object containing all plots. As returned by matplotlib.pyplot.subplots
    """

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
    if save:
        fig.savefig(f"{settings.figure_dir}/{save}")

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
    print(f"Max weight {np.max(np.array(graph.es['weight']))}")
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
    dpi: int | float = 100,
    save: Optional[str] = None
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
    dpi : int | float, default 100
        The resolution of the figure in dots-per-inch.
    save : str, default None
        Output filename. Uses the internal 'sctoolbox.settings.figure_dir'.

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

        # OpenAI GPT-4 supported >>>
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
    avail_clusters = set(filtered["receptor_cluster"].unique()).union(set(filtered["ligand_cluster"].unique()))

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
    fig = circos.plotfig(dpi=dpi, figsize=figsize)

    # add custom legend to plot
    patch_legend = circos.ax.legend(
        handles=patch_handles,
        bbox_to_anchor=(1, 1),
        fontsize=10,
        handlelength=1
    )
    circos.ax.add_artist(patch_legend)

    if save:
        fig.savefig(f"{settings.figure_dir}/{save}", dpi=dpi)

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
                         dpi: int = 100,
                         save: Optional[str] = None) -> matplotlib.axes.Axes:
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
    dpi : int, default 100
        Dots per inch.
    save : Optional[str], default None
        Output filename. Uses the internal 'sctoolbox.settings.figure_dir'.

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
    with plt.rc_context({"figure.figsize": figsize, "figure.dpi": dpi}):
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
        plt.savefig(f"{settings.figure_dir}/{save}")

    return plot


@deco.log_anndata
@beartype
def connectionPlot(adata: sc.AnnData,
                   restrict_to: Optional[list[str]] = None,
                   figsize: Tuple[int | float, int | float] = (10, 15),
                   dpi: int = 100,
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
                   alpha_range: Optional[Tuple[int | float, int | float]] = None) -> npt.ArrayLike:
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
    dpi : float, default 100
        The resolution of the figure in dots-per-inch.
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
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi, gridspec_kw={'wspace': wspace})
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

    if save:
        plt.savefig(f"{settings.figure_dir}/{save}", bbox_inches='tight')

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
        interaction_score = np.percentile(table["interaction_score"], interaction_perc)
    elif interaction_score is None:
        interaction_score = min(table["interaction_score"]) - 1

    subset = table[
        (table["receptor_percent"] >= min_perc)
        & (table["ligand_percent"] >= min_perc)
        & (table["interaction_score"] > interaction_score)
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
        raise ValueError("Could not find interaction data! Please setup with `calculate_interaction_table(...)` before running this function.")

# ----------------------------------------- differences calculation ---------------------------------------------------------------------#


@deco.log_anndata
@beartype
def calculate_condition_differences(adata: sc.AnnData,
                                    condition_columns: List[str],
                                    cluster_column: str,
                                    min_perc: Optional[int | float] = None,
                                    interaction_score: Optional[float | int] = None,
                                    interaction_perc: Optional[int | float] = None,
                                    condition_filters: Optional[Dict[str, List[str]]] = None,
                                    gene_column: Optional[str] = None,
                                    cluster_filter: Optional[List[str]] = None,
                                    gene_filter: Optional[List[str]] = None,
                                    normalize: Optional[int] = None,
                                    weight_by_ep: Optional[bool] = True,
                                    inplace: bool = False) -> Optional[Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
    """
    Calculate interaction quantile rank differences between conditions.

    This function compares values of the first condition within each
    combination of the subsequent conditions.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object that holds the expression values and metadata
    condition_columns : List[str]
        Names of columns in adata.obs for hierarchical filtering, in order.
        The first column contains values to be compared within each combination of the other columns.
    cluster_column : str
        Name of the cluster column in adata.obs.
    min_perc : Optional[int | float], default None
        Minimum percentage of cells in a cluster that express the respective gene. A value from 0-100.
    interaction_score : Optional[float | int], default None
        Filter receptor-ligand interactions below given score. Ignored if `interaction_perc` is set.
    interaction_perc : Optional[int | float], default None
        Filter receptor-ligand interactions below the given percentile. Overwrite `interaction_score`. Value from 0-100.
    condition_filters : Optional[Dict[str, List[str]]], default None
        Dictionary mapping condition column names to lists of values to include.
        If a column is not in this dictionary or the dictionary is None, all unique values for that column will be used.
    gene_column : Optional[str], default None
        Column in adata.var that holds gene symbols/ids. Uses index when None.
    cluster_filter : Optional[List[str]], default None
        List of cluster names to include in the analysis. If None, all clusters will be included.
    gene_filter : Optional[List[str]], default None
        List of genes to include in the analysis. If None, all genes will be included.
    normalize : Optional[int], default None
        Correct clusters to given size. If None, max clustersize is used.
    weight_by_ep : Optional[bool], default True
        Whether to weight the expression Z-Score by the expression proprotion.
    inplace : bool, default False
        Whether to copy `adata` or modify it inplace.

    Returns
    -------
    Optional[Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]
        If not inplace, return nested dictionary with results organized by condition combinations:
        - First level keys are condition dimensions (e.g., 'group_timepoint')
        - Second level keys are specific comparisons (e.g., 'timepoint=4d_Infected_vs_Control')
        - Third level contains 'differences': DataFrame of interaction differences

    Raises
    ------
    ValueError
        If invalid keys are provided in condition_filters
        If no valid values exist for a condition after filtering
        If fewer than one condition column is provided
        If no valid values match any provided filters
    TypeError
        If cluster_filter is not a list
        If gene_filter is not a list
    """

    if condition_filters is not None:
        invalid_keys = set(condition_filters.keys()) - set(condition_columns)
        if invalid_keys:
            raise ValueError(f"Invalid keys in condition_filters: {invalid_keys}. Valid keys are: {condition_columns}")


    if len(condition_columns) < 1:
        raise ValueError(f"Need at least one condition column, got {len(condition_columns)}")

    # Function to create a filtered AnnData object
    def create_filtered_adata(condition_values):
        if len(condition_values) != len(condition_columns):
            raise ValueError(f"Expected {len(condition_columns)} condition values, got {len(condition_values)}")

        # Create filter query
        condition_query = " & ".join([f"{col} == '{val}'" for col, val in zip(condition_columns, condition_values)])

        # Check if any cells match the condition
        matching_cells = adata.obs.eval(condition_query).sum()
        if matching_cells == 0:
            warnings.warn(f"No cells found for condition {condition_values}. Skipping.")
            return None

        # Create filtered AnnData
        filtered_adata = adata[adata.obs.eval(condition_query)].copy()

        # Apply cluster filter if provided
        if cluster_filter is not None:
            available_clusters = set(filtered_adata.obs[cluster_column].unique())
            requested_clusters = set(cluster_filter)
            valid_clusters = requested_clusters.intersection(available_clusters)

            if not valid_clusters:
                warnings.warn(f"No valid clusters for condition {condition_values}. Skipping.")
                return None

            cluster_mask = filtered_adata.obs[cluster_column].isin(valid_clusters)
            if cluster_mask.sum() == 0:
                warnings.warn(f"No cells match cluster filter for {condition_values}. Skipping.")
                return None

            filtered_adata = filtered_adata[cluster_mask].copy()

            del cluster_mask

        if gene_filter is not None and gene_column in filtered_adata.var.columns:
            available_genes = set(filtered_adata.var[gene_column])
            requested_genes = set(gene_filter)
            valid_genes = requested_genes.intersection(available_genes)

            if not valid_genes:
                warnings.warn(f"No valid genes for condition {condition_values}. Skipping.")
                return None

            gene_mask = filtered_adata.var[gene_column].isin(valid_genes)
            if gene_mask.sum() == 0:
                warnings.warn(f"No genes match gene filter for {condition_values}. Skipping.")
                return None

            filtered_adata = filtered_adata[:, gene_mask].copy()

            del gene_mask

        # Calculate interaction table
        try:
            calculate_interaction_table(
                adata=filtered_adata,
                cluster_column=cluster_column,
                gene_index=gene_column,
                normalize=normalize,
                weight_by_ep=weight_by_ep,
                inplace=True,
                overwrite=True
            )
            return filtered_adata
        except Exception as e:
            warnings.warn(f"Error calculating interactions for {condition_values}: {str(e)}")
            return None

    # Get all possible values for each condition
    condition_values_dict = {}
    for col in condition_columns:
        if col in condition_filters and condition_filters[col]:
            # Use filtered values
            available_values = set(adata.obs[col].unique())
            requested_values = set(condition_filters[col])
            valid_values = list(requested_values.intersection(available_values))

            if not valid_values:
                raise ValueError(f"No valid values for condition '{col}'. Available: {available_values}")

            condition_values_dict[col] = valid_values
        else:
            # Use all values
            condition_values_dict[col] = sorted(list(adata.obs[col].unique()))

    # Function to recursively process condition combinations
    def process_combinations(current_level=1, fixed_conditions=None):
        if fixed_conditions is None:
            fixed_conditions = {}

        # If all secondary conditions are processed, perform comparison
        if current_level >= len(condition_columns):

            # Get primary condition values for comparison
            primary_column = condition_columns[0]
            primary_values = condition_values_dict[primary_column]

            if len(primary_values) < 2:
                warnings.warn(f"Need at least 2 values for {primary_column} to compare, got {primary_values}")
                return {}

            # Create a description of the fixed conditions
            fixed_desc = "_".join([f"{col}={val}" for col, val in fixed_conditions.items()])

            print(f"Comparing {primary_column} values within {fixed_desc}")

            # Dictionary to store AnnData objects for each primary value
            primary_adatas = {}

            # Process each primary value
            for primary_value in primary_values:
                # Combine with fixed conditions
                all_values = [primary_value] + [fixed_conditions[col] for col in condition_columns[1:]]

                # Create filtered AnnData
                print(f"Processing {list(zip(condition_columns, all_values))}")
                filtered_adata = create_filtered_adata(all_values)

                if filtered_adata is not None:
                    primary_adatas[primary_value] = filtered_adata
                    print(f"Processed {primary_column}={primary_value} for {fixed_desc} ({filtered_adata.n_obs} cells)")

            # Compare all pairs of primary values
            results = {}

            # Compare all pairs of primary values
            for i in range(len(primary_values)):
                for j in range(i + 1, len(primary_values)):
                    value_a = primary_values[i]
                    value_b = primary_values[j]

                    if value_a not in primary_adatas or value_b not in primary_adatas:
                        warnings.warn(f"Missing data for comparison between {value_a} and {value_b}")
                        continue

                    print(f"Comparing {value_b} vs {value_a} within {fixed_desc}")

                    # Calculate differences
                    differences = calculate_differences(
                        primary_adatas[value_a],
                        primary_adatas[value_b],
                        value_a,
                        value_b,
                        min_perc=min_perc,
                        interaction_score=interaction_score,
                        interaction_perc=interaction_perc
                    )

                    if differences is not None:
                        # Create comparison key
                        if fixed_desc:
                            comparison_key = f"{fixed_desc}_{value_b}_vs_{value_a}"
                        else:
                            comparison_key = f"{value_b}_vs_{value_a}"

                        # Store results
                        results[comparison_key] = {
                            'differences': differences
                        }

                        # Save to file if needed
                        differences.to_csv(f"{settings.table_dir}/{comparison_key}_differences.csv", sep='\t', index=False)
            primary_adatas.clear()
            del primary_adatas

            return results

        # Process the next level of conditions
        current_column = condition_columns[current_level]
        possible_values = condition_values_dict[current_column]

        # Dictionary to store results for this level
        level_results = {}

        # Process each value of the current condition
        for value in possible_values:
            # Update fixed conditions
            new_fixed = fixed_conditions.copy()
            new_fixed[current_column] = value

            # Recursively process the next level
            next_results = process_combinations(current_level + 1, new_fixed)

            # Merge with level results
            level_results.update(next_results)

            del new_fixed

        return level_results

    # Function to calculate differences between two AnnData objects
    def calculate_differences(adata_a, adata_b, value_a, value_b, **kwargs):
        try:
            # Get interaction tables
            interactions_a = get_interactions(
                anndata=adata_a,
                min_perc=kwargs.get('min_perc'),
                interaction_score=kwargs.get('interaction_score'),
                interaction_perc=kwargs.get('interaction_perc')
            )

            interactions_b = get_interactions(
                anndata=adata_b,
                min_perc=kwargs.get('min_perc'),
                interaction_score=kwargs.get('interaction_score'),
                interaction_perc=kwargs.get('interaction_perc')
            )

            print(f"Found {len(interactions_a)} interactions for {value_a}")
            print(f"Found {len(interactions_b)} interactions for {value_b}")

            # Add condition labels
            interactions_a = interactions_a.copy()
            interactions_b = interactions_b.copy()

            interactions_a['condition'] = 'a'
            interactions_b['condition'] = 'b'

            # Combine for quantile ranking
            merged_long = pd.concat([interactions_a, interactions_b], ignore_index=True)

            # Calculate quantile ranking
            merged_long['quantile_rank'] = merged_long['interaction_score'].rank(
                method='average',
                pct=True
            )

            # Extract ranks
            ranks_a = merged_long[merged_long['condition'] == 'a'][['receptor_gene', 'ligand_gene', 'receptor_cluster', 'ligand_cluster', 'quantile_rank']]
            ranks_b = merged_long[merged_long['condition'] == 'b'][['receptor_gene', 'ligand_gene', 'receptor_cluster', 'ligand_cluster', 'quantile_rank']]

            del merged_long

            # Merge ranks back
            interactions_a = pd.merge(
                interactions_a,
                ranks_a,
                on=['receptor_gene', 'ligand_gene', 'receptor_cluster', 'ligand_cluster']
            )

            del ranks_a

            interactions_b = pd.merge(
                interactions_b,
                ranks_b,
                on=['receptor_gene', 'ligand_gene', 'receptor_cluster', 'ligand_cluster']
            )

            del ranks_b

            # Merge interaction tables
            merged = pd.merge(
                interactions_a, interactions_b,
                on=['receptor_gene', 'ligand_gene', 'receptor_cluster', 'ligand_cluster'],
                suffixes=('_a', '_b')
            )

            del interactions_a
            del interactions_b

            # Calculate differences
            merged['rank_diff'] = merged['quantile_rank_b'] - merged['quantile_rank_a']
            merged['abs_diff'] = merged['rank_diff'].abs()

            # Sort by absolute difference
            merged = merged.sort_values('abs_diff', ascending=False)

            # Store metadata
            merged.attrs['condition_a'] = value_a
            merged.attrs['condition_b'] = value_b

            return merged

        except Exception as e:
            warnings.warn(f"Error calculating differences: {str(e)}")
            return None

    # Start processing
    print("Starting comparison processing...")
    all_results = {condition_columns[0]: process_combinations()}

    if len(all_results[condition_columns[0]]) == 0:
        warnings.warn("No comparisons found. Check your condition values and filters.")
    else:
        print(f"Completed processing with {len(all_results[condition_columns[0]])} comparisons")

    if inplace:
        if "condition-differences" not in adata.uns:
            adata.uns["condition-differences"] = {}
        adata.uns["condition-differences"] = all_results
    else:
        return all_results


@deco.log_anndata
@beartype
def calculate_condition_differences_over_time(
        adata: sc.AnnData,
        timepoint_column: str,
        condition_column: str,
        condition_value: str,
        cluster_column: str,
        min_perc: Optional[int | float] = None,
        interaction_score: Optional[float | int] = None,
        interaction_perc: Optional[int | float] = None,
        reference_timepoint: Optional[str] = None,
        timepoint_order: List[str] = None,
        gene_column: Optional[str] = None,
        cluster_filter: Optional[List[str]] = None,
        gene_filter: Optional[List[str]] = None,
        normalize: Optional[int] = None,
        weight_by_ep: Optional[bool] = True,
        inplace: bool = False,
        overwrite: bool = False,
        save: Optional[str] = None) -> Optional[Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
    """
    Analyze cell-cell interactions across multiple timepoints for a specific condition.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object that contains the full dataset.
    timepoint_column : str
        The column in adata.obs containing timepoint information.
    condition_column : str
        The column in adata.obs containing condition information.
    condition_value : str
        The specific condition value to analyze (e.g., "treatment" or "control").
    cluster_column : str
        The column in adata.obs containing cluster information.
    min_perc : Optional[int | float], default None
        Minimum percentage of cells in a cluster that express the respective gene. A value from 0-100.
    interaction_score : Optional[float | int], default None
        Filter receptor-ligand interactions below given score. Ignored if `interaction_perc` is set.
    interaction_perc : Optional[int | float], default None
        Filter receptor-ligand interactions below the given percentile. Overwrite `interaction_score`. Value from 0-100.
    reference_timepoint : Optional[str], default None
        If specified, all timepoints will be compared to this reference timepoint.
        If None, consecutive timepoints will be compared.
    timepoint_order : List[str], default None
        Custom ordering of timepoints.
    gene_column : Optional[str], default None
        Column in adata.var that holds gene symbols/ids.
    cluster_filter : Optional[List[str]], default None
        List of cluster names to include in the analysis. If None, all clusters will be included.
    gene_filter : Optional[List[str]], default None
        List of genes to include in the analysis. If None, all genes will be included.
    normalize : Optional[int], default None
        Correct clusters to given size. If None, max clustersize is used.
    weight_by_ep : Optional[bool], default True
        Whether to weight the expression Z-Score by the expression proprotion.
    inplace : bool, default False
        Whether to copy `adata` or modify it inplace.
    overwrite : bool, default False
        Whether to overwrite existing temporal difference analysis.
    save : Optional[str], default None
        Output filename base for saving results.

    Returns
    -------
    Optional[Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]
        If not inplace, return hierarchical dictionary with the structure:
        {
            "time_series": {
                "comparison_key": {
                    "differences": pd.DataFrame
                }
            }
        }
        Compatible with condition_differences_network function.

    Raises
    ------
    ValueError
        1: If fewer than 2 timepoints found for the specified condition.
        2: If the reference timepoint is not found.
        3: If cluster_filter or gene_filter contain invalid entries.
    TypeError
        If cluster_filter or gene_filter are not lists.
    """

    # Validate input parameters
    if cluster_filter is not None and not isinstance(cluster_filter, list):
        raise TypeError(f"cluster_filter must be a list, got {type(cluster_filter)}")

    if gene_filter is not None:
        if not isinstance(gene_filter, list):
            raise TypeError(f"gene_filter must be a list, got {type(gene_filter)}")
        if gene_column is None:
            warnings.warn("gene_filter provided but gene_column is None. Gene filtering will be skipped.")

    # Get all timepoints for the specified condition
    condition_mask = adata.obs[condition_column] == condition_value
    available_timepoints = set(adata.obs.loc[condition_mask, timepoint_column].unique())

    if len(available_timepoints) < 2:
        raise ValueError(f"Found fewer than 2 timepoints for condition '{condition_value}'.")

    # Handle custom timepoint ordering
    if timepoint_order is not None:
        # Validate that all specified timepoints exist in the data
        missing_timepoints = [tp for tp in timepoint_order if tp not in available_timepoints]
        if missing_timepoints:
            raise ValueError(f"The following timepoints specified in timepoint_order were not found in the data: {missing_timepoints}")
        timepoints = [tp for tp in timepoint_order if tp in available_timepoints]
    else:
        raise ValueError("timepoint_order must be explicitly provided")

    print(f"Using {len(timepoints)} timepoints for condition '{condition_value}': {timepoints}")

    # Determine which comparisons to make
    comparisons = []
    if reference_timepoint is not None:
        if reference_timepoint not in timepoints:
            raise ValueError(f"Reference timepoint '{reference_timepoint}' not found in data or in the specified timepoint order.")
        # Compare all other timepoints to the reference
        comparisons = [(reference_timepoint, tp) for tp in timepoints if tp != reference_timepoint]
    else:
        # Default to sequential comparison
        comparisons = [(timepoints[i], timepoints[i + 1]) for i in range(len(timepoints) - 1)]

    # Create formatted condition strings for each timepoint
    timepoint_conditions = {tp: f"{condition_value}_{tp}" for tp in timepoints}

    # Function to create a filtered AnnData for a specific timepoint
    def create_filtered_adata(timepoint):
        print(f"Processing timepoint: {timepoint}")

        # Create the filter query
        condition_query = f"{condition_column} == '{condition_value}' & {timepoint_column} == '{timepoint}'"

        # Check if any cells match before creating subset
        matching_cells = adata.obs.eval(condition_query).sum()
        if matching_cells == 0:
            print(f"No cells found for timepoint {timepoint}. Skipping.")
            return None

        # Create filtered AnnData
        filtered_adata = adata[adata.obs.eval(condition_query)].copy()

        # Apply cluster filter if provided
        if cluster_filter is not None:
            available_clusters = set(filtered_adata.obs[cluster_column].unique())
            requested_clusters = set(cluster_filter)
            valid_clusters = requested_clusters.intersection(available_clusters)

            if not valid_clusters:
                print(f"No valid clusters for timepoint {timepoint}. Skipping.")
                return None

            cluster_mask = filtered_adata.obs[cluster_column].isin(valid_clusters)
            filtered_adata = filtered_adata[cluster_mask].copy()

        # Apply gene filter if provided
        if gene_filter is not None and gene_column in filtered_adata.var.columns:
            available_genes = set(filtered_adata.var[gene_column])
            requested_genes = set(gene_filter)
            valid_genes = requested_genes.intersection(available_genes)

            if not valid_genes:
                print(f"No valid genes for timepoint {timepoint}. Skipping.")
                return None

            gene_mask = filtered_adata.var[gene_column].isin(valid_genes)
            filtered_adata = filtered_adata[:, gene_mask].copy()

        # Calculate interaction table
        try:
            calculate_interaction_table(
                adata=filtered_adata,
                cluster_column=cluster_column,
                gene_index=gene_column,
                normalize=normalize,
                weight_by_ep=weight_by_ep,
                inplace=True,
                overwrite=True
            )
            return filtered_adata
        except Exception as e:
            print(f"Error calculating interactions for timepoint {timepoint}: {str(e)}")
            return None

    # Function to get interaction data from AnnData
    def get_filtered_interactions(adata_obj):
        if adata_obj is None or "receptor-ligand" not in adata_obj.uns:
            return None

        try:
            interactions = get_interactions(
                anndata=adata_obj,
                min_perc=min_perc,
                interaction_score=interaction_score,
                interaction_perc=interaction_perc
            )
            return interactions
        except Exception as e:
            print(f"Error getting interactions: {str(e)}")
            return None

    # Function to calculate differences between two timepoints
    def calculate_timepoint_diff(tp1, tp2):
        print(f"Comparing timepoints: {tp1} vs {tp2}")

        # Get condition labels
        tp1_condition = timepoint_conditions[tp1]
        tp2_condition = timepoint_conditions[tp2]

        # Process first timepoint
        adata_tp1 = create_filtered_adata(tp1)
        if adata_tp1 is None:
            print(f"Failed to process timepoint {tp1}. Skipping comparison.")
            return None

        interactions_tp1 = get_filtered_interactions(adata_tp1)
        if interactions_tp1 is None or len(interactions_tp1) == 0:
            print(f"No interactions found for timepoint {tp1}. Skipping comparison.")
            del adata_tp1
            return None

        # Process second timepoint
        adata_tp2 = create_filtered_adata(tp2)
        if adata_tp2 is None:
            print(f"Failed to process timepoint {tp2}. Skipping comparison.")
            del adata_tp1
            del interactions_tp1
            return None

        interactions_tp2 = get_filtered_interactions(adata_tp2)
        if interactions_tp2 is None or len(interactions_tp2) == 0:
            print(f"No interactions found for timepoint {tp2}. Skipping comparison.")
            del adata_tp1
            del adata_tp2
            del interactions_tp1
            return None

        print(f"Found {len(interactions_tp1)} interactions for {tp1}")
        print(f"Found {len(interactions_tp2)} interactions for {tp2}")

        # Prepare for comparison
        interactions_tp1 = interactions_tp1.copy()
        interactions_tp2 = interactions_tp2.copy()

        interactions_tp1['timepoint'] = 'tp1'
        interactions_tp2['timepoint'] = 'tp2'

        # Combine for quantile ranking
        tp1_for_ranking = interactions_tp1[['receptor_gene', 'ligand_gene', 'receptor_cluster', 'ligand_cluster', 'interaction_score']]
        tp2_for_ranking = interactions_tp2[['receptor_gene', 'ligand_gene', 'receptor_cluster', 'ligand_cluster', 'interaction_score']]

        combined_scores = pd.concat([
            tp1_for_ranking['interaction_score'],
            tp2_for_ranking['interaction_score']
        ])

        # Calculate ranks
        ranks = combined_scores.rank(method='average', pct=True)

        # Split ranks back to respective timepoints
        tp1_ranks = ranks[:len(tp1_for_ranking)]
        tp2_ranks = ranks[len(tp1_for_ranking):]

        # Add ranks back to original dataframes
        interactions_tp1['quantile_rank'] = tp1_ranks.values
        interactions_tp2['quantile_rank'] = tp2_ranks.values

        del tp1_for_ranking
        del tp2_for_ranking
        del combined_scores
        del ranks

        # Merge interaction tables
        interactions_tp1['join_key'] = interactions_tp1.apply(
            lambda x: f"{x['receptor_gene']}|{x['ligand_gene']}|{x['receptor_cluster']}|{x['ligand_cluster']}",
            axis=1
        )

        interactions_tp2['join_key'] = interactions_tp2.apply(
            lambda x: f"{x['receptor_gene']}|{x['ligand_gene']}|{x['receptor_cluster']}|{x['ligand_cluster']}",
            axis=1
        )

        # Select only necessary columns for merge
        tp1_for_merge = interactions_tp1[['join_key', 'interaction_score', 'quantile_rank']]
        tp2_for_merge = interactions_tp2[['join_key', 'interaction_score', 'quantile_rank']]

        # Rename columns
        tp1_for_merge.columns = ['join_key', 'interaction_score_a', 'quantile_rank_a']
        tp2_for_merge.columns = ['join_key', 'interaction_score_b', 'quantile_rank_b']

        # Merge on join key
        merged = pd.merge(tp1_for_merge, tp2_for_merge, on='join_key')

        # Split join key back to original components
        key_components = merged['join_key'].str.split('|', expand=True)
        merged['receptor_gene'] = key_components[0]
        merged['ligand_gene'] = key_components[1]
        merged['receptor_cluster'] = key_components[2]
        merged['ligand_cluster'] = key_components[3]

        # Drop the join key
        merged.drop('join_key', axis=1, inplace=True)

        # Calculate differences
        merged['rank_diff'] = merged['quantile_rank_b'] - merged['quantile_rank_a']
        merged['abs_diff'] = merged['rank_diff'].abs()

        # Sort by absolute difference
        merged = merged.sort_values('abs_diff', ascending=False)

        # Add metadata using attrs
        merged.attrs['timepoint_1'] = tp1
        merged.attrs['timepoint_2'] = tp2
        merged.attrs['condition'] = condition_value
        merged.attrs['condition_a'] = tp1_condition
        merged.attrs['condition_b'] = tp2_condition
        merged.attrs['group_name'] = f"Timepoint Comparison: {tp2} vs {tp1}"
        merged.attrs['condition_name'] = "time_series"

        del adata_tp1
        del adata_tp2
        del interactions_tp1
        del interactions_tp2
        del tp1_for_merge
        del tp2_for_merge

        # Save results if requested
        if save:
            comparison_key = f"{tp2_condition}_vs_{tp1_condition}"
            csv_path = f"{save}_{comparison_key}_differences.csv"
            merged.to_csv(f"{settings.table_dir}/{csv_path}", sep='\t', index=False)

        return merged

    # Initialize the hierarchical results dictionary
    hierarchical_results = {"time_series": {}}

    # Process each comparison individually to minimize memory usage
    for tp1, tp2 in comparisons:
        if tp1 == tp2:
            continue

        # Calculate differences
        diff_result = calculate_timepoint_diff(tp1, tp2)

        if diff_result is not None:
            # Create comparison key
            tp1_condition = timepoint_conditions[tp1]
            tp2_condition = timepoint_conditions[tp2]
            comparison_key = f"{tp2_condition}_vs_{tp1_condition}"

            # Store in results dictionary
            hierarchical_results["time_series"][comparison_key] = {"differences": diff_result}

    # Store results in the AnnData object if inplace
    if inplace:
        if "temporal-differences" not in adata.uns or overwrite:
            adata.uns["temporal-differences"] = hierarchical_results
        else:
            warnings.warn("Temporal differences already exist in adata.uns['temporal-differences']. Set overwrite=True to replace.")
        return None
    else:
        return hierarchical_results


@beartype
def condition_differences_network(diff_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
                                  n_top: int = 100,
                                  figsize: Tuple[int, int] = (22, 16),
                                  dpi: int = 300,
                                  save: Optional[str] = None,
                                  split_by_direction: bool = True,
                                  hub_threshold: int = 4) -> List[matplotlib.figure.Figure]:
    """
    Visualize differences between conditions as a receptor-ligand network with hubs separated.

    Creates a multi-grid visualization where network hubs are displayed separately.

    Parameters
    ----------
    diff_results : Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
        Results from calculate_condition_differences or analyze_condition_over_time function
    n_top : int, default 20
        Number of top differential interactions to display
    figsize : Tuple[int, int], default (22, 16)
        Size of the figure
    dpi : int, default 300
        The resolution of the figure.
    save : Optional[str], default None
        Output filename. Uses the internal 'sctoolbox.settings.figure_dir'.
    split_by_direction : bool, default True
        Whether to create separate networks for positive and negative differences
    hub_threshold : int, default 4
        Minimum number of connections for a node to be considered a hub

    Returns
    -------
    List[matplotlib.figure.Figure]
        List of generated figures

    Raises
    ------
    ValueError
        If no valid differences are found in the provided diff_results.
    """
    # Check if data is available
    if not diff_results or all(len(dimension) == 0 for dimension in diff_results.values()):
        raise ValueError("No valid condition differences found in the provided results.")

    # List to store all generated figures
    figures = []

    # Dictionary to store cluster column colors
    all_cell_types = set()

    # Collect all cell types across all comparisons
    for _, con_results in diff_results.items():
        for comparison_key, comparison_data in con_results.items():
            if 'differences' not in comparison_data:
                continue

            diff_df = comparison_data['differences']

            # Add cell types to the set
            all_cell_types.update(diff_df['receptor_cluster'].unique())
            all_cell_types.update(diff_df['ligand_cluster'].unique())

    # Get colormap
    color_palette = plt.colormaps.get_cmap('tab20')

    # Define node shapes
    cluster_shapes = ['o', 's', 'p', '^', 'D', 'v', '<', '>', '8', 'h', 'H', 'd', 'P', 'X']

    # Create a mapping of cell type to (color, shape) pairs
    cell_colors = {}
    cell_shapes = {}

    # Create the 280 pairs of (color, sape) combinations
    sorted_cell_types = sorted(all_cell_types)
    for i, ct in enumerate(sorted_cell_types):
        color_idx = i % 20
        shape_idx = i // 20

        cell_colors[ct] = color_palette(color_idx)

        if shape_idx < len(cluster_shapes):
            cell_shapes[ct] = cluster_shapes[shape_idx]
        else:
            # If more then 280 clusters, then clycle
            cell_shapes[ct] = cluster_shapes[shape_idx % len(cluster_shapes)]

    # Process each condition dimension
    for con_name, con_results in diff_results.items():

        # Process each comparison in this dimension
        for comparison_key, comparison_data in con_results.items():
            # Get the differences table
            if 'differences' not in comparison_data:
                warnings.warn(f"No differences table found for {comparison_key}, skipping visualization")
                continue

            diff_df = comparison_data['differences']

            # Skip if no differences
            if len(diff_df) == 0:
                warnings.warn(f"No differences found for {comparison_key}, skipping visualization")
                continue

            # Get condition metadata for title
            condition_a = diff_df.attrs.get('condition_a', 'Condition A')
            condition_b = diff_df.attrs.get('condition_b', 'Condition B')

            # Determine directions to plot
            directions = []
            if split_by_direction:
                # Get top positive differences
                pos_diff = diff_df[diff_df['rank_diff'] > 0].sort_values('rank_diff', ascending=False).head(n_top)
                if len(pos_diff) > 0:
                    directions.append(('positive', pos_diff, 'Higher in ' + condition_b))

                # Get top negative differences
                neg_diff = diff_df[diff_df['rank_diff'] < 0].sort_values('rank_diff', ascending=True).head(n_top)
                if len(neg_diff) > 0:
                    directions.append(('negative', neg_diff, 'Higher in ' + condition_a))
            else:
                # Get top absolute differences
                top_diff = diff_df.sort_values('abs_diff', ascending=False).head(n_top)
                if len(top_diff) > 0:
                    directions.append(('all', top_diff, 'All differences'))

            # Skip if no directions to plot
            if not directions:
                warnings.warn(f"No differences found for {comparison_key}, skipping visualization")
                continue

            # Create visualizations for each direction
            for direction_name, top_diff, direction_label in directions:
                # Create a graph
                G = nx.DiGraph()

                # Add nodes and edges for top interactions
                for _, row in top_diff.iterrows():
                    # Node identifiers
                    r_node = f"{row['receptor_gene']}_{row['receptor_cluster']}"
                    l_node = f"{row['ligand_gene']}_{row['ligand_cluster']}"

                    # Add nodes with metadata
                    G.add_node(r_node,
                               cell_type=row['receptor_cluster'],
                               gene=row['receptor_gene'],
                               is_receptor=True)

                    G.add_node(l_node,
                               cell_type=row['ligand_cluster'],
                               gene=row['ligand_gene'],
                               is_receptor=False)

                    # Add edge with metadata
                    G.add_edge(l_node, r_node,
                               weight=abs(row['rank_diff']),
                               diff=row['rank_diff'],
                               interaction_a=row['interaction_score_a'],
                               interaction_b=row['interaction_score_b'])

                # Skip if no edges in the graph
                if len(G.edges) == 0:
                    warnings.warn(f"No {direction_name} differences found for {comparison_key}, skipping visualization")
                    continue

                # Identify hub nodes (nodes with high connectivity)
                node_connections = Counter()
                for u, v in G.edges():
                    node_connections[u] += 1
                    node_connections[v] += 1

                # Nodes that have connections greater than or equal to the threshold are considered hubs
                hub_nodes = {node for node, count in node_connections.items() if count >= hub_threshold}

                # Create a dictionary to organize hubs and their direct connections
                hub_networks = {}
                non_hub_edges = []

                # If there are hub nodes, create a separate subgraph for each hub
                if hub_nodes:
                    # Create a subgraph for each hub node
                    for hub in hub_nodes:
                        # Get all neighbors of the hub
                        neighbors = set(G.successors(hub)).union(set(G.predecessors(hub)))
                        # Create a subgraph with the hub and its neighbors
                        nodes_in_subgraph = {hub}.union(neighbors)
                        subgraph = G.subgraph(nodes_in_subgraph).copy()
                        hub_networks[hub] = subgraph

                # Create a subgraph with remaining non-hub nodes and edges
                for u, v in G.edges():
                    if u not in hub_nodes and v not in hub_nodes:
                        non_hub_edges.append((u, v))

                # Create the non-hub subgraph
                if non_hub_edges:
                    non_hub_subgraph = G.edge_subgraph(non_hub_edges).copy()
                else:
                    non_hub_subgraph = nx.DiGraph()

                # Determine layout for the multi-grid visualization
                num_hubs = len(hub_networks)
                has_non_hub = len(non_hub_edges) > 0

                # Calculate grid layout parameters with last column for cluster legend
                if num_hubs == 0:
                    # Only non-hub network + legend
                    rows, cols = 1, 2
                elif num_hubs == 1:
                    rows, cols = 1, 3
                elif num_hubs == 2:
                    rows, cols = 1, 4
                elif num_hubs == 3:
                    rows, cols = 2, 3
                elif num_hubs <= 6:
                    rows, cols = 2, 4
                else:
                    cols = 4
                    rows = math.ceil((num_hubs + (1 if has_non_hub else 0)) / (cols - 1))  # +1 for legend

                # Create figure with GridSpec for multi-grid layout
                fig = plt.figure(figsize=figsize, dpi=dpi)

                # Create grid with the required number of rows and columns
                # while reserving the last column for the legend and make halve size
                width_ratios = [1] * (cols - 1) + [0.5]

                # Create the grid with appropriate spacing
                gs = gridspec.GridSpec(
                    rows, cols,
                    figure=fig,
                    width_ratios=width_ratios,
                    wspace=0.3, hspace=0.35
                )

                # Determine min and max values for the colormap
                if direction_name == 'positive':
                    vmin = 0
                    # vmin = min(row['rank_diff'] for _, row in top_diff.iterrows()) - 0.2
                    # vmax = max(row['rank_diff'] for _, row in top_diff.iterrows())
                    vmax = 1
                    cmap = 'Reds'
                elif direction_name == 'negative':
                    # vmin = min(row['rank_diff'] for _, row in top_diff.iterrows()) - 0.2
                    vmin = -1
                    vmax = 0
                    # vmax = max(row['rank_diff'] for _, row in top_diff.iterrows())
                    cmap = 'Blues_r'
                else:
                    vmin = -1
                    vmax = 1
                    cmap = 'RdBu_r'

                # Create a normalized colormap
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                colormap = plt.colormaps.get_cmap(cmap)

                # Function to draw a network on a given axis
                def draw_network(graph, ax, title=None):
                    # Layout for this network
                    pos = nx.spring_layout(graph, k=2.0, iterations=200, seed=42, scale=1.0)

                    # Draw edges
                    for u, v, data in graph.edges(data=True):
                        # Edge color based on difference
                        edge_color = colormap(norm(data['diff']))

                        width = 1.5

                        # Draw the edge
                        nx.draw_networkx_edges(
                            graph, pos,
                            edgelist=[(u, v)],
                            width=width,
                            edge_color=[edge_color],
                            arrows=True,
                            arrowsize=15,
                            arrowstyle='-|>',
                            connectionstyle='arc3,rad=0.1',
                            ax=ax
                        )

                        # Add edge label with difference score
                        x = (pos[u][0] + pos[v][0]) / 2
                        y = (pos[u][1] + pos[v][1]) / 2

                        edge_label = f"{data['diff']:.2f}"
                        ax.text(
                            x, y, edge_label, fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', boxstyle='round,pad=0.2'),
                            ha='center', va='center'
                        )

                    # Draw nodes with consistent shape and color based on cluster
                    for cell_type in sorted(all_cell_types):
                        # Get all nodes for this cluster
                        ct_nodes = [n for n in graph.nodes() if graph.nodes[n].get('cell_type') == cell_type]

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

                    # Add node labels with adjusted positions
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

                # Create a layout plan for where to place each element
                layout_plan = []

                # Create a legend spanning all rows in the last column
                legend_position = gs[:, -1]

                # Create positions for the network plots (excluding the legend column)
                available_positions = []
                for r in range(rows):
                    for c in range(cols - 1):
                        available_positions.append((r, c))

                # Place the non-hub network first (if available)
                if has_non_hub:
                    # Place non-hub network in the first position
                    row, col = available_positions.pop(0)
                    if num_hubs <= 3 and cols > 3:
                        # For few hubs, let the non-hub network span two columns
                        non_hub_position = gs[row, col:(col + 2)]
                        # Remove the next position due tospanning
                        if available_positions and available_positions[0][0] == row and available_positions[0][1] == col + 1:
                            available_positions.pop(0)
                    else:
                        non_hub_position = gs[row, col]

                    layout_plan.append(('non-hub', non_hub_position))

                # Place hub networks in the remaining positions
                hub_names = sorted(hub_networks.keys())
                for i, hub in enumerate(hub_names):
                    if i < len(available_positions):
                        row, col = available_positions[i]
                        hub_position = gs[row, col]
                        layout_plan.append((hub, hub_position))
                    else:
                        warnings.warn(f"Too many hubs to display. Showing only {len(available_positions)} out of {num_hubs}. \
                                      \n Consider increasing the hub_threshold or reducing the n_top parameter.")
                        break

                # Add the legend to the layout plan
                layout_plan.append(('legend', legend_position))

                # Now draw networks according to the layout plan
                for item_type, position in layout_plan:
                    if item_type == 'non-hub':
                        ax_non_hub = fig.add_subplot(position)
                        draw_network(non_hub_subgraph, ax_non_hub, "Non-Hub Nodes")
                    elif item_type == 'legend':
                        ax_legend = fig.add_subplot(position)
                        ax_legend.axis('off')
                    else:
                        # This is a hub
                        hub = item_type
                        ax_hub = fig.add_subplot(position)

                        # Get hub info
                        hub_gene = G.nodes[hub].get('gene', '')
                        hub_cell_type = G.nodes[hub].get('cell_type', '')
                        hub_type = "Receptor" if G.nodes[hub].get('is_receptor', False) else "Ligand"

                        # Create descriptive title
                        hub_title = f"Hub: {hub_gene} ({hub_type}) in {hub_cell_type}\n{len(hub_networks[hub].edges())} connections"

                        draw_network(hub_networks[hub], ax_hub, hub_title)

                # Calculate which cell types are present in this graph
                present_cell_types = set()
                for n in G.nodes():
                    ct = G.nodes[n].get('cell_type')
                    if ct is not None:
                        present_cell_types.add(ct)

                # Create sorted list of present cell types
                present_cell_types = sorted(present_cell_types)

                # Create legend elements
                legend_elements = [
                    matplotlib.patches.Patch(facecolor='white', edgecolor='black', label='Cell Types:', alpha=0.7)
                ]

                # Add elements for each cell type that exists in this graph
                for ct in sorted(present_cell_types):
                    legend_elements.append(
                        lines.Line2D(
                            [0], [0],
                            color=cell_colors[ct],
                            marker=cell_shapes[ct],
                            markersize=10,
                            linestyle='none',
                            markeredgecolor='black',
                            markeredgewidth=0.7,
                            label=ct
                        )
                    )

                # Add explanation of score differences
                if direction_name == 'positive':
                    colorbar_title = f'Edges colored by quantile rank differences: Interaction is higher in {condition_b}'
                elif direction_name == 'negative':
                    colorbar_title = f'Edges colored by quantile rank differences: Interaction is higher in {condition_a}'

                # Distribute legend
                if rows >= 2:
                    # Split legend elements between cell types and other info
                    split_idx = 1 + len(present_cell_types)

                    # Create section for cell types
                    cell_type_elements = legend_elements[:split_idx]

                    # Add cell types legend at the top of the legend column
                    legend = ax_legend.legend(
                        handles=cell_type_elements,
                        loc="upper center",
                        fontsize=12,
                        framealpha=0.9,
                        title_fontsize=12,
                        handlelength=1.5,
                        handletextpad=0.6,
                        labelspacing=1.0,
                        borderpad=1,
                        ncol=1
                    )
                    ax_legend.add_artist(legend)
                else:
                    # For single row layout
                    ax_legend.legend(
                        handles=legend_elements,
                        loc="center",
                        fontsize=12,
                        framealpha=0.9,
                        title_fontsize=12,
                        handlelength=1.5,
                        handletextpad=0.6,
                        labelspacing=1.0,
                        borderpad=1,
                        ncol=1
                    )

                # Add main title for the entire figure
                if hub_nodes:
                    hub_info = f"{len(hub_nodes)} Hubs (≥{hub_threshold} connections)"
                else:
                    hub_info = ""

                main_title = f"{condition_b} - {condition_a} | {direction_label} | {hub_info}"
                fig.suptitle(main_title, fontsize=16, y=0.98)
                plt.subplots_adjust(bottom=0.15)

                # Add colorbar for edge colors at the bottom of the figure
                sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
                sm.set_array([])
                cbar_ax = fig.add_axes([0.35, 0.06, 0.3, 0.02])
                cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal", shrink=0.6, pad=0.15)
                cbar.set_label(colorbar_title, fontsize=12)
                cbar.ax.tick_params(labelsize=12)

                arrow_ax = fig.add_axes([0.15, 0.06, 0.25, 0.02])
                arrow_ax.axis('off')

                arrow_x_start = 0.2
                arrow_x_end = 0.25
                arrow_y = 0.5

                arrow_line = lines.Line2D(
                    [arrow_x_start, arrow_x_end],
                    [arrow_y, arrow_y],
                    color='black',
                    linewidth=2,
                    transform=arrow_ax.transAxes
                )
                arrow_ax.add_artist(arrow_line)

                arrow_head = lines.Line2D(
                    [arrow_x_end], [arrow_y],
                    marker='>',
                    markersize=10,
                    color='black',
                    transform=arrow_ax.transAxes
                )
                arrow_ax.add_artist(arrow_head)

                arrow_ax.text(
                    0.30, 0.5,
                    "Edge Direction:\nLigand to Receptor",
                    fontsize=12,
                    ha='left',
                    va='center',
                    transform=arrow_ax.transAxes
                )

                # Save if requested
                if save:
                    plt.savefig(f"{settings.figure_dir}/{save}", bbox_inches='tight')

                # Store the figure for return
                figures.append(fig)

    # Return the list of figures
    return figures


@beartype
def plot_all_condition_differences(
    diff_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = None,
    n_top: int = 100,
    figsize: Tuple[int, int] = (22, 16),
    dpi: int = 300,
    save_prefix: Optional[str] = None,
    split_by_direction: bool = True,
    hub_threshold: int = 4,
    show: bool = True,
    return_figures: bool = False
) -> Optional[Dict[str, List[matplotlib.figure.Figure]]]:
    """
    Plot network visualizations for all condition differences.

    This is a wrapper function that calls condition_differences_network on all differences
    found in the output from calculate_condition_differences.

    Parameters
    ----------
    diff_results : Dict[str, Dict[str, Dict[str, pd.DataFrame]]], default None
        Results from calculate_condition_differences or calculate_condition_differences_over_time.
    n_top : int, default 20
        Number of top differential interactions to display in each network
    figsize : Tuple[int, int], default (22, 16)
        Size of the figure
    dpi : int, default 300
        The resolution of the figures
    save_prefix : Optional[str], default None
        Prefix for saved figures. If provided, each figure will be saved with this prefix
        followed by the condition dimension and comparison name
    split_by_direction : bool, default True
        Whether to create separate networks for positive and negative differences
    hub_threshold : int, default 4
        Minimum number of connections for a node to be considered a hub
    show : bool, default True
        Whether to display the figures
    return_figures : bool, default False
        Whether to return the generated figures as a dictionary

    Returns
    -------
    Optional[Dict[str, List[matplotlib.figure.Figure]]]
        If return_figures is True, returns a dictionary mapping condition dimensions
        to lists of generated figures

    Raises
    ------
    ValueError
        If no diff_results are provided
    """
    # Get the differences results if not provided
    if diff_results is None:
        raise ValueError(
            "Run calculate_condition_differences or calculate_condition_differences_over_time first, "
            "or provide diff_results."
        )

    if not diff_results or all(len(dimension) == 0 for dimension in diff_results.values()):
        raise ValueError("No condition differences found in the provided results.")

    # Dictionary to store all generated figures if return_figures is True
    all_figures = {} if return_figures else None

    # Process each condition dimension
    for condition_dim, comparisons in diff_results.items():
        print(f"Processing condition dimension: {condition_dim}")

        # Skip if no comparisons in this dimension
        if len(comparisons) == 0:
            warnings.warn(f"No comparisons found for dimension '{condition_dim}', skipping")
            continue

        # Create save name if saving is requested
        if save_prefix:
            save_name = f"{save_prefix}_{condition_dim}"
        else:
            save_name = None

        # Call condition_differences_network for this dimension
        try:
            # Create subdictionary with just this dimension
            subset_diff_results = {condition_dim: comparisons}

            figures = condition_differences_network(
                diff_results=subset_diff_results,
                n_top=n_top,
                figsize=figsize,
                dpi=dpi,
                save=save_name,
                split_by_direction=split_by_direction,
                hub_threshold=hub_threshold
            )

            # Store figures if requested
            if return_figures:
                all_figures[condition_dim] = figures

            # Display figures if requested
            if show:
                for fig in figures:
                    plt.figure(fig.number)
                    plt.show()
            else:
                for fig in figures:
                    plt.close(fig)

            print(f"Generated {len(figures)} network plots for '{condition_dim}'")

        except Exception as e:
            warnings.warn(f"Error plotting networks for dimension '{condition_dim}': {str(e)}")

    return all_figures if return_figures else None


@beartype
def track_clusters_or_genes(
    diff_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    clusters: Optional[List[str]] = None,
    genes: Optional[List[str]] = None,
    timepoint_order: List[str] = None,
    min_interactions: int = 5,
    n_top: int = 100,
    figsize: Tuple[int, int] = (22, 16),
    dpi: int = 300,
    save_prefix: Optional[str] = None,
    split_by_direction: bool = True,
    hub_threshold: int = 4
) -> List[matplotlib.figure.Figure]:
    """
    Track the evolution of interactions involving specific clusters or genes across timepoints.

    Parameters
    ----------
    diff_results : Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
        Results from calculate_condition_differences_over_time
    clusters : Optional[List[str]], default None
        List of cluster names to track. If None, no cluster filtering is applied.
    genes : Optional[List[str]], default None
        List of gene names to track. If None, no gene filtering is applied.
    timepoint_order : List[str], default None
        Ordered list of timepoints. If None, will attempt to infer from the data.
    min_interactions : int, default 5
        Minimum number of interactions required to generate a network for a timepoint
    n_top : int, default 20
        Number of top differential interactions to display in each network
    figsize : Tuple[int, int], default (22, 16)
        Size of the figure
    dpi : int, default 300
        The resolution of the figure
    save_prefix : Optional[str], default None
        Prefix for saved figures
    split_by_direction : bool, default True
        Whether to create separate networks for positive and negative differences
    hub_threshold : int, default 4
        Minimum number of connections for a node to be considered a hub

    Returns
    -------
    List[matplotlib.figure.Figure]
        List of generated figures

    Raises
    ------
    ValueError
        If neither clusters nor genes are provided
        If timepoint_order is not provided
        If no temporal comparison is found in the provided results
        If no comparisons have at least min_interactions matching the specified clusters/genes
    """

    # Validate inputs
    if clusters is None and genes is None:
        raise ValueError("At least one of clusters or genes must be provided.")

    if timepoint_order is None:
        raise ValueError("timepoint_order must be provided")

    # Find the time dimension (e. g. "time_series")
    time_dim = None
    for dim in diff_results:
        if diff_results[dim]:
            time_dim = dim
            break

    if not time_dim:
        raise ValueError("No temporal comparison found in the provided results.")

    # Extract all timepoints from the comparisons
    available_timepoints = set()
    time_comparisons = {}

    for comp_key, comp_data in diff_results[time_dim].items():
        if 'differences' not in comp_data:
            continue

        diff_df = comp_data['differences']

        if not hasattr(diff_df, 'attrs'):
            continue

        tp1 = diff_df.attrs.get('timepoint_1')
        tp2 = diff_df.attrs.get('timepoint_2')

        if tp1 and tp2:
            available_timepoints.add(tp1)
            available_timepoints.add(tp2)
            time_comparisons[comp_key] = (tp1, tp2)

    # Create filtered version of diff_results containing only the specified clusters/genes
    filtered_results = {time_dim: {}}

    for comp_key, comp_data in diff_results[time_dim].items():
        if 'differences' not in comp_data:
            continue

        diff_df = comp_data['differences']
        filtered_df = diff_df.copy()

        # Apply cluster filter if provided
        if clusters is not None:
            cluster_mask = (
                filtered_df['receptor_cluster'].isin(clusters) | filtered_df['ligand_cluster'].isin(clusters)
            )
            filtered_df = filtered_df[cluster_mask]

        # Apply gene filter if provided
        if genes is not None:
            gene_mask = (
                filtered_df['receptor_gene'].isin(genes) | filtered_df['ligand_gene'].isin(genes)
            )
            filtered_df = filtered_df[gene_mask]

        # Only add if enough interactions
        if len(filtered_df) >= min_interactions:
            for attr_name in diff_df.attrs:
                filtered_df.attrs[attr_name] = diff_df.attrs[attr_name]
            filtered_results[time_dim][comp_key] = {'differences': filtered_df}
        else:
            print(f"Skipping {comp_key}: Only {len(filtered_df)} interactions match criteria (minimum {min_interactions})")

    if not any(filtered_results[time_dim].values()):
        raise ValueError(f"No comparisons have at least {min_interactions} interactions matching the specified clusters/genes.")

    # Generate network visualizations for each timepoint
    figures = condition_differences_network(
        diff_results=filtered_results,
        n_top=n_top,
        figsize=figsize,
        dpi=dpi,
        save=save_prefix,
        split_by_direction=split_by_direction,
        hub_threshold=hub_threshold
    )

    # Add filter information to titles
    for fig in figures:
        current_title = fig._suptitle.get_text() if fig._suptitle else ""
        filter_desc = []
        if clusters:
            filter_desc.append(f"Clusters: {', '.join(clusters)}")
        if genes:
            filter_desc.append(f"Genes: {', '.join(genes)}")
        filter_info = " | ".join(filter_desc)
        new_title = f"{current_title}\nFiltered by: {filter_info}"
        fig.suptitle(new_title)

    return figures
