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
import seaborn as sns
import igraph as ig
import pycirclize
from tqdm import tqdm
import warnings
import logging
import liana.resource as liana_res

from beartype.typing import Optional, Tuple
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
                                normalize: int = 1000,
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
        Corresponds to `download_db(ligand_column, receptor_column)`.
        Uses index when None.
    normalize : int, default 1000
        Correct clusters to given size.
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

    # cluster scaling factor for cluster size correction
    scaling_factor = {k: v / normalize for k, v in clust_sizes.items()}

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
    for _, (receptor, ligand) in tqdm(adata.uns["receptor-ligand"]["database"][[r_col, l_col]].iterrows(),
                                      total=len(adata.uns["receptor-ligand"]["database"]),
                                      desc="finding receptor-ligand interactions"):
        # skip interaction if not in data
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
    interactions["receptor_score"] = interactions["receptor_score"] * interactions["receptor_scale_factor"]
    interactions["ligand_score"] = interactions["ligand_score"] * interactions["ligand_scale_factor"]
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
             hide_edges: Optional[list[Tuple[str, str]]] = None) -> npt.ArrayLike:
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
    graph.vs['size'] = 0.1  # node size
    graph.vs['label_size'] = 12  # label size
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
