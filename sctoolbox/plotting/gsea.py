"""Plots for gsea analysis."""

import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scanpy as sc
from gseapy import enrichment_map
import networkx as nx
import numpy as np
import warnings

from beartype import beartype
from beartype.typing import Optional, Any, Literal, Tuple
from numpy.typing import NDArray

from sctoolbox.plotting.general import clustermap_dotplot, _save_figure
from sctoolbox.utils.bioutils import pseudobulk_table
from sctoolbox.utils.checker import check_columns
import sctoolbox.utils.decorator as deco

from sctoolbox import settings
logger = settings.logger


@deco.log_anndata
@beartype
def term_dotplot(adata: sc.AnnData,
                 term: str,
                 groupby: str,
                 gene_col: str = "Lead_genes",
                 term_col: str = "Term",
                 groups: Optional[list[str] | str] = None,
                 hue: Literal["Mean Expression", "Zscore"] = "Zscore",
                 **kwargs: Any) -> NDArray[Axes]:
    """
    Plot mean expression and zscore of cluster for one GO-term.

    Parameters
    ----------
    adata : sc.anndata
        Anndata object containing adata.uns['gsea']['enrichment_table']
    term : str
        Name of GO-term, e.g 'ATP Metabolic Process (GO:0046034)'
    groupby : str
        Key from `adata.obs` to group cells by.
    gene_col : str, default "Lead_genes"
        Name of column the containg gene (lists).
        Typical names used by different methods:
            prerank -> 'Lead_genes'
            enrichr -> 'Genes'
    term_col : str, default 'Term'
        Name of column containg the terms.
    groups : Optional[list[str] | str], default None
        Set subset of group column.
    hue : Literal["Mean Expression", "Zscore"], default "Zscore"
        Choose dot coloring.
    **kwargs : Any
        Additional parameters for sctoolbox.plotting.general.clustermap_dotplot

    Notes
    -----
    All genes will be converted to uppercase for comparison.

    Returns
    -------
    NDArray[Axes]
        Array of Axes objects containing the dotplot and the dendrogram(s).

    Raises
    ------
    ValueError
        If gsea results cannot be found in adata.uns.
        If no genes are matching the given term in term_table

    Examples
    --------
    .. plot::
        :context: close-figs

        # --- hide: start ---
        import pandas as pd
        import sctoolbox.plotting as pl
        # --- hide: stop ---

        term_table = pd.DataFrame({
            "Term": "Actin Filament Organization (GO:0007015)",
            "Lead_genes": ["COBL", "WIPF1;SH3KBP1"]
        })

        pl.gsea.term_dotplot(term="Actin Filament Organization (GO:0007015)",
                             term_table=term_table,
                             adata=adata,
                             groupby="louvain")

    """
    if "gsea" not in adata.uns.keys():
        msg = "Could not find gsea results. Please run 'tools.gsea.gene_set_enrichment' before running this function."
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Get enrichment table.")
    term_table = adata.uns['gsea']['enrichment_table'].copy()

    check_columns(term_table, columns=[term_col])

    # get related genes
    logger.info("Get term related genes.")
    active_genes = list(set(term_table.loc[term_table[term_col] == term][gene_col].str.split(";").explode()))
    active_genes = list(map(str.upper, active_genes))

    # get index name
    index_name = "index" if not adata.var.index.name else adata.var.index.name

    if not active_genes:
        msg = f"No genes matching the term '{term}' found in term_table"
        logger.error(msg)
        raise ValueError(msg)

    # subset adata to active genes
    logger.info("Subset AnnData object.")
    subset = adata[:, adata.var.index.str.upper().isin(active_genes)]

    bulks = pseudobulk_table(subset, groupby=groupby)

    if groups:
        bulks = bulks.loc[:, groups]
    # convert from wide to long format
    long_bulks = pd.melt(bulks.reset_index(),
                         id_vars=index_name,
                         var_name=groupby,
                         value_name="Mean Expression").rename({index_name: "Gene"}, axis=1)

    logger.info("Calculate Z scores.")
    zscore = pd.DataFrame(stats.zscore(bulks.T).T)
    zscore.index = bulks.index
    zscore.columns = bulks.columns

    long_zscore = pd.melt(zscore.reset_index(),
                          id_vars=index_name,
                          var_name=groupby,
                          value_name="Zscore").rename({index_name: "Gene"}, axis=1)

    long_bulks[groupby] = long_bulks[groupby].astype(str)
    long_zscore[groupby] = long_zscore[groupby].astype(str)

    # combine expression and zscores
    comb = pd.merge(long_bulks, long_zscore, on=["Gene", groupby], how="outer")

    return clustermap_dotplot(comb, x=groupby, y="Gene", title=term, size="Mean Expression", hue=hue, **kwargs)


@beartype
def gsea_network(adata: sc.AnnData,
                 score_col: Optional[str] = None,
                 clust_col: str = "Cluster",
                 sig_col: Optional[Literal['Adjusted P-value', 'P-value', 'FDR q-val', 'NOM p-val']] = None,
                 cutoff: int | float = 0.05,
                 scale: int | float = 1,
                 resolution: int | float = 0.35,
                 figsize: Optional[Tuple[int | float, int | float]] = None,
                 ncols: int = 3,
                 save: Optional[str] = None,
                 ) -> None:
    """
    Plot GO network per cluster.

    Node size corresponds to the percentage of marker genes of the term available in the cluster.
    Colour of the node corresponds to the significance of the enriched terms.
    Edge size corresponds to the number of genes that overlap between the two connected nodes.

    Parameters
    ----------
    adata : sc.anndata
        Anndata object containing adata.uns['gsea']['enrichment_table']
    score_col : str, default 'NES'
        Name of enrichment scoring column.
    clust_col : Optional[str], default None
        Column name in adata.uns['gsea']['enrichment_table'] containing enrichment score.
        If None uses default stored in adata.uns['gsea']['score_col']
    sig_col : Optional[Literal['Adjusted P-value', 'P-value', 'FDR q-val', 'NOM p-val']], default None
        Column containing significance of enrichted termn.
        If None uses default stored in adata.uns['gsea']['stat_col']
    cutoff : int | float, default 0.05
        Set cutoff for sig_col. Only nodes with value < cutoff are shown.
    scale : int | float, default 1
        Scale factor for node positions.
    resolution : int | float, 0.35
        The compactness of the spiral layout returned.
        Lower values result in more compressed spiral layouts.
    figsize : Optional[Tuple[int | float, int | float]], default None
        Set size of figure, if None uses default settings: (8 * ncols, 6 * nrows)
    ncols :  int, default 3,
        Set number of columns for plot.
    save : Optional[str], default None
        Filename suffix to save the figure.
        The cluster name is added as prefix to the name.

    Notes
    -----
    Default values expect input dataframe to be prerank output.

    Raises
    ------
    ValueError
        If gsea key is not found in adata.uns.
        If no cluster with valid pathways are found.
    """
    if "gsea" not in adata.uns.keys():
        msg = "Could not find gsea results. Please run 'tools.gsea.gene_set_enrichment' before running this function."
        logger.error(msg)
        raise ValueError(msg)

    term_table = adata.uns['gsea']['enrichment_table'].copy()
    sig_col = sig_col if sig_col else adata.uns['gsea']['stat_col']
    score_col = score_col if score_col else adata.uns['gsea']['score_col']

    check_columns(term_table, columns=[score_col, clust_col, sig_col])

    logger.info("Calculate enrichment map...")
    with warnings.catch_warnings():
        # hide future warnings until gseapy fixes them
        warnings.filterwarnings(action='ignore', message=".*Series.replace.*|.*chained assignment.*")

        # Get cluster with enrichted pathways after filtering
        nodes, _ = enrichment_map(term_table, column=sig_col, cutoff=cutoff)
        min_NES, max_NES = min(list(nodes[score_col])), max(list(nodes[score_col]))
        node_count = nodes.Cluster.value_counts()
        valid_cluster = list(node_count[node_count > 1].index)

    if len(valid_cluster) == 0:
        msg = "No cluster with enrichted pathways found."
        logger.error(msg)
        raise ValueError(msg)

    # Plot setup
    logger.info("Plotting network...")
    num_clust = len(valid_cluster)
    ncols = min(ncols, num_clust)
    nrows = int(np.ceil(num_clust / ncols))
    figsize = figsize if figsize else (8 * ncols, 6 * nrows)
    fig, axarr = plt.subplots(nrows, ncols, figsize=figsize)
    axarr = np.array(axarr).reshape((-1, 1)) if ncols == 1 else axarr  # reshape 1-column array
    axarr = np.array(axarr).reshape((1, -1)) if nrows == 1 else axarr  # reshape 1-row array
    axes = axarr.flatten()

    node_sizes = list()
    width_sizes = list()
    for i, cluster in enumerate(valid_cluster):
        # Create cluster subset
        tmp = term_table[term_table[clust_col] == cluster]

        with warnings.catch_warnings():
            # hide future warnings until gseapy fixes them
            warnings.filterwarnings(action='ignore', message=".*Series.replace.*|.*chained assignment.*")

            # Calculate enrichment map
            nodes, edges = enrichment_map(tmp, column=sig_col, cutoff=cutoff)

        # build graph
        G = nx.from_pandas_edgelist(edges,
                                    source='src_idx',
                                    target='targ_idx',
                                    edge_attr=['jaccard_coef', 'overlap_coef', 'overlap_genes'])

        # Add nodes without any edges
        node_set = set(nodes.index)
        edge_set = set(edges.src_idx) | set(edges.targ_idx)
        if not node_set.issubset(list(edge_set)):
            for j in node_set ^ edge_set:
                G.add_node(j)
                # Move node without any edges to end of Dataframe to correct order
                nodes.loc[len(nodes)] = nodes.iloc[j, :]
                nodes.drop(j, inplace=True)
                nodes.index = range(len(nodes))

        # init node cooridnates
        pos = nx.layout.spiral_layout(G, scale=scale, resolution=resolution)

        # draw node
        n = nx.draw_networkx_nodes(G,
                                   pos=pos,
                                   cmap=plt.cm.RdYlBu,
                                   node_color=list(nodes[score_col]),
                                   node_size=list(nodes.Hits_ratio * 500),
                                   vmin=min_NES,
                                   vmax=max_NES,
                                   linewidths=0,
                                   ax=axes[i],
                                   label=nodes.Term)

        # draw node label
        nx.draw_networkx_labels(G,
                                pos=pos,
                                labels=nodes["Term"].to_dict(),
                                font_size=10,
                                ax=axes[i],
                                clip_on=False)

        # draw edge
        edge_weight = nx.get_edge_attributes(G, 'jaccard_coef').values()
        edge_width = list(map(lambda x: x * 10, edge_weight))
        nx.draw_networkx_edges(G,
                               pos=pos,
                               width=edge_width,
                               ax=axes[i],
                               edge_color='#CDDBD4')

        # Set subplot title
        axes[i].title.set_text(cluster)

        # Draw colorbar
        # https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes('bottom', size='5%', pad=0)
        fig.colorbar(n, cax=cax, orientation='horizontal', label=score_col)

        # Get sizes for legend
        node_sizes += list(nodes.Hits_ratio)
        width_sizes += edge_width

    # Hide plots not filled in
    for ax in axes[num_clust:]:
        ax.axis('off')

    # Add title
    fig.suptitle("Network of enrichted Pathways per Cluster", fontsize=20)

    # ---------------------------- legend -------------------------------------
    step_num = 5

    # Edge legend
    width_sizes = [0, 1] if not width_sizes else width_sizes
    s_steps = np.linspace(min(width_sizes), max(width_sizes), step_num)
    line_list = [Line2D([], [], color='black', alpha=1, linewidth=s, label=f"{np.round(s / 10, 2)}") for s in s_steps]
    line_list.insert(0, Line2D([], [], alpha=0, label="Shared significant genes\nbetween Pathways"))

    # Node size legend
    # Note The markersize does not fit 100% to the actial scatter size.
    # The *20 scaling is an approximation.
    line_list.append(Line2D([], [], alpha=0, label="Ratio of \nPathway Genes in Cluster"))
    s_steps = np.linspace(min(node_sizes), max(node_sizes), step_num)
    line_list += [Line2D([], [], color='black', alpha=1, linewidth=0, markersize=s * 20, marker="o", label=f"{np.round(s, 2)}") for s in s_steps]

    fig.legend(handles=line_list,
               bbox_to_anchor=(1, 0.5, 0, 0),
               fontsize=15,
               labelspacing=1,
               loc='center left')

    # ---------------------------- save figure --------------------------------

    fig.tight_layout()
    _save_figure(save)


def cluster_dotplot(adata: sc.AnnData,
                    cluster_col: str = "Cluster",
                    sig_col: Optional[Literal['Adjusted P-value', 'P-value', 'FDR q-val', 'NOM p-val']] = None,
                    top_up: int = 5,
                    top_down: int = 5,
                    cutoff: float = 0.05,
                    save_figs: bool = True,
                    save_prefix: str = "",
                    **kwargs: Any) -> dict:
    """
    Plot up/down regulated pathways per cluster.

    Parameters
    ----------
    adata : sc.anndata
        Anndata object containing adata.uns['gsea']['enrichment_table']
    cluster_col : str, default 'Cluster'
        Cluster column name.
    sig_col : Optional[Literal['Adjusted P-value', 'P-value', 'FDR q-val', 'NOM p-val']], default None
        Column containing significance of enrichted termn.
        If None uses default stored in adata.uns['gsea']['stat_col']
    top_up : int, default 5
        Number of upregulated pathways to be plotted.
    top_down : int, default 5
        Number of downregulated pathways to be plotted.
    cutoff : float, default 0.05
        Filter cutoff for sig_col.
    save_figs : bool, default True
        If True, save each plot.
    save_prefix : str, default ""
        Prefix for filenames.
    **kwargs : Any
        Additional parameters for sctoolbox.plotting.gsea.gsea_dot

    Returns
    -------
    dict
        Dictionary with cluster name as key and dotplot axes object as value.

    Raises
    ------
    ValueError
        If gsea results cannot be found in adata.uns.
    """
    if "gsea" not in adata.uns.keys():
        msg = "Could not find gsea results. Please run 'tools.gsea.gene_set_enrichment' before running this function."
        logger.error(msg)
        raise ValueError(msg)

    term_table = adata.uns['gsea']['enrichment_table'].copy()
    sig_col = sig_col if sig_col else adata.uns['gsea']['stat_col']
    score_col = adata.uns['gsea']['score_col']

    check_columns(term_table, columns=[cluster_col, sig_col])

    dotplots = dict()

    for c in term_table[cluster_col].unique():
        logger.info(f"Plotting dotplot for cluster {c}")
        tmp = term_table[(term_table[cluster_col] == c) & (term_table[sig_col] <= cutoff)].copy()
        tmp[score_col] = tmp[score_col].astype(float)
        tmp = pd.concat([tmp.nlargest(top_up, score_col), tmp.nsmallest(top_down, score_col)])
        tmp.sort_values(score_col, inplace=True, ascending=False)
        if tmp.empty:
            logger.info(f"No enrichted pathways of cluster {c} found with cutoff set to {cutoff}")
            continue
        save = f"{save_prefix}_GSEA_dotplot_top_pathways_per_cluster_{c}.pdf" if save_figs else None
        empty_adata = sc.AnnData()
        empty_adata.uns["gsea"] = adata.uns['gsea'].copy()
        empty_adata.uns["gsea"]["enrichment_table"] = tmp
        axes = gsea_dot(empty_adata, save=save, title=f"Top regulated pathways of cluster {c}", **kwargs)
        dotplots[c] = axes

    return dotplots


def gsea_dot(adata: sc.AnnData,
             sig_col: Optional[Literal['Adjusted P-value', 'P-value', 'FDR q-val', 'NOM p-val']] = None,
             x: Optional[str] = None,
             cluster_col: str = "Cluster",
             cutoff: float = 0.05,
             top_term: Optional[int] = None,
             figsize: Tuple[int, int] = (5, 8),
             sizes: Tuple[int, int] = (50, 200),
             x_label_rotation: int = 0,
             cmap: str = "viridis",
             title: str = "Top regulated pathways",
             title_size: int = 16,
             save: Optional[str] = None) -> Axes:
    """
    Plot up/down regulated pathways.

    Parameters
    ----------
    adata : sc.anndata
        Anndata object containing adata.uns['gsea']['enrichment_table']
    sig_col : Optional[Literal['Adjusted P-value', 'P-value', 'FDR q-val', 'NOM p-val']], default None
        Column containing significance of enrichted termn.
        If None uses default stored in adata.uns['gsea']['stat_col']
    x : Optional[str], default None
        Column name in adata.uns['gsea']['enrichment_table'] containing enrichment score.
        If None uses default stored in adata.uns['gsea']['score_col']
    cluster_col : str, default 'Cluster'
        Cluster column name in adata.uns['gsea']['enrichment_table']
    cutoff : float, default 0.05
        Filter cutoff for sig_col.
    top_term : Optional[int], default None
        Select top_terms per cluster.
    figsize : Tuple[int, int], default (5, 8)
        Tuple setting the figure size.
    sizes : Tuple[int, int], default (50, 200)
        Dot size min, max tuple
    x_label_rotation : int, default 0
        Set x-tick label rotation angle
    cmap : str, default "viridis"
        Colormap for dots
    title : str, default "Top regualted pathways"
        Figure title
    title_size : int, default 16
        Title font size.
    save : Optional[str], default None
        Filename suffix to save the figure.

    Returns
    -------
    Axes
        Axes object.

    Raises
    ------
    ValueError
        If gsea results cannot be found in adata.uns.
    """
    if "gsea" not in adata.uns.keys():
        msg = "Could not find gsea results. Please run 'tools.gsea.gene_set_enrichment' before running this function."
        logger.error(msg)
        raise ValueError(msg)

    term_table = adata.uns['gsea']['enrichment_table'].copy()
    sig_col = sig_col if sig_col else adata.uns['gsea']['stat_col']
    x = x if x else adata.uns['gsea']['score_col']

    check_columns(term_table, columns=[cluster_col, x, sig_col])

    # Filter enrichment table
    logger.info("Filtering enrichment table...")
    term_table = term_table[term_table[sig_col] <= cutoff]
    if top_term:
        term_table = term_table.groupby(cluster_col).apply(
            lambda y: y.sort_values(by=x, ascending=False)
            .head(top_term)
            .reset_index(drop=True)
        ).reset_index(drop=True)

    term_table["% Genes in set"] = [
        int(round(eval(operation) * 100, 0))
        for operation in term_table[adata.uns['gsea']['overlap_col']]
    ]

    logger.info("Generating dotplot...")
    norm = plt.Normalize(term_table[sig_col].min(), term_table[sig_col].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # Create figure
    fig, ax = plt.subplots(1, figsize=figsize)
    plot = sns.scatterplot(data=term_table,
                           y="Term",
                           x=x,
                           size="% Genes in set",
                           sizes=sizes,
                           hue=sig_col,
                           palette=cmap,
                           ax=ax
                           )
    # Move legend to right side
    sns.move_legend(plot, loc='upper left', bbox_to_anchor=(1, 1, 0, 0))

    # extract the existing handles and labels
    handles, labels = ax.get_legend_handles_labels()
    i = labels.index('% Genes in set')

    ax.legend(handles[i:], labels[i:], bbox_to_anchor=(1.05, 1),
              loc=2, borderaxespad=0., fontsize=13,
              frameon=False, alignment="left")
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.4, anchor=(0.1, 0.1), label=sig_col, aspect=10)
    cbar.set_label(sig_col, rotation=0, ha="left", fontsize=13)
    ax.set_ylabel("")
    ax.set_title(title, **{"fontsize": title_size})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_label_rotation, ha='right')
    ax.grid(True, axis="y")
    fig.tight_layout()
    _save_figure(save)

    return ax
