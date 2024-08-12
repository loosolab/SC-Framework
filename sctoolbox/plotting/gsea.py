"""Plots for gsea analysis."""

import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
from gseapy import enrichment_map
import networkx as nx
import numpy as np

from beartype import beartype
from beartype.typing import Optional, Any, Literal, Tuple

from sctoolbox.plotting.general import clustermap_dotplot, _save_figure
from sctoolbox.utils.bioutils import pseudobulk_table
import sctoolbox.utils.decorator as deco


@deco.log_anndata
@beartype
def term_dotplot(term: str,
                 term_table: pd.DataFrame,
                 adata: sc.AnnData,
                 groupby: str,
                 gene_col: str = "Lead_genes",
                 term_col: str = "Term",
                 groups: Optional[list[str] | str] = None,
                 hue: Literal["Mean Expression", "Zscore"] = "Zscore",
                 **kwargs: Any) -> list:
    """
    Plot mean expression and zscore of cluster for one GO-term.

    Parameters
    ----------
    term : str
        Name of GO-term, e.g 'ATP Metabolic Process (GO:0046034)'
    term_table : pd.DataFrame
        Table of GO-term enriched genes.
        Output of sctoolbox.tools.gsea.gene_set_enrichment().
    adata : sc.AnnData
        Anndata object.
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
    list
        List of matplotlib.axes.Axes objects containing the dotplot and the dendrogram(s).

    Raises
    ------
    ValueError
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
    # get related genes
    active_genes = list(set(term_table.loc[term_table[term_col] == term][gene_col].str.split(";").explode()))
    active_genes = list(map(str.upper, active_genes))

    # get index name
    index_name = "index" if not adata.var.index.name else adata.var.index.name

    if not active_genes:
        raise ValueError(f"No genes matching the term '{term}' found in term_table")

    # subset adata to active genes
    subset = adata[:, adata.var.index.str.upper().isin(active_genes)]

    bulks = pseudobulk_table(subset, groupby=groupby)

    if groups:
        bulks = bulks.loc[:, groups]
    # convert from wide to long format
    long_bulks = pd.melt(bulks.reset_index(),
                         id_vars=index_name,
                         var_name=groupby,
                         value_name="Mean Expression").rename({index_name: "Gene"}, axis=1)

    zscore = stats.zscore(bulks.T).T

    long_zscore = pd.melt(zscore.reset_index(),
                          id_vars=index_name,
                          var_name=groupby,
                          value_name="Zscore").rename({index_name: "Gene"}, axis=1)

    # combine expression and zscores
    comb = long_bulks.merge(long_zscore, on=["Gene", groupby])

    return clustermap_dotplot(comb, x=groupby, y="Gene", title=term, size="Mean Expression", hue=hue, **kwargs)


@beartype
def gsea_network(enr_res: pd.DataFrame,
                 clust_col: str = "Cluster",
                 sig_col: Literal["Adjusted P-value", "P-value", "FDR q-val", "NOM p-val"] = "Adjusted P-value",
                 cutoff: int | float = 0.05,
                 scale: int | float = 1,
                 resolution: int | float = 0.35,
                 figsize: Optional[Tuple[int | float, int | float]] = None,
                 ncols: int = 3,
                 save: Optional[str] = None,
                 ) -> None:
    """
    Plot GO network per cluster.

    Node size corresponds to the percentage of gene overlap in a certain term of interest.
    Colour of the node corresponds to the significance of the enriched terms.
    Edge size corresponds to the number of genes that overlap between the two connected nodes.

    Parameters
    ----------
    enr_res : pd.DataFrame
        Dataframe containing 2D gsea results.
    clust_col : str, default 'Cluster'
        Column name of cluster annotation in enr_res.
    sig_col : Literal['Adjusted P-value', 'P-value', 'FDR q-val', 'NOM p-val'], default 'Adjusted P-value'
        Column containing significance of enrichted termn.
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

    Raises
    ------
    ValueError
        If no cluster with valied pathways are found.
    """

    # Get cluster with enrichted pathways after filtering
    nodes, _ = enrichment_map(enr_res, column=sig_col, cutoff=cutoff)
    valid_cluster = list(nodes.Cluster.unique())

    if len(valid_cluster) == 0:
        raise ValueError("No cluster with enrichted pathways found.")

    # Plot setup
    num_clust = len(valid_cluster)
    ncols = min(ncols, num_clust)
    nrows = int(np.ceil(num_clust/ncols))
    figsize = figsize if figsize else (8 * ncols, 6 * nrows)
    fig, axarr = plt.subplots(nrows, ncols, figsize=figsize)
    axarr = np.array(axarr).reshape((-1, 1)) if ncols == 1 else axarr    # reshape 1-column array
    axarr = np.array(axarr).reshape((1, -1)) if nrows == 1 else axarr  # reshape 1-row array
    axes = axarr.flatten()

    for i, cluster in enumerate(valid_cluster):
        # Create cluster subset
        tmp = enr_res[enr_res[clust_col] == cluster]
        
        # Calculate enrichment map
        nodes, edges = enrichment_map(tmp, column=sig_col, cutoff=cutoff)

        # build graph
        G = nx.from_pandas_edgelist(edges,
                                    source='src_idx',
                                    target='targ_idx',
                                    edge_attr=['jaccard_coef', 'overlap_coef', 'overlap_genes'])

        # init node cooridnates
        pos=nx.layout.spiral_layout(G, scale=scale, resolution=resolution)

        # draw node
        nx.draw_networkx_nodes(G,
                               pos=pos,
                               cmap=plt.cm.RdYlBu,
                               node_color=list(nodes.NES),
                               node_size=list(nodes.Hits_ratio *500),
                               ax=axes[i],
                               label="a")
        # draw node label
        nx.draw_networkx_labels(G,
                                pos=pos,
                                labels=nodes.Term.to_dict(),
                                font_size=10,
                                ax=axes[i],
                                clip_on=False)
        # draw edge
        edge_weight = nx.get_edge_attributes(G, 'jaccard_coef').values()
        nx.draw_networkx_edges(G,
                               pos=pos,
                               width=list(map(lambda x: x*10, edge_weight)),
                               ax=axes[i],
                               edge_color='#CDDBD4') 

        # Set title to subplots
        axes[i].title.set_text(cluster)

    # Hide plots not filled in
    for ax in axes[num_clust:]:
        ax.axis('off')

    # Add title
    fig.suptitle("Network of enrichted Pathways per Cluster")

    fig.tight_layout()
    _save_figure(save)
