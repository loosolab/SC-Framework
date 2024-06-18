"""Plots for gsea analysis."""

import scipy.stats as stats
import pandas as pd
import scanpy as sc

from beartype import beartype
from beartype.typing import Optional, Any, Literal

from sctoolbox.plotting.general import clustermap_dotplot
from sctoolbox.utils.bioutils import pseudobulk_table
import sctoolbox.utils.decorator as deco


@deco.log_anndata
@beartype
def term_dotplot(term: str,
                 term_table: pd.DataFrame,
                 adata: sc.AnnData,
                 groupby: str,
                 groups: Optional[list[str] | str] = None,
                 hue: Literal["Mean Expression", "Zscore"] = "Zscore",
                 **kwargs: Any) -> list:
    """
    Plot mean expression and zscore of cluster for one GO-term.

    Parameters
    ----------
    term: str
        Name of GO-term, e.g 'ATP Metabolic Process (GO:0046034)'
    term_table: pd.DataFrame
        Table of GO-term enriched genes.
        Output of sctoolbox.tools.gsea.enrichr_marker_genes().
        The DataFrame needs to contain the columns:
            'Term', 'Genes'
    adata: sc.AnnData
        Anndata object.
    groupby: str
        Key from `adata.obs` to group cells by.
    groups: Optional[list[str] | str], default None
        Set subset of group column.
    hue: Literal["Mean Expression", "Zscore"], default "Zscore"
        Choose dot coloring.
    **kwargs: Any
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

        term_table = pd.DataFrame({
            "Term": "Actin Filament Organization (GO:0007015)",
            "Genes": ["COBL", "WIPF1;SH3KBP1"]
        })

        pl.gsea.term_dotplot(term="Actin Filament Organization (GO:0007015)",
                             term_table=term_table,
                             adata=adata,
                             groupby="louvain")

    """
    # get related genes
    active_genes = list(set(term_table.loc[term_table["Term"] == term]["Genes"].str.split(";").explode()))
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
