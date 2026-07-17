"""Plot velocity related figures e.g. pseudo-time heatmap."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import scanpy as sc

import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco
from sctoolbox.plotting.general import _save_figure

# type hint imports
from beartype.typing import Optional, Tuple, Any
from beartype import beartype
from matplotlib.axes import Axes


####################################################################################################
#                        Plots correlating gene expression with pseudotime                         #
####################################################################################################

@deco.log_anndata
@beartype
def pseudotime_heatmap(adata: sc.AnnData,
                       genes: list[str],
                       sortby: Optional[str] = None,
                       layer: Optional[str] = None,
                       figsize: Optional[Tuple[int | float, int | float]] = None,
                       shrink_cbar: int | float = 0.5,
                       title: Optional[str] = None,
                       save: Optional[str] = None,
                       **kwargs: Any) -> Axes:
    """
    Plot heatmap of genes along pseudotime sorted by 'sortby' column in adata.obs.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object
    genes : list[str]
        List of genes for heatmap.
    sortby : Optional[str], default None
        Sort genes by condition
    layer : Optional[str], default None
        Use different layer of anndata object.
    figsize : Optional[Tuple[int | float, int | float]], default None
        Tuple of integers setting the heatmap figsize.
    shrink_cbar : int | float, default 0.5
        Shrink color bar by set ratio.
    title : Optional[str], default None
        Set title for plot.
    save : Optional[str], default None
        Path and name of file to be saved.
    **kwargs : Any
        Additional arguments passed to seaborn.heatmap.

    Returns
    -------
    ax : Axes
        Axes object containing the plot.
    """

    adata_sub = adata[:, genes].copy()

    # Sort adata
    if sortby is not None:
        obs_sorted = adata_sub.obs.sort_values(sortby)
        adata_sub = adata_sub[obs_sorted.index, :]

    # Collect matrix
    if layer is not None:
        mat = adata_sub.layers[layer]
    else:
        mat = adata_sub.X

    mat = mat.todense() if issparse(mat) else mat
    mat = mat.T     # pseudotime on x-axis

    # Convert to pandas dataframe
    mat = pd.DataFrame(mat)
    mat.index = genes

    # z-score normalize per row
    mat = utils.tables.table_zscore(mat)

    # Plot heatmap
    n_genes = len(mat)
    n_cells = mat.shape[1]
    nrows = 1

    if figsize is None:
        figsize = (6, n_genes / 5)

    # Setup figure
    fig, axarr = plt.subplots(nrows, 1, sharex=True, figsize=figsize)  # , height_ratios=(1, len(mat)))
    axarr = [axarr] if type(axarr).__name__.startswith("Axes") else axarr
    i = 0

    parameters = {"center": 0,
                  "cmap": "bwr"}
    parameters.update(kwargs)
    ax = sns.heatmap(mat, ax=axarr[i],
                     yticklabels=True,  # make sure all labels are shown
                     cbar_kws={"label": "Expr.z-score",
                               "shrink": shrink_cbar,
                               "anchor": (0, 0),
                               "aspect": 20 * shrink_cbar * 2},  # width of cbar after shrink by adjusting aspect
                     **parameters)
    ax.set_xticks([])  # remove x-ticks

    if title is not None:
        ax.set_title(title)

    # Draw pseudotime arrow below heatmap
    ax.annotate('', xy=(0, n_genes + 1), xycoords=ax.transData, xytext=(n_cells, n_genes + 1),
                arrowprops=dict(arrowstyle="<-", color='black'))
    ax.text(n_cells / 2, n_genes + 1.2, f"Pseudotime (n={n_cells:,} cells)", transform=ax.transData, ha="center", va="top")

    # Save figure
    _save_figure(save)

    return ax
