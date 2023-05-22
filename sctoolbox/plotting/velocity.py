import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import issparse

import sctoolbox.utils as utils


####################################################################################################
#                        Plots correlating gene expression with pseudotime                         #
####################################################################################################

def pseudotime_heatmap(adata, genes,
                       sortby=None,
                       layer=None,
                       figsize=None,
                       shrink_cbar=0.5,
                       title=None,
                       save=None,
                       **kwargs):
    """ Plot heatmap of genes along pseudotime sorted by 'sortby' column in adata.obs. """

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
    mat = utils.table_zscore(mat)

    # Plot heatmap
    n_genes = len(mat)
    n_cells = mat.shape[1]
    nrows = 1

    if figsize is None:
        figsize = (6, n_genes / 5)

    # Setup figure
    fig, axarr = plt.subplots(nrows, 1, sharex=True, figsize=figsize)  # , height_ratios=(1, len(mat)))
    axarr = [axarr] if type(axarr).__name__.startswith("AxesSubplot") else axarr
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
    utils.save_figure(save)

    return ax
