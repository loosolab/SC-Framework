"""Plots for highly variable genes, e.g. as a result of sc.tl.highly_variable."""

import matplotlib.pyplot as plt
import sctoolbox.utils.decorator as deco
import sctoolbox.utils as utils
from sctoolbox.plotting.general import _save_figure
import scanpy as sc

from beartype import beartype
from beartype.typing import Any, Optional


@deco.log_anndata
@beartype
def violin_HVF_distribution(adata: sc.AnnData, save: Optional[str] = None, **kwargs: Any):
    """
    Plot the distribution of the HVF as violinplot.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing columns ['highly_variable', 'n_cells_by_counts'] column.
    save: Optional[str]
        Path to save figure. Uses `sctoolbox.settings.figdir`.
    **kwargs : Any
        Keyword arguments to be passed to matplotlib.pyplot.violinplot.
    """
    utils.checker.check_columns(adata.var, ['highly_variable', 'n_cells_by_counts'])
    # get the number of cells per highly variable feature
    hvf_var = adata.var[adata.var['highly_variable']]  # 'highly_variable' is a boolean column
    n_cells = hvf_var['n_cells_by_counts']
    n_cells.reset_index(drop=True, inplace=True)
    # violin plot
    fig, ax = plt.subplots()
    ax.violinplot(n_cells, showmeans=True, showmedians=True, **kwargs)
    ax.set_title('Distribution of the number of cells per highly variable feature')
    ax.set_ylabel('Number of cells')
    ax.set_xlabel('Highly variable features')

    # save
    if save:
        _save_figure(save)

    plt.show()


@deco.log_anndata
@beartype
def scatter_HVF_distribution(adata: sc.AnnData, **kwargs: Any):
    """
    Plot the distribution of the HVF as scatterplot.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing columns ['variability_score', 'n_cells'] column.
    **kwargs : Any
        Keyword arguments to be passed to matplotlib.pyplot.scatter.
    """
    utils.checker.check_columns(adata.var, ['variability_score', 'n_cells'])
    variabilities = adata.var[['variability_score', 'n_cells']]
    fig, ax = plt.subplots()
    ax.scatter(variabilities['n_cells'], variabilities['variability_score'], **kwargs)
    ax.set_title('Distribution of the number of cells and variability score per feature')
    ax.set_xlabel('Number of cells')
    ax.set_ylabel('variability score')
    plt.show()
