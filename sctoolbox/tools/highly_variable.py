"""Tools to calculate and annotate highly variable genes."""
import numpy as np
from kneed import KneeLocator
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import scanpy as sc

from typing import Optional, Tuple
from beartype import beartype

import sctoolbox.utils as utils
from sctoolbox.plotting.general import _save_figure
from sctoolbox._settings import settings
import sctoolbox.utils.decorator as deco
logger = settings.logger


@deco.log_anndata
@beartype
def annot_HVG(anndata: sc.AnnData,
              min_mean: float = 0.0125,
              max_iterations: int = 10,
              hvg_range: Tuple[int, int] = (1000, 5000),
              step: int | float = 10,
              inplace: bool = True,
              save: Optional[bool] = None,
              **kwargs: dict) -> Optional[sc.AnnData]:
    """
    Annotate highly variable genes (HVG). Tries to annotate in given range of HVGs, by gradually in-/ decreasing min_mean of scanpy.pp.highly_variable_genes.

    Parameters
    ----------
    anndata : sc.AnnData
        Anndata object to annotate.
    min_mean : float, default 0.0125
        Starting min_mean parameter for finding HVGs.
    max_iterations : int, default 10
        Maximum number of min_mean adjustments.
    hvg_range : Tuple[int, int], default (1000, 5000)
        Number of HVGs should be in the given range. Will issue a warning if result is not in range.
        Default limits are chosen as proposed by https://doi.org/10.15252/msb.20188746.
    step : int | float, default 10
        Value min_mean is adjusted by in each iteration. Will divide min_value (below range) or multiply (above range) by this value.
    inplace : bool, default False
        Whether the anndata object is modified inplace.
    save : Optional[str], default None
        Path to save the plot to. If None, the plot is not saved.
    **kwargs : dict
        Additional arguments forwarded to scanpy.pp.highly_variable_genes().

    Notes
    -----
    Logarithmized data is expected.

    Returns
    -------
    Optional[sc.AnnData]
        Adds annotation of HVG to anndata object. Information is added to Anndata.var["highly_variable"].
        If inplace is False, the function returns None
        Else returns a chagned copy of the input anndata object.
    """

    adata_m = anndata if inplace else anndata.copy()

    logger.info("Annotating highy variable genes (HVG)")

    # adjust min_mean to get a HVG count in a certain range
    for i in range(max_iterations + 1):
        sc.pp.highly_variable_genes(adata_m, min_mean=min_mean, inplace=True, **kwargs)

        # counts True values in column
        hvg_count = sum(adata_m.var.highly_variable)

        # adjust min_mean
        # skip adjustment if in last iteration
        if i < max_iterations and hvg_count < hvg_range[0]:
            min_mean /= step
        elif i < max_iterations and hvg_count > hvg_range[1]:
            min_mean *= step + 0.00001  # This .000001 is to avoid an infinit loop if the current mean lie in the above if.
        else:
            break

    # warn if outside of range
    if hvg_count < hvg_range[0] or hvg_count > hvg_range[1]:
        warnings.warn(f"Number of HVGs not in range. Range is {hvg_range} but counted {hvg_count}.")
    else:
        _ = sc.pl.highly_variable_genes(anndata, show=False)  # Plot dispersion of HVG

        _save_figure(save)
        logger.info("Total HVG=" + str(anndata.var["highly_variable"].sum()))

    # Adding info in anndata.uns["infoprocess"]
    # cr.build_infor(anndata, "Scanpy annotate HVG", "min_mean= " + str(min_mean) + "; Total HVG= " + str(hvg_count), inplace=True)

    if not inplace:
        return adata_m


@deco.log_anndata
@beartype
def get_variable_features(adata: sc.AnnData,
                          max_cells: Optional[float | int] = None,
                          min_cells: Optional[float | int] = None,
                          show: bool = True,
                          inplace: bool = True) -> Optional[sc.AnnData]:
    """
    Get the highly variable features of anndata object. Adds the column "highly_variable" to adata.var. If show is True, the plot is shown.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object containing counts for variables.
    max_cells : Optional[float | int], default None
        The maximum variability score to set as threshold.
    min_cells : Optional[float | int], default None
        The minimum variability score to set as threshold.
    show : bool, default True
        Show plot of variability scores and thresholds.
    inplace : bool, default True
        If True, the anndata object is modified. Otherwise, a new anndata object is returned.

    Notes
    -----
    Designed for scATAC-seq data

    Returns
    -------
    Optional[sc.AnnData]
        If inplace is False, the function returns None
        If inplace is True, the function returns an anndata object.
    """

    utils.check_module("kneed")
    utils.check_module("statsmodels")

    if inplace is False:
        adata = adata.copy()

    # get number of cells per feature
    n_cells = adata.var['n_cells_by_counts'].sort_values(ascending=False)
    x = np.arange(len(n_cells))

    if max_cells is None:
        # Subset data to reduce computational time
        target = 10000
        step = int(len(n_cells) / target)
        if step > 0:
            idx_selection = np.arange(len(n_cells), step=step)
            n_cells = n_cells[idx_selection]
            x = x[idx_selection]

        # Smooth using lowess (prevents early finding of knees due to noise)
        n_cells = sm.nonparametric.lowess(n_cells, x, return_sorted=False, frac=0.05)

        # Find knee
        kneedle = KneeLocator(x, n_cells, curve="convex", direction="decreasing", online=False)
        max_cells = kneedle.knee_y

    # Set "highly_variable" column in var
    adata.var["highly_variable"] = (adata.var['n_cells_by_counts'] <= max_cells) & (adata.var['n_cells_by_counts'] >= min_cells)

    # Create plot
    if show is True:
        fig, ax = plt.subplots()
        ax.set_xlabel("Ranked features")
        ax.set_ylabel("Number of cells")

        ax.plot(x, n_cells)

        # Horizontal line at knee
        ax.axhline(max_cells, linestyle="--", color="r")
        xlim = ax.get_xlim()
        ax.text(xlim[1], max_cells, " {0:.2f}".format(max_cells), fontsize=12, ha="left", va="center", color="red")

        ax.axhline(min_cells, linestyle="--", color="b")
        ax.text(xlim[1], min_cells, " {0:.2f}".format(min_cells), fontsize=12, ha="left", va="center", color="blue")

    # Return the copy of the adata
    if inplace is False:
        return adata
