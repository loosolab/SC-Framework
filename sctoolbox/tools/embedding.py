"""Embedding tools."""
import scanpy as sc
import multiprocessing as mp
import pandas as pd
import itertools
import scipy
import re
import numpy as np
import warnings

from beartype.typing import Iterable, Any, Literal, Optional, Tuple
from beartype import beartype

import sctoolbox.utils as utils


@beartype
def wrap_umap(adatas: Iterable[sc.AnnData], threads: Optional[int] = None, **kwargs: Any) -> None:
    """
    Compute umap for a list of adatas in parallel.

    Parameters
    ----------
    adatas : Iterable[sc.AnnData]
        List of anndata objects to compute umap on.
    threads : Optional[int], default None
        Number of threads to use. None to use settings.get_threads.
    **kwargs : Any
        Additional arguments to be passed to sc.tl.umap.
    """
    if threads is None:
        threads = settings.get_threads()

    pool = mp.Pool(threads)

    kwargs["copy"] = True  # always copy

    jobs = []
    for i, adata in enumerate(adatas):
        adata_minimal = utils.adata.get_minimal_adata(adata)
        job = pool.apply_async(sc.tl.umap, args=(adata_minimal, ), kwds=kwargs)
        jobs.append(job)
    pool.close()

    utils.multiprocessing.monitor_jobs(jobs, "Computing UMAPs ")
    pool.join()

    # Get results and add to adatas
    for i, adata in enumerate(adatas):
        adata_return = jobs[i].get()
        adata.obsm["X_umap"] = adata_return.obsm["X_umap"]


@beartype
def correlation_matrix(adata: sc.AnnData,
                       which: Literal["obs", "var"] = "obs",
                       basis: str = "pca",
                       n_components: Optional[int] = None,
                       ignore: Optional[list[str]] = None,
                       method: Literal["spearmanr", "pearsonr"] = "spearmanr") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute a matrix of correlation values between an embedding and given columns.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    which : Literal["obs", "var"], default "obs"
        Whether to use the observations ("obs") or variables ("var") for the correlation.
    basis : str, default "pca"
        Dimensionality reduction to calculate correlation with. Must be a key in adata.obsm, or a basis available as "X_<basis>" such as "umap", "tsne" or "pca".
    n_components : int, default None
        Number of components to use for the correlation.
    ignore : Optional[list[str]], default None
        List of column names to ignore for correlation. By default (None) all numeric columns are used.
        All non numeric columns are ignored by default and cannot be used for correlation.
    method : Literal["spearmanr", "pearson"], default "spearmanr"
        Method to use for correlation. Must be either "pearsonr" or "spearmanr".

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        correlation coefficient, p-values

    Raises
    ------
    ValueError
        If "basis" is not found in data, if "which" is not "obs" or "var", if "method" is not "pearsonr" or "spearmanr", or if "which" is "var" and "basis" not "pca".
    KeyError
        If any of the given columns is not found in the respective table.
    """

    # Check that basis is in adata.obsm
    if basis not in adata.obsm:
        basis = "X_" + basis if not basis.startswith("X_") else basis  # check if basis is available as "X_<basis>"
        if basis not in adata.obsm:
            raise KeyError(f"The given basis '{basis}' cannot be found in adata.obsm. The available keys are: {list(adata.obsm.keys())}.")

    # Establish which table to use
    if which == "obs":
        table = adata.obs.copy()
        mat = adata.obsm[basis]
    elif which == "var":
        if "pca" not in basis.lower():
            raise ValueError("Correlation with 'var' can only be calculated with PCA components!")
        table = adata.var.copy()
        mat = adata.varm["PCs"]

    # Check that method is available
    try:
        corr_method = getattr(scipy.stats, method)
    except AttributeError:
        s = f"'{method}' is not a valid method within scipy.stats. Please choose one of pearsonr/spearmanr."
        raise ValueError(s)

    # Get columns
    numeric_columns = table.select_dtypes(include='number').columns.tolist()

    if ignore:
        ignore_s, numeric_columns_s = set(ignore), set(numeric_columns)
        invalid_ignore = list(ignore_s - numeric_columns_s)
        if invalid_ignore:
            warnings.warn(f"Ignore columns {invalid_ignore} not present in table {which}.")
        numeric_columns = list(numeric_columns_s - ignore_s)

    # Get table of pcs and columns
    n_components = min(n_components, mat.shape[1]) if n_components else mat.shape[1]  # make sure we don't exceed the number of pcs available
    if "pca" in basis.lower():
        comp_columns = [f"PC{i+1}" for i in range(n_components)]  # e.g. PC1, PC2, ...
    else:
        comp_columns = [f"{re.sub('^X_', '', basis.upper())}{i+1}" for i in range(n_components)]  # e.g. UMAP1, UMAP2, ...
    comp_table = pd.DataFrame(mat[:, :n_components], columns=comp_columns)
    comp_table[numeric_columns] = table[numeric_columns].reset_index(drop=True)

    # Calculate correlation of columns
    combinations = list(itertools.product(numeric_columns, comp_columns))

    corr_table = pd.DataFrame(index=numeric_columns, columns=comp_columns, dtype=float)
    pvalue_table = pd.DataFrame(index=numeric_columns, columns=comp_columns, dtype=float)
    for row, col in combinations:
        # remove NaN values and the corresponding values from both lists
        x = np.vstack([comp_table[row], comp_table[col]])  # stack values of row and column
        x = x[:, ~np.any(np.isnan(x), axis=0)]  # remove columns with NaN values

        res = corr_method(x[0], x[1])

        corr_table.loc[row, col] = res.statistic
        pvalue_table.loc[row, col] = res.pvalue

    return corr_table, pvalue_table
