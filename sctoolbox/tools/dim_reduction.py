"""Tools for dimensionality reduction with PCA/SVD."""
import numpy as np
import scanpy as sc
import scipy
from scipy.sparse.linalg import svds
from kneed import KneeLocator

from beartype.typing import Optional, Any, Literal, List, Union
from beartype import beartype

import sctoolbox.tools.embedding as scem
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
from sctoolbox.utils.io import update_yaml
logger = settings.logger


############################################################################
#                             PCA / SVD                                    #
############################################################################

@deco.log_anndata
@beartype
def compute_PCA(anndata: sc.AnnData,
                mask_var: Optional[str | List] = "highly_variable",
                inplace: bool = False,
                report: bool = False,
                **kwargs: Any) -> Optional[sc.AnnData]:
    """
    Compute a principal component analysis.

    Parameters
    ----------
    anndata : sc.AnnData
        Anndata object to add the PCA to.
    mask_var : Optional[str], default 'highly_variable'
        To run only on a certain set of genes given by a boolean array or a string referring to an array in var. By default, uses .var['highly_variable'] if available, else everything.
    inplace : bool, default False
        Whether the anndata object is modified inplace.
    report : bool, default False
        Will add information to the report methods if `sctoolbox.settings.report_dir` is set.
    **kwargs : Any
        Additional parameters forwarded to scanpy.pp.pca().

    Returns
    -------
    Optional[sc.AnnData]
        Returns anndata object with PCA components. Or None if inplace = True.
    """

    adata_m = anndata if inplace else anndata.copy()

    # Computing PCA
    logger.info("Computing PCA")
    sc.pp.pca(adata_m, mask_var=mask_var, **kwargs)

    # generate method report
    if settings.report_dir and report:
        # method
        update_yaml(d={"hvg": int(adata_m.var["highly_variable"].sum()), "pc_count": adata_m.obsm["X_pca"].shape[1]},
                    yml="method.yml", path_prefix="report")

    if not inplace:
        return adata_m


@beartype
def lsi(data: sc.AnnData,
        scale_embeddings: bool = True,
        n_comps: int = 50,
        use_highly_variable: bool = False) -> None:
    """
    Run Latent Semantic Indexing for dimensionality reduction.

    Values represent the similarity of cells in the original space.
    doi: 10.3389/fphys.2013.00008

    Parameters
    ----------
    data : sc.AnnData
        AnnData object with peak counts.
    scale_embeddings : bool, default True
        Scale embeddings to zero mean and unit variance.
    n_comps : int, default 50
        Number of components to calculate with SVD.
    use_highly_variable : bool, default True
        If true, use highly variable genes to compute LSI.

    Raises
    ------
    ValueError
        If highly variable genes are used and not found in adata.var['highly_variable'].

    Notes
    -----
    Function is taken from muon package.
    """

    adata = data

    # Subset adata to highly variable genes
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise ValueError("Highly variable genes not found in adata.var['highly_variable'].")
        adata_comp = adata[:, adata.var['highly_variable']]
    else:
        adata_comp = adata

    # In an unlikely scnenario when there are less 50 features, set n_comps to that value
    n_comps = min(n_comps, adata_comp.X.shape[1])

    # logging.info("Performing SVD")
    cell_embeddings, svalues, peaks_loadings = svds(adata_comp.X, k=n_comps)

    # Re-order components in the descending order
    cell_embeddings = cell_embeddings[:, ::-1]
    svalues = svalues[::-1]
    peaks_loadings = peaks_loadings[::-1, :]

    if scale_embeddings:
        cell_embeddings = (cell_embeddings - cell_embeddings.mean(axis=0)) / cell_embeddings.std(
            axis=0
        )

    # try for dense then sparse matrix
    try:
        power = adata_comp.X ** 2
    except TypeError:
        power = adata_comp.X.power(2)

    var_explained = np.round(svalues ** 2 / np.sum(power), decimals=3)
    stdev = svalues / np.sqrt(adata_comp.X.shape[0] - 1)

    # Add results to adata
    adata.obsm["X_lsi"] = cell_embeddings
    # if highly variable genes are used, only store the loadings for those genes and set the rest to 0
    if use_highly_variable:
        adata.varm["LSI"] = np.zeros(shape=(adata.n_vars, n_comps))
        adata.varm["LSI"][adata.var['highly_variable']] = peaks_loadings.T
    else:
        adata.varm["LSI"] = peaks_loadings.T

    # Add variance explained to uns
    adata.uns["lsi"] = {"stdev": stdev,
                        "variance": svalues,
                        "variance_ratio": var_explained}

    # Save to PCA to make it compatible with scanpy
    adata.obsm["X_pca"] = adata.obsm["X_lsi"]
    adata.varm["PCs"] = adata.varm["LSI"]
    adata.uns["pca"] = adata.uns["lsi"]


@beartype
def apply_svd(adata: sc.AnnData,
              layer: Optional[str] = None) -> sc.AnnData:
    """
    Singular value decomposition of anndata object.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object to be decomposed.
    layer : Optional[str], default None
        The layer to be decomposed. If None, the layer is set to "X".

    Returns
    -------
    sc.AnnData
        The decomposed anndata object containing .obsm, .varm and .uns information.
    """

    if layer is None:
        mat = adata.X
    else:
        mat = adata.layers[layer]

    # SVD
    u, s, v = scipy.sparse.linalg.svds(mat, k=30, which="LM")  # find largest variance

    # u/s/v are reversed in scipy.sparse.linalg.svds:
    s = s[::-1]
    u = np.fliplr(u)
    v = np.flipud(v)

    # Visualize explained variance
    var_explained = np.round(s**2 / np.sum(s**2), decimals=3)

    adata.obsm["X_svd"] = u
    adata.varm["SVs"] = v.T
    adata.uns["svd"] = {"variance": s,
                        "variance_ratio": var_explained}

    # Hack to use the scanpy functions on SVD coordinates
    # adata.obsm["X_pca"] = adata.obsm["X_svd"]
    # adata.varm["PCs"] = adata.varm["SVs"]
    # adata.uns["pca"] = adata.uns["svd"]

    return adata


############################################################################
#                         Subset number of PCs                             #
############################################################################

@beartype
def propose_pcs(anndata: sc.AnnData,
                how: List[Literal["variance", "cumulative variance", "correlation"]] = ["variance", "correlation"],
                var_method: Literal["knee", "percent"] = "percent",
                perc_thresh: Union[int, float] = 30,
                corr_thresh: float = 0.3,
                variance_column: Optional[str] = "variance_ratio",
                corr_kwargs: Optional[dict] = {}) -> List[int]:
    """
    Propose a selection of PCs that can be used for further analysis.

    Note: Function expects PCA to be computed beforehand.

    Parameters
    ----------
    anndata: sc.AnnData
        Anndata object with PCA to get PCs from.
    how: List[Literal["variance", "cumulative variance", "correlation"]], default ["variance", "correlation"]
        Values to use for PC proposal. Will independently apply filters to all selected methods and return the intersection of PCs.
    var_method: Literal["knee", "percent"], default "percent"
        Either define a threshold based on a knee algorithm or use the percentile.
    perc_thresh: Union[int, float], default 30
        Percentile threshold of the PCs that should be included. Only for var_method="percent" and expects a value from 0-100.
    corr_thresh: float, default 0.3
        Filter PCs with a correlation greater than the given value.
    variance_column: Optional[str], default "variance_ratio"
        Column in anndata.uns to use for variance calculation.
    corr_kwargs: Optional(dict), default None
        Parameters forwarded to `sctoolbox.tools.correlation_matrix`.

    Returns
    -------
    List[int]
        List of PCs proposed for further use.

    Raises
    ------
    ValueError
        If PCA is not found in anndata.
    """

    # check if pca exists
    if "pca" not in anndata.uns or variance_column not in anndata.uns["pca"]:
        raise ValueError("PCA not found! Please make sure to compute PCA before running this function.")

    # setup PC names
    PC_names = np.arange(1, len(anndata.uns["pca"][variance_column]) + 1, dtype=int)

    variance = anndata.uns["pca"][variance_column]
    variance = variance * 100  # convert to percent

    selected_pcs = []

    if "variance" in how:

        if var_method == "knee":
            # compute knee
            kn = KneeLocator(PC_names, variance, curve='convex', direction='decreasing')
            knee = int(kn.knee)  # cast from numpy.int64

            selected_pcs.append(set(pc for pc in PC_names if pc <= knee))
        elif var_method == "percent":
            # compute percentile
            percentile = np.percentile(a=variance, q=100 - perc_thresh)

            selected_pcs.append(set(pc for pc, var in zip(PC_names, variance) if var >= percentile))

    if "cumulative variance" in how:

        cumulative = np.cumsum(variance)

        if var_method == "knee":
            # compute knee
            kn = KneeLocator(PC_names, cumulative, curve='concave', direction='increasing')
            knee = int(kn.knee)  # cast from numpy.int64

            selected_pcs.append(set(pc for pc in PC_names if pc <= knee))
        elif var_method == "percent":
            # compute percentile
            percentile = np.percentile(a=cumulative, q=perc_thresh)

            selected_pcs.append(set(pc for pc, cum in zip(PC_names, cumulative) if cum <= percentile))

    if "correlation" in how:
        # color by highest absolute correlation
        corrcoefs, _ = scem.correlation_matrix(adata=anndata, **corr_kwargs)

        abs_corrcoefs = list(corrcoefs.abs().max(axis=0))

        selected_pcs.append(set(pc for pc, cc in zip(PC_names, abs_corrcoefs) if cc < corr_thresh))

    # create overlap of selected PCs
    selected_pcs = list(set.intersection(*selected_pcs))

    # convert numpy.int64 to int
    selected_pcs = [int(x) for x in selected_pcs]

    return selected_pcs


@deco.log_anndata
@beartype
def subset_PCA(adata: sc.AnnData,
               n_pcs: Optional[int] = None,
               start: int = 0,
               select: Optional[List[int]] = None,
               inplace: bool = True,
               report: bool = False) -> Optional[sc.AnnData]:
    """
    Subset the PCA coordinates in adata.obsm["X_pca"] to the given number of pcs.

    Additionally, subset the PCs in adata.varm["PCs"] and the variance ratio in adata.uns["pca"]["variance_ratio"].

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object containing the PCA coordinates.
    n_pcs : Optional[int], default None
        Number of PCs to keep.
    start : int, default 0
        Index (0-based) of the first PC to keep. E.g. if start = 1 and n_pcs = 10, you will exclude the first PC to keep 9 PCs.
    select : Optional[List[int]], default None
        Provide a list of PC numbers to keep. E.g. [2, 3, 5] will select the second, third and fifth PC. Will overwrite the n_pcs and start parameter.
    inplace : bool, default True
        Whether to work inplace on the anndata object.
    report : bool, default False
        Will add information to the report methods if `sctoolbox.settings.report_dir` is set.

    Returns
    -------
    Optional[sc.AnnData]
        Anndata object with the subsetted PCA coordinates. Or None if inplace = True.
    """

    if inplace is False:
        adata = adata.copy()

    if select:
        # adjust selection to be 0-based
        select = [i - 1 for i in select]

        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, select]
        adata.varm["PCs"] = adata.varm["PCs"][:, select]

        if "variance_ratio" in adata.uns.get("pca", {}):
            adata.uns["pca"]["variance_ratio"] = adata.uns["pca"]["variance_ratio"][select]
    else:
        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, start:n_pcs]
        adata.varm["PCs"] = adata.varm["PCs"][:, start:n_pcs]

        if "variance_ratio" in adata.uns.get("pca", {}):
            adata.uns["pca"]["variance_ratio"] = adata.uns["pca"]["variance_ratio"][start:n_pcs]

    # generate method report
    if settings.report_dir and report:
        # method
        update_yaml(d={"pc_count": len(select) if select else n_pcs - start},
                    yml="method.yml", path_prefix="report")

    if inplace is False:
        return adata


############################################################################
#                      All in one dimension reduction                      #
############################################################################

# TODO doesn't use @deco.log_anndata as it is currently intended as a primarily internal convenience function.
@beartype
def dim_red(anndata: sc.AnnData, method: Optional[Literal["PCA", "LSI"]], method_kwargs: dict = {}, subset: Optional[List[int]] = None, neighbor_kwargs: dict = {}, inplace: bool = False) -> Optional[sc.AnnData]:
    """
    Compute a dimension reduction, select components and create a neighbor graph.

    An all in one dimension reduction function that is intended to be used to quickly compute all dimension reduction steps.
    E.g. to reproduce an analysis where all parameters are known already.

    Parameters
    ----------
    anndata : sc.AnnData
        The object to dimension reduce.
    method : Optional[Literal["PCA", "LSI"]]
        Either do a PCA (:func:`sctoolbox.tools.dim_reduction.comput_PCA`) or LSI (:func:`sctoolbox.tools.dim_reduction.lsi`).
        Will skip the dimension reduction if None (make sure it was computed beforehand).
    method_kwargs : dict, default {}
        Parameters of the chosen method.
    subset : Optional[List[int]], default None
        A list of integers specifing a subset of components to keep. Forwarded to "select" of :func:sctoolbox.`tools.dim_reduction.subset_PCA`.
    neighbor_kwargs : dict, default {}
        Parameters to the neighbor graph computation. :func:`scanpy.pp.neighbors`
    inplace : bool, default False
        Whether to modify the anndata object inplace.

    Returns
    -------
    Optional[sc.AnnData]
        The AnnData with dimension reduction and neighbor graph.
    """
    if not inplace:
        anndata = anndata.copy()

    # dimension reduction
    if method == "PCA":
        compute_PCA(anndata, **method_kwargs, inplace=True)
    elif method == "LSI":
        lsi(anndata, **method_kwargs)  # this is always inplace

    # component subset
    if subset:
        subset_PCA(anndata, select=subset, inplace=True)

    # neighbor graph
    sc.pp.neighbors(anndata, **neighbor_kwargs, copy=False)

    if not inplace:
        return anndata
