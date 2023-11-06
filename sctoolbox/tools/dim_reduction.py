"""Tools for dimensionality reduction with PCA/SVD."""
import numpy as np
import scanpy as sc
import scipy
from scipy.sparse.linalg import svds
from kneed import KneeLocator

from typing import Optional
from beartype import beartype

import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


############################################################################
#                             PCA / SVD                                    #
############################################################################

@deco.log_anndata
@beartype
def compute_PCA(anndata: sc.AnnData,
                use_highly_variable: bool = True,
                inplace: bool = False,
                **kwargs: dict) -> Optional[sc.AnnData]:
    """
    Compute PCA and add information to adata.uns['infoprocess'].

    Parameters
    ----------
    anndata : sc.AnnData
        Anndata object to add the PCA to.
    use_highly_variable : bool, default True
        If true, use highly variable genes to compute PCA.
    inplace : bool, default False
        Whether the anndata object is modified inplace.
    **kwargs : dict
        Additional parameters forwarded to scanpy.pp.pca().

    Returns
    -------
    Optional[sc.AnnData]
        Returns anndata object with PCA components. Or None if inplace = True.
    """

    adata_m = anndata if inplace else anndata.copy()

    # Computing PCA
    logger.info("Computing PCA")
    sc.pp.pca(adata_m, use_highly_variable=use_highly_variable, **kwargs)

    # Adding info in anndata.uns["infoprocess"]
    # cr.build_infor(adata_m, "Scanpy computed PCA", "use_highly_variable= " + str(use_highly_variable), inplace=True)

    if not inplace:
        return adata_m


# wrapper no longer used
# @beartype
# def norm_log_PCA(anndata: sc.AnnData,
#                 exclude_HEG: bool = True,
#                 use_HVG_PCA: bool = True,
#                 inplace: bool = False):
#    """
#    Defining the ideal number of highly variable genes (HGV), annotate them and compute PCA.
#
#    Parameters
#    ----------
#    anndata : sc.AnnData
#        Anndata object to work on.
#    exclude_HEG : boolean, default True
#        If True, highly expressed genes (HEG) will be not considered in the normalization.
#    use_HVG_PCA : boolean, default True
#        If true, highly variable genes (HVG) will be also considered to calculate PCA.
#    inplace : boolean, default False
#        Whether to work inplace on the anndata object.
#
#    Returns
#    -------
#    sc.AnnData or None:
#        Anndata with expression values normalized and log converted and PCA computed.
#    """
#    adata_m = anndata if inplace else anndata.copy()
#
#    # Normalization and converting to log
#    nc.adata_normalize_total(adata_m, exclude_HEG, inplace=True)
#
#    # Annotate highly variable genes
#    hv.annot_HVG(adata_m, inplace=True)
#
#    # Compute PCA
#    compute_PCA(adata_m, use_highly_variable=use_HVG_PCA, inplace=True)
#
#    if not inplace:
#        return adata_m


@beartype
def lsi(data: sc.AnnData,
        scale_embeddings: bool = True,
        n_comps: int = 50) -> None:
    """
    Run Latent Semantic Indexing.

    Parameters
    ----------
    data : sc.AnnData
        AnnData object with peak counts.
    scale_embeddings : boolean, default True
        Scale embeddings to zero mean and unit variance.
    n_comps : int, default 50
        Number of components to calculate with SVD.

    Notes
    -----
    Function is taken from muon package.

    Raises
    ------
    TypeError
        data must be anndata object.
    """

    adata = data

    # In an unlikely scnenario when there are less 50 features, set n_comps to that value
    n_comps = min(n_comps, adata.X.shape[1])

    # logging.info("Performing SVD")
    cell_embeddings, svalues, peaks_loadings = svds(adata.X, k=n_comps)

    # Re-order components in the descending order
    cell_embeddings = cell_embeddings[:, ::-1]
    svalues = svalues[::-1]
    peaks_loadings = peaks_loadings[::-1, :]

    if scale_embeddings:
        cell_embeddings = (cell_embeddings - cell_embeddings.mean(axis=0)) / cell_embeddings.std(
            axis=0
        )

    var_explained = np.round(svalues ** 2 / np.sum(svalues ** 2), decimals=3)
    stdev = svalues / np.sqrt(adata.X.shape[0] - 1)

    adata.obsm["X_lsi"] = cell_embeddings
    adata.varm["LSI"] = peaks_loadings.T
    adata.uns["lsi"] = {"stdev": stdev,
                        "variance": svalues,
                        "variance_ratio": var_explained}

    # Save to PCA to make it compatible with scanpy
    adata.obsm["X_pca"] = adata.obsm["X_lsi"]
    adata.varm["PCs"] = adata.varm["LSI"]
    adata.uns["pca"] = adata.uns["lsi"]


@beartype
def apply_svd(adata: sc.AnnData,
              layer: str = None) -> sc.AnnData:
    """
    Singular value decomposition of anndata object.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object to be decomposed.
    layer : str, default None
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
def define_PC(anndata: sc.AnnData) -> int:
    """
    Define threshold for most variable PCA components.

    Note: Function expects PCA to be computed beforehand.

    Parameters
    ----------
    anndata : sc.AnnData
        Anndata object with PCA to get significant PCs threshold from.

    Returns
    -------
    int
        An int representing the number of PCs until elbow, defining PCs with significant variance.

    Raises
    ------
    ValueError:
        If PCA is not found in anndata.
    """

    # check if pca exists
    if "pca" not in anndata.uns or "variance_ratio" not in anndata.uns["pca"]:
        raise ValueError("PCA not found! Please make sure to compute PCA before running this function.")

    # prepare values
    y = anndata.uns["pca"]["variance_ratio"]
    x = range(1, len(y) + 1)

    # compute knee
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    knee = int(kn.knee)  # cast from numpy.int64

    # Adding info in anndata.uns["infoprocess"]
    # cr.build_infor(anndata, "PCA_knee_threshold", knee)

    return knee


@deco.log_anndata
@beartype
def subset_PCA(adata: sc.AnnData,
               n_pcs: int,
               start: int = 0,
               inplace: bool = True) -> Optional[sc.AnnData]:
    """
    Subset the PCA coordinates in adata.obsm["X_pca"] to the given number of pcs.

    Additionally, subset the PCs in adata.varm["PCs"] and the variance ratio in adata.uns["pca"]["variance_ratio"].

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object containing the PCA coordinates.
    n_pcs : int
        Number of PCs to keep.
    start : int, default 0
        Index (0-based) of the first PC to keep. E.g. if start = 1 and n_pcs = 10, you will exclude the first PC to keep 9 PCs.
    inplace : bool, default True
        Whether to work inplace on the anndata object.

    Returns
    -------
    Optional[sc.AnnData]
        Anndata object with the subsetted PCA coordinates. Or None if inplace = True.
    """

    if inplace is False:
        adata = adata.copy()

    adata.obsm["X_pca"] = adata.obsm["X_pca"][:, start:n_pcs]
    adata.varm["PCs"] = adata.varm["PCs"][:, start:n_pcs]

    if "variance_ratio" in adata.uns.get("pca", {}):
        adata.uns["pca"]["variance_ratio"] = adata.uns["pca"]["variance_ratio"][start:n_pcs]

    if inplace is False:
        return adata
