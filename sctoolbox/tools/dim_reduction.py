"""
Tools for dimensionality reduction with PCA/SVD.
"""

import numpy as np
import scanpy as sc
import scipy
from scipy.sparse.linalg import svds
from kneed import KneeLocator
from anndata import AnnData

import sctoolbox.tools.highly_variable as hv
import sctoolbox.tools.norm_correct as nc


############################################################################
#                             PCA / SVD                                    #
############################################################################

def compute_PCA(anndata, use_highly_variable=True, inplace=False, **kwargs):
    """
    Compute PCA and add information to adata.uns['infoprocess']

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object to add the PCA to.
    use_highly_variable : boolean, default True
        If true, use highly variable genes to compute PCA.
    inplace : boolean, default False
        Whether the anndata object is modified inplace.
    **kwargs :
        Additional parameters forwarded to scanpy.pp.pca().

    Returns
    -------
    anndata.AnnData or None:
        Returns anndata object with PCA components. Or None if inplace = True.
    """
    adata_m = anndata if inplace else anndata.copy()

    # Computing PCA
    print("Computing PCA")
    sc.pp.pca(adata_m, use_highly_variable=use_highly_variable, **kwargs)

    # Adding info in anndata.uns["infoprocess"]
    # cr.build_infor(adata_m, "Scanpy computed PCA", "use_highly_variable= " + str(use_highly_variable), inplace=True)

    if not inplace:
        return adata_m


def norm_log_PCA(anndata, exclude_HEG=True, use_HVG_PCA=True, inplace=False):
    """
    Defining the ideal number of highly variable genes (HGV), annotate them and compute PCA.

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object to work on.
    exclude_HEG : boolean, default True
        If True, highly expressed genes (HEG) will be not considered in the normalization.
    use_HVG_PCA : boolean, default True
        If true, highly variable genes (HVG) will be also considered to calculate PCA.
    inplace : boolean, default False
        Whether to work inplace on the anndata object.

    Returns
    -------
    anndata.Anndata or None:
        Anndata with expression values normalized and log converted and PCA computed.
    """
    adata_m = anndata if inplace else anndata.copy()

    # Normalization and converting to log
    nc.adata_normalize_total(adata_m, exclude_HEG, inplace=True)

    # Annotate highly variable genes
    hv.annot_HVG(adata_m, inplace=True)

    # Compute PCA
    compute_PCA(adata_m, use_highly_variable=use_HVG_PCA, inplace=True)

    if not inplace:
        return adata_m


def lsi(data, scale_embeddings=True, n_comps=50):
    """Run Latent Semantic Indexing.

    Note: Function is from muon package.

    :param anndata.AnnData data: AnnData object with peak counts.
    :param bool scale_embeddings: Scale embeddings to zero mean and unit variance, defaults to True.
    :param int n_comps: Number of components to calculate with SVD, defaults to 50.
    :raises TypeError: data must be anndata object.
    """
    if isinstance(data, AnnData):
        adata = data
    else:
        raise TypeError("Expected AnnData object!")

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


def apply_svd(adata, layer=None):
    """ Singular value decomposition of anndata object.

    Parameters
    -----------
    adata : anndata.AnnData
        The anndata object to be decomposed.
    layer : string, optional
        The layer to be decomposed. If None, the layer is set to "X". Default: None.

    Returns:
    --------
    adata : anndata.AnnData
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

def define_PC(anndata):
    """
    Define threshold for most variable PCA components.

    Note: Function expects PCA to be computed beforehand.

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object with PCA to get significant PCs threshold from.

    Returns
    -------
    int :
        An int representing the number of PCs until elbow, defining PCs with significant variance.
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


def subset_PCA(adata, n_pcs, start=0, inplace=True):
    """
    Subset the PCA coordinates in adata.obsm["X_pca"] to the given number of pcs.
    Additionally, subset the PCs in adata.varm["PCs"] and the variance ratio in adata.uns["pca"]["variance_ratio"].

    Parameters
    -----------
    adata : anndata.AnnData
        Anndata object containing the PCA coordinates.
    n_pcs : int
        Number of PCs to keep.
    start : int, default 0
        Index (0-based) of the first PC to keep. E.g. if start = 1 and n_pcs = 10, you will exclude the first PC to keep 9 PCs.
    inplace : bool, default True
        Whether to work inplace on the anndata object.

    Returns
    --------
    adata or None
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


def atac_norm(adata, method):  # , condition_col='nb_features'):
    """A function that normalizes count matrix using two methods (total and TFIDF) seperately,
    calculates PCA and UMAP and plots both UMAPs.

    :param anndata.AnnData adata: AnnData object with peak counts.
    :param str condition_col: Name of the column to use as color in the umap plot, defaults to 'nb_features'
    :param bool remove_pc1: Removing first component after TFIDF normalization and LSI, defaults to True
    :return anndata.AnnData: Two AnnData objects with normalized matrices (Total and TFIDF) and UMAP.
    """

    adata = adata.copy()  # make sure the original data is not modified

    if method == "total":  # perform total normalization and pca
        print('Performing total normalization and PCA...')
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.pca(adata)

    elif method == "tfidf":
        print('Performing TFIDF and LSI...')
        nc.tfidf(adata)
        lsi(adata)  # corresponds to PCA

    else:
        raise ValueError("Method must be either 'total' or 'tfidf'")

    return adata


"""
    # perform tfidf and latent semantic indexing
    print('Performing TFIDF and LSI...')

    sc.pp.neighbors(adata_tfidf, n_neighbors=15, n_pcs=50, method='umap', metric='euclidean', use_rep='X_pca')
    sc.tl.umap(adata_tfidf, min_dist=0.1, spread=2)
    print('Done')

    # perform total normalization and pca
    print('Performing total normalization and PCA...')
    sc.pp.normalize_total(adata_total)
    adata_total.layers['normalised'] = adata_total.X.copy()
    epi.pp.log1p(adata_total)
    sc.pp.pca(adata_total, svd_solver='arpack', n_comps=50, use_highly_variable=False)
    sc.pp.neighbors(adata_total, n_neighbors=15, n_pcs=50, method='umap', metric='euclidean')
    sc.tl.umap(adata_total, min_dist=0.1, spread=2)
    print('Done')

    print('Plotting UMAP...')
    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axes = axarr.flatten()
    sc.pl.pca(adata_tfidf, color=condition_col, title='TFIDF', legend_loc='none', ax=axes[0], show=False)
    sc.pl.pca(adata_total, color=condition_col, title='Total', legend_loc='right margin', ax=axes[1], show=False)
    sc.pl.umap(adata_tfidf, color=condition_col, title='', legend_loc='none', ax=axes[2], show=False)
    sc.pl.umap(adata_total, color=condition_col, title='', legend_loc='right margin', ax=axes[3], show=False)

    plt.tight_layout()

    return adata_tfidf, adata_total
"""
