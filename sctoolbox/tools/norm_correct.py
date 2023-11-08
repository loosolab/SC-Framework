"""Normalization and correction tools."""
import numpy as np
from scipy import sparse
import io
from contextlib import redirect_stderr
import copy
import multiprocessing as mp
import anndata
import scanpy as sc
import scanpy.external as sce

from typing import Optional, Any, Union, Literal, Callable
from beartype import beartype

import sctoolbox.utils as utils
from sctoolbox.tools.dim_reduction import lsi
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger

batch_methods = Literal["bbknn",
                        "combat",
                        "mnn",
                        "harmony",
                        "scanorama"]

#####################################################################
# --------------------- Normalization methods --------------------- #
#####################################################################


def atac_norm(*args: Any, **kwargs: Any):
    """Normalize ATAC data - deprecated functionality. Use normalize_adata instead."""

    logger.warning("The function 'atac_norm' is deprecated. Use 'normalize_adata' instead.")
    return normalize_adata(*args, **kwargs)


@deco.log_anndata
@beartype
def normalize_adata(adata: sc.AnnData,
                    method: str | list[str],
                    exclude_highly_expressed: bool = True,
                    target_sum: Optional[int] = None) -> dict[str, sc.AnnData]:
    """
    Normalize the count matrix and calculate dimension reduction using different methods.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    method : str | list[str]
        Normalization method. Either 'total' and/or 'tfidf'.
        - 'total': Performs normalization for total counts, log1p and PCA.
        - 'tfidf': Performs TFIDF normalization and LSI (corresponds to PCA). This method is often used for scATAC-seq data.
    exclude_highly_expressed : bool, default True
        Parameter for sc.pp.normalize_total. Decision to exclude highly expressed genes (HEG) from total normalization.
    target_sum : Optional[int], default None
        Parameter for sc.pp.normalize_total. Decide the target sum of each cell after normalization.

    Returns
    -------
    dict[str, sc.AnnData]
        Dictionary containing method name as key, and anndata as values.
        Each anndata is the annotated data matrix with normalized count matrix and PCA/LSI calculated.

    Raises
    ------
    ValueError
        If method is not valid. Needs to be either 'total' or 'tfidf'.
    """

    if isinstance(method, str):
        method = [method]

    adatas = {}
    for method_str in method:  # method is a list
        adata = adata.copy()  # make sure the original data is not modified

        if method_str == "total":  # perform total normalization and pca
            logger.info('Performing total normalization and PCA...')
            sc.pp.normalize_total(adata, exclude_highly_expressed=exclude_highly_expressed, target_sum=target_sum)
            sc.pp.log1p(adata)
            sc.pp.pca(adata)

        elif method_str == "tfidf":
            logger.info('Performing TFIDF and LSI...')
            tfidf(adata)
            lsi(adata)  # corresponds to PCA

        else:
            raise ValueError(f"Method '{method_str}' is invalid - must be either 'total' or 'tfidf'.")

        adatas[method_str] = adata

    return adatas


@beartype
def tfidf(data: sc.AnnData,
          log_tf: bool = True,
          log_idf: bool = True,
          log_tfidf: bool = False,
          scale_factor: int = 1e4) -> None:
    """
    Transform peak counts with TF-IDF (Term Frequency - Inverse Document Frequency).

    TF: peak counts are normalised by total number of counts per cell.
    DF: total number of counts for each peak.
    IDF: number of cells divided by DF.
    By default, log(TF) * log(IDF) is returned.

    Parameters
    ----------
    data : sc.AnnData
        AnnData object with peak counts.
    log_tf : bool, default True
        Log-transform TF term if True.
    log_idf : bool, default True
        Log-transform IDF term if True.
    log_tfidf : bool, default Frue
        Log-transform TF*IDF term if True. Can only be used when log_tf and log_idf are False.
    scale_factor : int, default 1e4
        Scale factor to multiply the TF-IDF matrix by.

    Notes
    -----
    Function is from the muon package.

    Raises
    ------
    AttributeError:
        log(TF*IDF) requires log(TF) and log(IDF) to be False.
    """

    adata = data

    if log_tfidf and (log_tf or log_idf):
        raise AttributeError(
            "When returning log(TF*IDF), \
            applying neither log(TF) nor log(IDF) is possible."
        )

    if sparse.issparse(adata.X):
        n_peaks = np.asarray(adata.X.sum(axis=1)).reshape(-1)
        n_peaks = sparse.dia_matrix((1.0 / n_peaks, 0), shape=(n_peaks.size, n_peaks.size))
        # This prevents making TF dense
        tf = np.dot(n_peaks, adata.X)
    else:
        n_peaks = np.asarray(adata.X.sum(axis=1)).reshape(-1, 1)
        tf = adata.X / n_peaks
    if scale_factor is not None and scale_factor != 0 and scale_factor != 1:
        tf = tf * scale_factor
    if log_tf:
        tf = np.log1p(tf)

    idf = np.asarray(adata.shape[0] / adata.X.sum(axis=0)).reshape(-1)
    if log_idf:
        idf = np.log1p(idf)

    if sparse.issparse(tf):
        idf = sparse.dia_matrix((idf, 0), shape=(idf.size, idf.size))
        tf_idf = np.dot(tf, idf)
    else:
        tf_idf = np.dot(sparse.csr_matrix(tf), sparse.csr_matrix(np.diag(idf)))

    if log_tfidf:
        tf_idf = np.log1p(tf_idf)

    adata.X = np.nan_to_num(tf_idf, 0)


@beartype
def tfidf_normalization(matrix: sparse.spmatrix,
                        tf_type: Literal["raw", "term_frequency", "log"] = "term_frequency",
                        idf_type: Literal["unary", "inverse_freq", "inverse_freq_smooth"] = "inverse_freq") -> sparse.csr_matrix:
    """
    Perform TF-IDF normalization on a sparse matrix.

    The different variants of the term frequency and inverse document frequency are obtained from https://en.wikipedia.org/wiki/Tf-idf.

    Parameters
    ----------
    matrix : scipy.sparse matrix
        The matrix to be normalized.
    tf_type : Literal["term_frequency", "log", "raw"], default "term_frequency"
        The type of term frequency to use. Can be either "raw", "term_frequency" or "log".
    idf_type : Literal["inverse_freq", "unary", "inverse_freq_smooth"], default "inverse_freq"
        The type of inverse document frequency to use. Can be either "unary", "inverse_freq" or "inverse_freq_smooth".

    Returns
    -------
    sparse.csr_matrix
        tfidf normalized sparse matrix.

    Notes
    -----
    This function requires a lot of memory. Another option is to use the ac.pp.tfidf of the muon package.
    """

    # t - term (peak)
    # d - document (cell)
    # N - count of corpus (total set of cells)

    # Normalize matrix to number of found peaks
    dense = matrix.todense()
    peaks_per_cell = dense.sum(axis=1)  # i.e. the length of the document(number of words)

    # Decide on which Term frequency to use:
    if tf_type == "raw":
        tf = dense
    elif tf_type == "term_frequency":
        tf = dense / peaks_per_cell     # Counts normalized to peaks (words) per cell (document)
    elif tf_type == "log":
        tf = np.log1p(dense)            # for binary documents, this scales with "raw"

    # Decide on the Inverse document frequency to use
    N = dense.shape[0]     # number of cells (number of documents)
    df = dense.sum(axis=0)  # number of cells carrying each peak (number of documents containing each word)

    if idf_type == "unary":
        idf = np.ones(dense.shape[1])  # shape is number of peaks
    elif idf_type == "inverse_freq":
        idf = np.log(N / df)    # each cell has at least one peak (each document has one word), so df is always > 0
    elif idf_type == "inverse_freq_smooth":
        idf = np.log(N / (df + 1)) + 1

    # Obtain TF_IDF
    tf_idf = np.array(tf) * np.array(idf).squeeze()
    tf_idf = sparse.csr_matrix(tf_idf)

    return tf_idf


###################################################################################
# --------------------------- Batch correction methods -------------------------- #
###################################################################################


@beartype
def wrap_corrections(adata: sc.AnnData,
                     batch_key: str,
                     methods: Union[batch_methods,
                                    list[batch_methods],
                                    Callable] = ["bbknn", "mnn"],
                     method_kwargs: dict = {}) -> dict[str, sc.AnnData]:
    """
    Calculate multiple batch corrections for adata using the 'batch_correction' function.

    Parameters
    ----------
    adata : sc.AnnData
        An annotated data matrix object to apply corrections to.
    batch_key : str
        The column in adata.obs containing batch information.
    methods : list[batch_methods] | Callable | batch_methods, default ["bbknn", "mnn"]
        The method(s) to use for batch correction. Options are:
        - bbknn
        - mnn
        - harmony
        - scanorama
        - combat
        Or provide a custom batch correction function. See `batch_correction(method)` for more information.
    method_kwargs : dict, default {}
        Dict with methods as keys. Values are dicts of additional parameters forwarded to method. See batch_correction(**kwargs).

    Returns
    -------
    dict[str, sc.AnnData]
        Dictonary of batch corrected anndata objects. Where the key is the correction method and the value is the corrected anndata.

    Raises
    ------
    ValueError
        If not all methods in methods are valid.
    """

    # Ensure that methods can be looped over
    if isinstance(methods, str):
        methods = [methods]

    # check method_kwargs keys
    unknown_keys = set(method_kwargs.keys()) - set(methods)
    if unknown_keys:
        raise ValueError(f"Unknown methods in `method_kwargs` keys: {unknown_keys}")

    # Check the existance of packages before running batch_corrections
    required_packages = {"harmony": "harmonypy", "bbknn": "bbknn", "scanorama": "scanorama"}
    for method in methods:
        if method in required_packages:  # not all packages need external tools
            f = io.StringIO()
            with redirect_stderr(f):  # make the output of check_module silent; mnnpy prints ugly warnings
                utils.check_module(required_packages[method])

    # Collect batch correction per method
    anndata_dict = {'uncorrected': adata}
    for method in methods:
        anndata_dict[method] = batch_correction(adata, batch_key, method, **method_kwargs.setdefault(method, {}))  # batch correction returns the corrected adata

    logger.info("Finished batch correction(s)!")

    return anndata_dict


@deco.log_anndata
@beartype
def batch_correction(adata: sc.AnnData,
                     batch_key: str,
                     method: Union[batch_methods,
                                   list[batch_methods],
                                   Callable] = ["bbknn", "mnn"],
                     highly_variable: bool = True,
                     **kwargs: Any) -> sc.AnnData:
    """
    Perform batch correction on the adata object using the 'method' given.

    Parameters
    ----------
    adata : sc.AnnData
        An annotated data matrix object to apply corrections to.
    batch_key : str
        The column in adata.obs containing batch information.
    method : str or function
        Either one of the predefined methods or a custom function for batch correction.
        Note: The custom function is expected to accept an anndata object as the first parameter and return the batch corrected anndata.

        Available methods:
            - bbknn
            - mnn
            - harmony
            - scanorama
            - combat
    highly_variable : bool, default True
        Only for method 'mnn'. If True, only the highly variable genes (column 'highly_variable' in .var) will be used for batch correction.
    **kwargs :
        Additional arguments will be forwarded to the method function.

    Returns
    -------
    sc.AnnData
        A copy of the anndata with applied batch correction.

    Raises
    ------
    ValueError:
        1. If batch_key column is not in adata.obs
        2. If batch correction method is invalid.
    """

    if not callable(method):
        method = method.lower()

    logger.info(f"Running batch correction with '{method}'...")

    # Check that batch_key is in adata object
    if batch_key not in adata.obs.columns:
        raise ValueError(f"The given batch_key '{batch_key}' is not in adata.obs.columns")

    # Run batch correction depending on method
    if method == "bbknn":
        import bbknn  # sc.external.pp.bbknn() is broken due to n_trees / annoy_n_trees change

        # Get number of pcs in adata, as bbknn hardcodes n_pcs=50
        n_pcs = adata.obsm["X_pca"].shape[1]

        # Run bbknn
        adata = bbknn.bbknn(adata, batch_key=batch_key, n_pcs=n_pcs, copy=True, **kwargs)  # bbknn is an alternative to neighbors

    elif method == "mnn":

        var_table = adata.var  # var_table before batch correction

        # split adata on batch_key
        batch_categories = list(set(adata.obs[batch_key]))
        adatas = [adata[adata.obs[batch_key] == category] for category in batch_categories]

        # Set highly variable genes as var_subset if chosen (and available)
        var_subset = None
        if highly_variable:
            if "highly_variable" in adata.var.columns:
                var_subset = adata.var[adata.var.highly_variable].index

        # give individual adatas to mnn_correct
        corrected_adatas, _, _ = sce.pp.mnn_correct(adatas, batch_key=batch_key, var_subset=var_subset,
                                                    batch_categories=batch_categories, do_concatenate=False, **kwargs)

        # Join corrected adatas
        corrected_adatas = corrected_adatas[0]  # the output is a dict of list ([adata1, adata2, (...)], )
        adata = anndata.concat(corrected_adatas, join="outer", uns_merge="first")
        adata.var = var_table  # add var table back into corrected adata

        sc.pp.scale(adata)  # from the mnnpy github example
        sc.tl.pca(adata)  # rerun pca
        sc.pp.neighbors(adata)

    elif method == "harmony":
        adata = adata.copy()  # there is no copy option for harmony
        adata.obs[batch_key] = adata.obs[batch_key].astype("str")  # harmony expects a batch key as string

        sce.pp.harmony_integrate(adata, key=batch_key, **kwargs)
        adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]
        sc.pp.neighbors(adata)

    elif method == "scanorama":
        adata = adata.copy()  # there is no copy option for scanorama

        # scanorama expect the batch key in a sorted format
        # therefore anndata.obs should be sorted based on batch column before this method.
        original_order = adata.obs.index
        adata = adata[adata.obs[batch_key].argsort()]  # sort the whole adata to make sure obs is the same order as matrix

        sce.pp.scanorama_integrate(adata, key=batch_key, **kwargs)
        adata.obsm["X_pca"] = adata.obsm["X_scanorama"]
        sc.pp.neighbors(adata)

        # sort the adata back to the original order
        adata = adata[original_order]

    elif method == "combat":

        corrected_mat = sc.pp.combat(adata, key=batch_key, inplace=False, **kwargs)

        adata = adata.copy()  # make sure adata is not modified
        adata.X = sparse.csr_matrix(corrected_mat)

        sc.pp.pca(adata)
        sc.pp.neighbors(adata)

    elif callable(method):
        adata = method(adata.copy(), **kwargs)
    else:
        raise ValueError(f"Method '{method}' is not a valid batch correction method.")

    return adata  # the corrected adata object


@deco.log_anndata
@beartype
def evaluate_batch_effect(adata: sc.AnnData,
                          batch_key: str,
                          obsm_key: str = 'X_umap',
                          col_name: str = 'LISI_score',
                          max_dims: int = 5,
                          inplace: bool = False) -> Optional[sc.AnnData]:
    """
    Evaluate batch effect methods using LISI.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object with PCA and umap/tsne for batch evaluation.
    batch_key : str
        The column in adata.obs containing batch information.
    obsm_key : str, default 'X_umap'
        The column in adata.obsm containing coordinates.
    col_name : str, default 'LISI_score'
        Column name for storing the LISI score in .obs.
    max_dims : int, default 5
        Maximum number of dimensions of adata.obsm matrix to use for LISI (to speed up computation).
    inplace : bool, default False
        Whether to work inplace on the anndata object.

    Returns
    -------
    Optional[sc.AnnData]
        if inplace is True, LISI_score is added to adata.obs inplace (returns None), otherwise a copy of the adata is returned.

    Notes
    -----
    - LISI score is calculated for each cell and it is between 1-n for a data-frame with n categorical variables.
    - indicates the effective number of different categories represented in the local neighborhood of each cell.
    - If the cells are well-mixed, then we expect the LISI score to be near n for a data with n batches.
    - The higher the LISI score is, the better batch correction method worked to normalize the batch effect and mix the cells from different batches.
    - For further information on LISI: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1850-9

    Raises
    ------
    KeyError:
        1. If obsm_key is not in adata.obsm.
        2. If batch_key is no column in adata.obs.
    """

    # Load LISI
    utils.check_module("harmonypy")
    from harmonypy.lisi import compute_lisi

    # Handle inplace option
    adata_m = adata if inplace else adata.copy()

    # checks
    if obsm_key not in adata_m.obsm:
        raise KeyError(f"adata.obsm does not contain the obsm key: {obsm_key}")

    if batch_key not in adata_m.obs:
        raise KeyError(f"adata.obs does not contain the batch key: {batch_key}")

    # run LISI on all adata objects
    obsm_matrix = adata_m.obsm[obsm_key][:, :max_dims]
    lisi_res = compute_lisi(obsm_matrix, adata_m.obs, [batch_key])
    adata_m.obs[col_name] = lisi_res.flatten()

    if not inplace:
        return adata_m


@beartype
def wrap_batch_evaluation(adatas: dict[str, sc.AnnData],
                          batch_key: str,
                          obsm_keys: str | list[str] = ['X_pca', 'X_umap'],
                          threads: int = 1,
                          max_dims: int = 5,
                          inplace: bool = False) -> dict[str, sc.AnnData]:
    """
    Evaluate batch correction methods for a dict of anndata objects (using LISI score calculation).

    Parameters
    ----------
    adatas : dict[str, sc.AnnData]
        Dict containing an anndata object for each batch correction method as values. Keys are the name of the respective method.
        E.g.: {"bbknn": anndata}
    batch_key : str
        The column in adata.obs containing batch information.
    obsm_keys : str | list[str], default ['X_pca', 'X_umap']
        Key(s) to coordinates on which the score is calculated.
    threads : int, default 1
        Number of threads to use for parallelization.
    max_dims : int, default 5
        Maximum number of dimensions of adata.obsm matrix to use for LISI (to speed up computation).
    inplace : bool, default False
        Whether to work inplace on the anndata dict.

    Returns
    -------
    dict[str, sc.AnnData]
        Dict containing an anndata object for each batch correction method as values of LISI scores added to .obs.
    """

    if utils._is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    # Handle inplace option
    adatas_m = adatas if inplace else copy.deepcopy(adatas)

    # Ensure that obsm_key can be looped over
    if isinstance(obsm_keys, str):
        obsm_keys = [obsm_keys]

    # Evaluate batch effect for every adata
    if threads == 1:

        pbar = tqdm(total=len(adatas_m) * len(obsm_keys), desc="Calculation progress ")
        for adata in adatas_m.values():
            for obsm in obsm_keys:
                evaluate_batch_effect(adata, batch_key, col_name=f"LISI_score_{obsm}", obsm_key=obsm, max_dims=max_dims, inplace=True)
                pbar.update()
    else:
        utils.check_module("harmonypy")
        from harmonypy.lisi import compute_lisi

        pool = mp.Pool(threads)
        jobs = {}
        for i, adata in enumerate(adatas_m.values()):
            for obsm_key in obsm_keys:
                obsm_matrix = adata.obsm[obsm_key][:, :max_dims]
                obs_mat = adata.obs[[batch_key]]

                job = pool.apply_async(compute_lisi, args=(obsm_matrix, obs_mat, [batch_key]))
                jobs[(i, obsm_key)] = job
        pool.close()

        # Monitor all jobs with a pbar
        utils.monitor_jobs(jobs, "Calculating LISI scores")  # waits for all jobs to finish
        pool.join()

        # Assign results to adata
        for adata_i, obsm_key in jobs:
            adata = list(adatas_m.values())[adata_i]
            adata.obs[f"LISI_score_{obsm_key}"] = jobs[(adata_i, obsm_key)].get().flatten()

    if not inplace:
        return adatas_m
