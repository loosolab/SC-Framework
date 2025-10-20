"""Normalization and correction tools."""
import numpy as np
from scipy import sparse
import io
from contextlib import redirect_stderr
import copy
import multiprocessing as mp
import scanpy as sc
import scanpy.external as sce

from beartype.typing import Optional, Any, Union, Literal, Callable
from beartype import beartype

import sctoolbox.utils as utils
import sctoolbox.tools.dim_reduction as dim_red
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
                    method: Literal["total", "tfidf"] | list[Literal["total", "tfidf"]],
                    exclude_highly_expressed: bool = True,
                    use_highly_variable: bool = False,
                    target_sum: Optional[float] = None,
                    keep_layer: Optional[str] = "raw",
                    n_comps: int = 50) -> Union[dict[str, sc.AnnData], sc.AnnData]:
    """
    Normalize the count matrix and calculate dimension reduction using different methods.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    method : Literal["total", "tfidf"] | list[Literal["total", "tfidf"]]
        Normalization method. Either 'total' and/or 'tfidf'.
        - 'total': Performs normalization for total counts, log1p and PCA.
        - 'tfidf': Performs TFIDF normalization and LSI (corresponds to PCA). This method is often used for scATAC-seq data.
    exclude_highly_expressed : bool, default True
        Parameter for sc.pp.normalize_total. Decision to exclude highly expressed genes (HEG) from total normalization.
    use_highly_variable : bool, default False
        Parameter for sc.pp.pca and lsi. Decision to use highly variable genes for PCA/LSI.
    target_sum : Optional[float], default None
        Parameter for sc.pp.normalize_total. Decide the target sum of each cell after normalization.
    keep_layer : Optional[str], default "raw"
        Will create a copy of the .X matrix with the given name before applying normalization.
    n_comps : int, default 50
        The number of components to calculate.

    Returns
    -------
    Union[dict[str, sc.AnnData], sc.AnnData]
        Annotated data matrix with normalized count matrix and PCA/LSI calculated.
        If method is a list, a dictionary with the method as key and the corresponding anndata object as value is returned.
    """

    if isinstance(method, str):
        return normalize_and_dim_reduct(anndata=adata,
                                        method=method,
                                        exclude_highly_expressed=exclude_highly_expressed,
                                        use_highly_variable=use_highly_variable,
                                        target_sum=target_sum,
                                        keep_layer=keep_layer,
                                        n_comps=n_comps)

    elif isinstance(method, list):
        adatas = {}
        for method_str in method:  # method is a list

            adatas[method_str] = normalize_and_dim_reduct(anndata=adata,
                                                          method=method_str,
                                                          exclude_highly_expressed=exclude_highly_expressed,
                                                          use_highly_variable=use_highly_variable,
                                                          target_sum=target_sum,
                                                          keep_layer=keep_layer,
                                                          n_comps=n_comps)

        return adatas


@deco.log_anndata
@beartype
def normalize_and_dim_reduct(anndata: sc.AnnData,
                             method: Literal["total", "tfidf"],
                             exclude_highly_expressed: bool = True,
                             use_highly_variable: bool = False,
                             target_sum: Optional[float] = None,
                             inplace: bool = False,
                             keep_layer: Optional[str] = "raw",
                             n_comps: int = 50) -> Optional[sc.AnnData]:
    """
    Normalize the count matrix and calculate dimension reduction using different methods.

    Parameters
    ----------
    anndata : sc.AnnData
        Annotated data matrix.
    method : Literal["total", "tfidf"],
        The normalization method. Either 'total' or 'tfidf'.
    exclude_highly_expressed : bool, default True
        Parameter for sc.pp.normalize_total. Decision to exclude highly expressed genes (HEG) from total normalization.
    use_highly_variable : bool, default False
        Parameter for sc.pp.pca and lsi. Decision to use highly variable genes for PCA/LSI.
    target_sum : Optional[float], default None
        Parameter for sc.pp.normalize_total. Decide the target sum of each cell after normalization.
    inplace : bool, default False
        If True, change the anndata object inplace. Otherwise return changed anndata object.
    keep_layer : Optional[str], default "raw"
        Will create a copy of the .X matrix with the given name before applying normalization.
    n_comps : int, default 50
        The number of components to calculate.

    Returns
    -------
    Optional[sc.AnnData]
        Annotated data matrix with normalized count matrix and PCA/LSI calculated.
    """

    adata = anndata if inplace else anndata.copy()

    if keep_layer:
        if keep_layer in adata.layers:
            logger.warning(f"A layer with the name '{keep_layer}' already exists. Skipping to avoid layer overwrite.")
        else:
            adata.layers[keep_layer] = adata.X.copy()

    if method == "total":  # perform total normalization and pca
        logger.info('Performing total normalization and PCA...')
        sc.pp.normalize_total(adata, exclude_highly_expressed=exclude_highly_expressed, target_sum=target_sum)
        sc.pp.log1p(adata)
        sc.pp.pca(adata, mask_var="highly_variable" if use_highly_variable else None, n_comps=n_comps)

    elif method == "tfidf":
        logger.info('Performing TFIDF and LSI...')
        tfidf(adata, inplace=True)
        dim_red.lsi(adata, use_highly_variable=use_highly_variable, n_comps=n_comps)  # corresponds to PCA

    if not inplace:
        return adata


@beartype
def tfidf(anndata: sc.AnnData,
          log_tf: bool = True,
          log_idf: bool = True,
          log_tfidf: bool = False,
          scale_factor: int = int(1e4),
          inplace: bool = False,
          layer: Optional[str] = None) -> Optional[sc.AnnData]:
    """
    Transform peak counts with TF-IDF (Term Frequency - Inverse Document Frequency).

    TF: peak counts are normalised by total number of counts per cell.
    DF: total number of counts for each peak.
    IDF: number of cells divided by DF.
    By default, log(TF) * log(IDF) is returned.

    Parameters
    ----------
    anndata : sc.AnnData
        AnnData object with peak counts.
    log_tf : bool, default True
        Log-transform TF term if True.
    log_idf : bool, default True
        Log-transform IDF term if True.
    log_tfidf : bool, default Frue
        Log-transform TF*IDF term if True. Can only be used when log_tf and log_idf are False.
    scale_factor : int, default 1e4
        Scale factor to multiply the TF-IDF matrix by.
    inplace : bool, default False
        If True, change the anndata object inplace. Otherwise return changed anndata object.
    layer : Optional[str], default None
        Perform tfidf on given layer. If None tfidf is run on adata.X.

    Notes
    -----
    Function is from the muon package.
    This function overwrites the .X matrix.

    Raises
    ------
    AttributeError
        log(TF*IDF) requires log(TF) and log(IDF) to be False.

    Returns
    -------
    Optional[sc.AnnData]
        TF-IDF normalized anndata object.
    """

    adata = anndata if inplace else anndata.copy()
    matrix = adata.layers[layer] if layer else adata.X

    if log_tfidf and (log_tf or log_idf):
        raise AttributeError(
            "When returning log(TF*IDF), \
            applying neither log(TF) nor log(IDF) is possible."
        )

    if sparse.issparse(matrix):
        n_peaks = np.asarray(matrix.sum(axis=1)).reshape(-1)
        n_peaks = sparse.dia_matrix((1.0 / n_peaks, 0), shape=(n_peaks.size, n_peaks.size))
        # This prevents making TF dense
        tf = np.dot(n_peaks, matrix)
    else:
        n_peaks = np.asarray(matrix.sum(axis=1)).reshape(-1, 1)
        tf = matrix / n_peaks
    if scale_factor is not None and scale_factor != 0 and scale_factor != 1:
        tf = tf * scale_factor
    if log_tf:
        tf = np.log1p(tf)

    idf = np.asarray(adata.shape[0] / matrix.sum(axis=0)).reshape(-1)
    if log_idf:
        idf = np.log1p(idf)

    if sparse.issparse(tf):
        idf = sparse.dia_matrix((idf, 0), shape=(idf.size, idf.size))
        tf_idf = np.dot(tf, idf)
    else:
        tf_idf = np.dot(sparse.csr_matrix(tf), sparse.csr_matrix(np.diag(idf)))

    if log_tfidf:
        tf_idf = np.log1p(tf_idf)

    if layer:
        adata.layers[layer] = np.nan_to_num(tf_idf, nan=0)
    else:
        adata.X = np.nan_to_num(tf_idf, nan=0)

    if not inplace:
        return adata


###################################################################################
# --------------------------- Batch correction methods -------------------------- #
###################################################################################


@beartype
def wrap_corrections(adata: sc.AnnData,
                     batch_key: str,
                     methods: Union[batch_methods,
                                    list[batch_methods],
                                    Callable] = ["bbknn", "mnn"],
                     method_kwargs: dict = {},
                     keep_layer: Optional[str] = "norm") -> dict[str, sc.AnnData]:
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
    keep_layer : Optional[str], default "norm"
        Will create a copy of the .X matrix with the given name before applying correction.

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
                utils.checker.check_module(required_packages[method])

    # keep .X as layer; will propagate through the batch corrections
    if keep_layer:
        if keep_layer in adata.layers:
            logger.warning(f"A layer with the name '{keep_layer}' already exists. Skipping to avoid layer overwrite.")
        else:
            adata.layers[keep_layer] = adata.X.copy()

    # Collect batch correction per method
    anndata_dict = {'uncorrected': adata.copy()}
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
                     dim_red_kwargs: dict = {},
                     **kwargs: Any) -> sc.AnnData:
    """
    Perform batch correction on the adata object using the 'method' given.

    Different correction methods will perform the batch correction on different aspects of the data,
    meaning calculations following the correction have to be redone. Here is an overview on the analysis
    steps until batch correction and where each of the methods is applied:

    Matrix (adata.X) -> Dimension reduction (e.g. PCA) -> Nearest neighbor

    +-----------+----------------------+
    | Method    | Applied to/ replaces |
    +===========+======================+
    | bbknn     | Nearest neighbor     |
    +-----------+----------------------+
    | mnn       | Matrix               |
    +-----------+----------------------+
    | harmony   | Dimension reduction  |
    +-----------+----------------------+
    | scanorama | Dimension reduction  |
    +-----------+----------------------+
    | combat    | Matrix               |
    +-----------+----------------------+

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
    dim_red_kwargs : dict, default {}
        Arguments to redo the steps following the selected batch correction (see table above). Forwarded to :func:`sctoolbox.tools.dim_reduction.dim_red`.
    **kwargs : Any
        Additional arguments will be forwarded to the method function.
        The following parameters are set unless specified to avoid potential issues with the annoy package and processor architecture:
        - bbknn: `computation="cKDTree"`
        - scanorama: `approx=False`
        See here for further information https://github.com/Teichlab/bbknn/issues/60, https://github.com/brianhie/scanorama?tab=readme-ov-file#troubleshooting

    Returns
    -------
    sc.AnnData
        A copy of the anndata with applied batch correction.

    Raises
    ------
    ValueError
        1. If batch_key column is not in adata.obs
        2. If batch correction method is invalid.
    KeyError
        If PCA has not been calculated before running bbknn.
    """

    if not callable(method):
        method = method.lower()

    logger.info(f"Running batch correction with '{method}'...")

    # Check that batch_key is in adata object
    if batch_key not in adata.obs.columns:
        raise ValueError(f"The given batch_key '{batch_key}' is not in adata.obs.columns")

    # ensure no side effects
    adata = adata.copy()

    # Run batch correction depending on method
    if method == "bbknn":
        import bbknn  # sc.external.pp.bbknn() is broken due to n_trees / annoy_n_trees change

        # Get number of pcs in adata, as bbknn hardcodes n_pcs=50
        try:
            n_pcs = adata.obsm["X_pca"].shape[1]
        except KeyError:
            raise KeyError("PCA has not been calculated. Please run sc.pp.pca() before running bbknn.")

        # to avoid annoy issues
        if "computation" not in kwargs:
            kwargs["computation"] = "cKDTree"

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
        adata = sc.concat(corrected_adatas, join="outer", uns_merge="first")
        adata.var = var_table  # add var table back into corrected adata

        sc.pp.scale(adata)  # from the mnnpy github example

        # dimension reduction and neighbor graph
        # mnn dim_red default
        mnn_dim_def = {
            "method": "PCA",  # TODO allow lsi
            "method_kwargs": {"mask_var": "highly_variable" if highly_variable else None}
        }
        mnn_dim_def.update(dim_red_kwargs)

        dim_red.dim_red(anndata=adata, inplace=True, **mnn_dim_def)

    elif method == "harmony":
        adata.obs[batch_key] = adata.obs[batch_key].astype("str")  # harmony expects a batch key as string

        sce.pp.harmony_integrate(adata, key=batch_key, **kwargs)
        adata.obsm["X_pca"] = adata.obsm.pop("X_pca_harmony")

        # don't redo dimension reduction but apply subset
        dim_red.dim_red(anndata=adata, inplace=True, method=None, **dim_red_kwargs)

    elif method == "scanorama":
        # scanorama expect the batch key in a sorted format
        # therefore anndata.obs should be sorted based on batch column before this method.
        original_order = adata.obs.index
        adata = adata[adata.obs[batch_key].argsort()]  # sort the whole adata to make sure obs is the same order as matrix

        # to avoid annoy issues
        if "approx" not in kwargs:
            kwargs["approx"] = False

        sce.pp.scanorama_integrate(adata, key=batch_key, **kwargs)  # TODO
        adata.obsm["X_pca"] = adata.obsm.pop("X_scanorama")

        # don't redo dimension reduction but apply subset
        dim_red.dim_red(anndata=adata, inplace=True, method=None, **dim_red_kwargs)

        # sort the adata back to the original order
        adata = adata[original_order]

    elif method == "combat":

        adata = adata.copy()  # make sure adata is not modified
        # run combat
        sc.pp.combat(adata, key=batch_key, inplace=True, **kwargs)

        sc.pp.pca(adata)  # TODO
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
                          perplexity: int = 30,
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
    perplexity : int, default 30
        Perplexity for the LISI score calculation.
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
    KeyError
        1. If obsm_key is not in adata.obsm.
        2. If batch_key is no column in adata.obs.
    """

    # Load LISI
    utils.checker.check_module("harmonypy")
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
    lisi_res = compute_lisi(obsm_matrix, adata_m.obs, [batch_key], perplexity=perplexity)
    adata_m.obs[col_name] = lisi_res.flatten()

    if not inplace:
        return adata_m


@beartype
def wrap_batch_evaluation(adatas: dict[str, sc.AnnData],
                          batch_key: str,
                          obsm_keys: str | list[str] = ['X_pca', 'X_umap'],
                          threads: Optional[int] = 1,
                          max_dims: int = 5,
                          inplace: bool = False) -> Optional[dict[str, sc.AnnData]]:
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
    threads : Optional[int], default 1
        Number of threads to use for parallelization. Set None to use settings.get_threads().
    max_dims : int, default 5
        Maximum number of dimensions of adata.obsm matrix to use for LISI (to speed up computation).
    inplace : bool, default False
        Whether to work inplace on the anndata dict.

    Returns
    -------
    Optional[dict[str, sc.AnnData]]
        Dict containing an anndata object for each batch correction method as values of LISI scores added to .obs.
    """

    if threads is None:
        threads = settings.get_threads()

    if utils.jupyter._is_notebook() is True:
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
            n_cells = adata.shape[0]
            perplexity = min(30, int(n_cells / 3))  # adjust perplexity for small datasets

            for obsm in obsm_keys:
                evaluate_batch_effect(adata, batch_key, col_name=f"LISI_score_{obsm}", obsm_key=obsm, max_dims=max_dims, perplexity=perplexity, inplace=True)
                pbar.update()
    else:
        utils.checker.check_module("harmonypy")
        from harmonypy.lisi import compute_lisi

        pool = mp.Pool(threads)
        jobs = {}
        for i, adata in enumerate(adatas_m.values()):
            n_cells = adata.shape[0]
            perplexity = min(30, int(n_cells / 3))  # adjust perplexity for small datasets

            for obsm_key in obsm_keys:
                obsm_matrix = adata.obsm[obsm_key][:, :max_dims]
                obs_mat = adata.obs[[batch_key]]

                job = pool.apply_async(compute_lisi, args=(obsm_matrix, obs_mat, [batch_key], perplexity,))
                jobs[(i, obsm_key)] = job
        pool.close()

        # Monitor all jobs with a pbar
        utils.multiprocessing.monitor_jobs(jobs, "Calculating LISI scores")  # waits for all jobs to finish
        pool.join()

        # Assign results to adata
        for adata_i, obsm_key in jobs:
            adata = list(adatas_m.values())[adata_i]
            adata.obs[f"LISI_score_{obsm_key}"] = jobs[(adata_i, obsm_key)].get().flatten()

    if not inplace:
        return adatas_m
