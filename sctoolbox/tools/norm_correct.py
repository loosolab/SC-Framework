"""
Normalization and correction tools
"""

import numpy as np
from scipy import sparse
import io
from contextlib import redirect_stderr
import copy
import multiprocessing as mp
import anndata
import scanpy as sc
import scanpy.external as sce

import sctoolbox.utils as utils
from sctoolbox.tools.dim_reduction import lsi, compute_PCA
from sctoolbox.tools import highly_variable as hv


#####################################################################
# --------------------- Normalization methods --------------------- #
#####################################################################

def atac_norm(adata, methods):
    """
    A function that normalizes count matrix using different methods.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    method : str or list of str
        Normalization method. Either 'total' or 'tfidf'.

        - 'total': Performs normalization for total counts, log1p and PCA.
        - 'tfidf': Performs TFIDF normalization and LSI (corresponds to PCA). This method is often used for scATAC-seq data.

    Returns
    -------
    dict of anndata.AnnData
        Dictionary containing method name as key, and anndata as values.
        Each anndata is the annotated data matrix with normalized count matrix and PCA/LSI calculated.
    """

    if isinstance(methods, str):
        methods = [methods]

    adatas = {}
    for method in methods:
        adata = adata.copy()  # make sure the original data is not modified

        if method == "total":  # perform total normalization and pca
            print('Performing total normalization and PCA...')
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            sc.pp.pca(adata)

        elif method == "tfidf":
            print('Performing TFIDF and LSI...')
            tfidf(adata)
            lsi(adata)  # corresponds to PCA

        else:
            raise ValueError(f"Method '{method}' is invalid - must be either 'total' or 'tfidf'.")

        adatas[method] = adata

    return adatas


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


def adata_normalize_total(anndata, excl=True, inplace=False, norm_kwargs={}, log_kwargs={}):
    """
    Normalizing the total counts and converting to log

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object to normalize.
    excl : boolean, default True
        Decision to exclude highly expressed genes (HEG) from normalization.
    inplace : boolean, default False
        Whether the anndata object is modified inplace.
    norm_kwargs : dict, default {}
        Additional parameters forwarded to scanpy.pp.normalize_total().
    log_kwargs : dict, default {}
        Additional parameters forwarded to scanpy.pp.log1p().

    Returns
    -------
    anndata.AnnData or None:
        Returns normalized and logged anndata object. Or None if inplace = True.
    """
    adata_m = anndata if inplace else anndata.copy()

    # Normalizing and logaritimyzing
    print("Normalizing the data and converting to log")
    sc.pp.normalize_total(adata_m, exclude_highly_expressed=excl, inplace=True, **norm_kwargs)
    sc.pp.log1p(adata_m, copy=False, **log_kwargs)

    # Adding info in anndata.uns["infoprocess"]
    # cr.build_infor(adata_m, "Scanpy normalization", "exclude_highly_expressed= " + str(excl), inplace=True)

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
    adata_normalize_total(adata_m, exclude_HEG, inplace=True)

    # Annotate highly variable genes
    hv.annot_HVG(adata_m, inplace=True)

    # Compute PCA
    compute_PCA(adata_m, use_highly_variable=use_HVG_PCA, inplace=True)

    if not inplace:
        return adata_m


def tfidf(data, log_tf=True, log_idf=True, log_tfidf=False, scale_factor=1e4):
    """Transform peak counts with TF-IDF (Term Frequency - Inverse Document Frequency).
    TF: peak counts are normalised by total number of counts per cell.
    DF: total number of counts for each peak.
    IDF: number of cells divided by DF.
    By default, log(TF) * log(IDF) is returned.

    Note: Function is from the muon package.

    :param anndata.AnnData data: AnnData object with peak counts.
    :param bool log_tf: Log-transform TF term, defaults to True.
    :param bool log_idf: Log-transform IDF term, defaults to True.
    :param bool log_tfidf: Log-transform TF*IDF term. Can only be used when log_tf and log_idf are False, defaults to False.
    :param int scale_factor: Scale factor to multiply the TF-IDF matrix by, defaults to 1e4.
    :raises TypeError: data must be anndata object.
    :raises AttributeError: log(TF*IDF) requires log(TF) and log(IDF) to be False.
    """
    if isinstance(data, anndata.AnnData):
        adata = data
    else:
        raise TypeError("Expected AnnData object!")

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


def tfidf_normalization(matrix, tf_type="term_frequency", idf_type="inverse_freq"):
    """ Perform TF-IDF normalization on a sparse matrix.
    The different variants of the term frequency and inverse document frequency are obtained from https://en.wikipedia.org/wiki/Tf-idf.

    Note: this function requires a lot of memory. Another option is to use the ac.pp.tfidf of the muon package.

    Parameters
    -----------
    matrix : scipy.sparse matrix
        The matrix to be normalized.
    tf_type : string, optional
        The type of term frequency to use. Can be either "raw", "term_frequency" or "log". Default: "term_frequency".
    idf_type : string, optional
        The type of inverse document frequency to use. Can be either "unary", "inverse_freq" or "inverse_freq_smooth". Default: "inverse_freq".
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

def wrap_corrections(adata,
                     batch_key,
                     methods=["bbknn", "mnn"],
                     method_kwargs={}):
    """
    Wrapper for calculating multiple batch corrections for adata using the 'batch_correction' function.

    Parameters
    ----------
    adata : anndata.AnnData
        An annotated data matrix object to apply corrections to.
    batch_key : str
        The column in adata.obs containing batch information.
    methods : list of str or function
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
    dict of anndata.Anndata :
        Dictonary of batch corrected anndata objects. Where the key is the correction method and the value is the corrected anndata.
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

    print("Finished batch correction(s)!")

    return anndata_dict


def batch_correction(adata, batch_key, method, highly_variable=True, **kwargs):
    """
    Perform batch correction on the adata object using the 'method' given.

    Parameters
    ----------
    adata : anndata.AnnData
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
    anndata.AnnData :
        A copy of the anndata with applied batch correction.
    """
    if not callable(method):
        method = method.lower()

    print(f"Running batch correction with '{method}'...")

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

        # Make sure that the batch_key is still a categorical
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")

        sc.pp.scale(adata)  # from the mnnpy github example
        sc.tl.pca(adata)  # rerun pca
        sc.pp.neighbors(adata)

    elif method == "harmony":
        adata = adata.copy()  # there is no copy option for harmony

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

    # Add information to adata.uns
    utils.add_uns_info(adata, "batch_correction", {"method": method, "batch_key": batch_key})

    return adata  # the corrected adata object


def evaluate_batch_effect(adata, batch_key, obsm_key='X_umap', col_name='LISI_score', max_dims=5, inplace=False):
    """
    Evaluate batch effect methods using LISI.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object with PCA and umap/tsne for batch evaluation.
    batch_key : str
        The column in adata.obs containing batch information.
    obsm_key : str, default 'X_umap'
        The column in adata.obsm containing coordinates.
    col_name : str
        Column name for storing the LISI score in .obs.
    max_dims : int, default 5
        Maximum number of dimensions of adata.obsm matrix to use for LISI (to speed up computation).
    inplace : boolean, default False
        Whether to work inplace on the anndata object.

    Returns
    -------
    anndata.Anndata or None:
        if inplace is True, LISI_score is added to adata.obs inplace (returns None), otherwise a copy of the adata is returned.

    NOTES
    -------
    - LISI score is calculated for each cell and it is between 1-n for a data-frame with n categorical variables.
    - indicates the effective number of different categories represented in the local neighborhood of each cell.
    - If the cells are well-mixed, then we expect the LISI score to be near n for a data with n batches.
    - The higher the LISI score is, the better batch correction method worked to normalize the batch effect and mix the cells from different batches.
    - For further information on LISI: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1850-9
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


def wrap_batch_evaluation(adatas, batch_key, obsm_keys=['X_pca', 'X_umap'], threads=1, max_dims=5, inplace=False):
    """
    Evaluating batch correction methods for a dict of anndata objects (using LISI score calculation)

    Parameters
    ----------
    adatas : dict of anndata.AnnData
        Dict containing an anndata object for each batch correction method as values. Keys are the name of the respective method.
        E.g.: {"bbknn": anndata}
    batch_key : str
        The column in adata.obs containing batch information.
    obsm_keys : str or list of str, default ['X_pca', 'X_umap']
        Key(s) to coordinates on which the score is calculated.
    max_dims : int, default 5
        Maximum number of dimensions of adata.obsm matrix to use for LISI (to speed up computation).
    threads : int
        Number of threads to use for parallelization.
    inplace : boolean, default False
        Whether to work inplace on the anndata dict.

    Returns
    -------
    dict of anndata.AnnData
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
