# Loading packages
import scanpy as sc
import scanpy.external as sce
from fitter import Fitter
import numpy as np
from kneed import KneeLocator
import scipy.stats
from scipy import sparse
from contextlib import redirect_stderr
import io
import copy
import multiprocessing as mp
import time
import matplotlib.pyplot as plt

import anndata
import sctoolbox.creators as cr
import sctoolbox.annotation as an
import sctoolbox.utilities as utils


def rename_categories(series):
    """
    Rename categories in a pandas series to numbers between 1-(number of categories).

    Parameters
    ----------
    series : pandas.Series
        Series to rename categories in.

    Returns
    -------
    pandas.Series
        Series with renamed categories.
    """

    n_categories = series.cat.categories
    new_names = [str(i) for i in range(1, len(n_categories) + 1)]
    translate_dict = dict(zip(series.cat.categories.tolist(), new_names))
    series = series.cat.rename_categories(translate_dict)

    return series


def recluster(adata, column, clusters,
              task="join", method="leiden", resolution=1, key_added=None, plot=True):
    """
    Recluster an anndata object based on an existing clustering column in .obs.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    column : str
        Column in adata.obs to use for re-clustering.
    clusters : str or list of str
        Clusters in `column` to re-cluster.
    task : str, default "join"
        Task to perform. Options are:
        - "join": Join clusters in `clusters` into one cluster.
        - "split": Split clusters in `clusters` are merged and then reclustered using `method` and `resolution`.
    method : str, default "leiden"
        Clustering method to use. Must be one of "leiden" or "louvain".
    resolution : float, default 1
        Resolution parameter for clustering.
    key_added : str, default None
        Name of the new column in adata.obs. If None, the column name is set to `<column>_recluster`.
    plot : bool, default True
        If a plot should be generated of the re-clustering.
    """

    adata_copy = adata.copy()

    # --- Get ready --- #
    # check if column is in adata.obs
    if column not in adata.obs.columns:
        raise ValueError(f"Column {column} not found in adata.obs")

    # Decide key_added
    if key_added is None:
        key_added = f"{column}_recluster"

    # Check that method is valid
    if method == "leiden":
        cl_function = sc.tl.leiden
    elif method == "louvain":
        cl_function = sc.tl.louvain
    else:
        raise ValueError(f"Method '{method} is not valid. Method must be one of: leiden, louvain")

    # --- Start reclustering --- #
    if task == "join":
        translate = {cluster: clusters[0] for cluster in clusters}
        adata.obs[key_added] = adata.obs[column].replace(translate)

    elif task == "split":
        cl_function(adata, restrict_to=(column, clusters), resolution=resolution, key_added=key_added)

    else:
        raise ValueError(f"Task '{task}' is not valid. Task must be one of: 'join', 'split'")

    adata.obs[key_added] = rename_categories(adata.obs[key_added])  # rename to start at 1

    # --- Plot reclustering before/after --- #
    if plot is True:

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        sc.pl.umap(adata_copy, color=column, ax=ax[0], show=False, legend_loc="on data")
        ax[0].set_title(f"Before re-clustering\n(column name: '{column}')")

        sc.pl.umap(adata, color=key_added, ax=ax[1], show=False, legend_loc="on data")
        ax[1].set_title(f"After re-clustering\n (column name: '{key_added}')")


# --------------------------- Batch correction methods -------------------------- #

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
    required_packages = {"harmony": "harmonypy", "bbknn": "bbknn", "mnn": "mnnpy", "scanorama": "scanorama"}
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
        adata = sce.pp.bbknn(adata, batch_key=batch_key, copy=True, **kwargs)  # bbknn is an alternative to neighbors

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

        sce.pp.harmony_integrate(adata, key=batch_key, **kwargs)
        adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]
        sc.pp.neighbors(adata)

    elif method == "scanorama":
        adata = adata.copy()  # there is no copy option for scanorama

        # scanorama expect the batch key in a sorted format
        # therefore anndata.obs should be sorted based on batch column before this method.
        adata = adata[adata.obs[batch_key].argsort()]  # sort the whole adata to make sure obs is the same order as matrix

        sce.pp.scanorama_integrate(adata, key=batch_key, **kwargs)
        adata.obsm["X_pca"] = adata.obsm["X_scanorama"]
        sc.pp.neighbors(adata)

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


# --------------------------- Automatic thresholds ------------------------- #

def get_threshold(data, interval, limit_on="both"):
    """
    Define cutoffs for the given data. Depending on distribution shape either by knee detection or using a percentile.

    Parameters
    ----------
    data : numbers list
        Data the thresholds will be calculated on.
    interval : int
        A percentage value from 0 to 100 to be used to calculate the confidence interval or percentile.
    limit_on : str, default 'both'
        Define a threshold for lower, upper or both sides of the distribution. Options: ['lower', 'upper', 'both']

    Returns
    -------
    number or number tuple:
        Depending on 'limit_on' will give a single value for the lower or upper threshold or a tuple for both (lower, upper).
        Lower threshold is defined as `interval`, upper as `100 - interval`.
    """
    # -------------------- setup -------------------- #
    # fall back to percentile thresholds if true
    fallback = False
    # check limit_on value
    if limit_on not in ["upper", "lower", "both"]:
        raise ValueError(f"Parameter limit_on has to be one of {['upper', 'lower', 'both']}. Got {limit_on}.")

    # compute skew and kurtosis
    # skew and kurtosis are used to identify distribution shape.
    # TODO why is skew rounded?
    # TODO why is kurtosis casted to int?
    # The skew value describes the symmetry of a distribution aka whether it's shifted to left or right.
    # >0 = right skew; longer tail on the right of its peak
    #  0 = no skew; symmetrical, equal tails on both sides of the peak
    # <0 = left skew; longer tail on the left of its peak
    # See https://www.scribbr.com/statistics/skewness/
    skew = round(scipy.stats.skew(data, bias=False), 1)
    # The kurtosis value describes the taildness aka outlier frequency of a distribution.
    # >3 = frequent outliers; leptokurtic
    #  3 = moderate outliers; mesokurtic
    # <3 = infrequent outliers; platykurtic
    # See https://www.scribbr.com/statistics/kurtosis/
    kurtosis = int(scipy.stats.kurtosis(data, fisher=False, bias=False))

    # -------------------- find thresholds -------------------- #
    # This is a skewed distribution, and platykurtic (not extremely sharp, not so long but onesided tail)
    # or
    # This is a leptokurtic distribution (extremely sharp, long tail)
    # TODO The comment does not seem to match the if condition.
    if skew != 0 and kurtosis != 3:
        # find out kind of distribution through fitting
        f = Fitter(data, distributions=["uniform", "expon", "powerlaw", "norm"])
        f.fit()
        # name of best fit
        best_fit = list(f.get_best().keys())[0]

        # This is the power law or exponential distributed data
        if best_fit == "expon" or best_fit == "powerlaw":
            # set knee as thresholds
            data.sort()

            # TODO why do knee location on binned data instead of raw?
            # reduce data to histon bins
            hist, bin_edges = np.histogram(a=data, bins=int(len(data) / 100), weights=range(0, len(data), 1))

            # TODO why have a list of knees?
            thresholds = []
            for curve in ["convex", "concave"]:
                # TODO why do increasing and decreasing?
                for direction in ["increasing", "decreasing"]:
                    # find knee
                    knee_obj = KneeLocator(x=range(1, len(hist) + 1), y=hist, curve=curve, direction=direction)

                    # convert knee (histon number) to bin edge (actual data value)
                    threshold = bin_edges[knee_obj.knee - 1]

                    # TODO why is this done?
                    # remove 0 if in thresholds
                    if threshold > 0:
                        thresholds.append(threshold)

            # Why only keep lowest value?
            thresholds = [min(thresholds), None]
        else:
            fallback = True

    # This is a normal distribution
    # or
    # This is a mesokurtic skewed distributed (long tail and not sharp)
    # or
    # This is the skewed shaped but not like exponential nor powerlaw
    # TODO The comment does not seem to match the if condition.
    if skew == 0 or skew != 0 and kurtosis == 3 or fallback:
        thresholds = [np.percentile(data, interval), np.percentile(data, 100 - interval)]

        # TODO why is this done?
        # remove 0 if in thresholds
        thresholds = [None if t == 0 else t for t in thresholds]

    # return thresholds
    if limit_on == "both":
        return tuple(thresholds)
    elif limit_on == "lower":
        return thresholds[0]
    else:
        return thresholds[1]


def calculate_qc_metrics(anndata, percent_top=None, inplace=False, **kwargs):
    """
    Calculating the qc metrics using `scanpy.pp.calculate_qc_metrics`

    TODO add logging
    TODO we may want to rethink if this function is necessary

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object the quality metrics are added to.
    percent_top : [int], default None
        Which proportions of top genes to cover. For more information see `scanpy.pp.calculate_qc_metrics(percent_top)`.
    inplace : bool, default False
        If the anndata object should be modified in place.
    ** kwargs :
        Additional parameters forwarded to scanpy.pp.calculate_qc_metrics. See https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.calculate_qc_metrics.html.

    Returns
    -------
    anndata.AnnData or None:
        Returns anndata object with added quality metrics to .obs and .var. Returns None if `inplace=True`.
    """
    # add metrics to copy of anndata
    if not inplace:
        anndata = anndata.copy()

    # compute metrics
    sc.pp.calculate_qc_metrics(adata=anndata, percent_top=percent_top, inplace=True, **kwargs)

    # return modified anndata
    if not inplace:
        return anndata


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
    cr.build_infor(adata_m, "Scanpy computed PCA", "use_highly_variable= " + str(use_highly_variable), inplace=True)

    if not inplace:
        return adata_m


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

    Notes
    -----
    Author: Guilherme Valente

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
    cr.build_infor(adata_m, "Scanpy normalization", "exclude_highly_expressed= " + str(excl), inplace=True)

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

    Notes
    -----
    Author: Guilherme Valente
    """
    adata_m = anndata if inplace else anndata.copy()

    # Normalization and converting to log
    adata_normalize_total(adata_m, exclude_HEG, inplace=True)

    # Annotate highly variable genes
    an.annot_HVG(adata_m, inplace=True)

    # Compute PCA
    compute_PCA(adata_m, use_highly_variable=use_HVG_PCA, inplace=True)

    if not inplace:
        return adata_m


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
    cr.build_infor(anndata, "PCA_knee_threshold", knee)

    return knee


def subset_PCA(adata, n_pcs, start=0, inplace=True):
    """
    Subset the PCA coordinates in adata.obsm["X_pca"] to the given number of pcs.

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

    if inplace is False:
        return adata


def evaluate_batch_effect(adata, batch_key, obsm_key='X_umap', col_name='LISI_score', inplace=False):
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

    lisi_res = compute_lisi(adata_m.obsm[obsm_key], adata_m.obs, [batch_key])
    adata_m.obs[col_name] = lisi_res.flatten()

    if not inplace:
        return adata_m


def wrap_batch_evaluation(adatas, batch_key, obsm_keys=['X_pca', 'X_umap'], threads=1, inplace=False):
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
                evaluate_batch_effect(adata, batch_key, col_name=f"LISI_score_{obsm}", obsm_key=obsm, inplace=True)
                pbar.update()
    else:
        utils.check_module("harmonypy")
        from harmonypy.lisi import compute_lisi

        pool = mp.Pool(threads)
        jobs = {}
        for i, adata in enumerate(adatas_m.values()):
            for obsm_key in obsm_keys:
                obsm_matrix = adata.obsm[obsm_key]
                obs_mat = adata.obs[[batch_key]]

                job = pool.apply_async(compute_lisi, args=(obsm_matrix, obs_mat, [batch_key]))  # callback=lambda x: adata.obs[f"LISI_score_{obsm_key}"] = x.flatten())
                jobs[(i, obsm_key)] = job
        pool.close()

        # Wait for all jobs to finish
        n_ready = sum([job.ready() for job in jobs.values()])
        pbar = tqdm(total=len(jobs), desc="Calculation progress ")
        while n_ready < len(jobs):
            n_ready = sum([job.ready() for job in jobs.values()])
            if pbar.n != n_ready:
                pbar.n = n_ready
                pbar.refresh()
            time.sleep(1)
        pbar.close()
        pool.join()

        # Assign results to adata
        for adata_i, obsm_key in jobs:
            adata = list(adatas_m.values())[adata_i]
            adata.obs[f"LISI_score_{obsm_key}"] = jobs[(adata_i, obsm_key)].get().flatten()

    if not inplace:
        return adatas_m
