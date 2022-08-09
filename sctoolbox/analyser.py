# Loading packages
import scanpy as sc
import scanpy.external as sce
from fitter import Fitter
import numpy as np
from kneed import KneeLocator
from scipy import sparse
from contextlib import redirect_stderr
import io
import pandas as pd

import anndata
import sctoolbox.creators as cr
import sctoolbox.annotation as an
import sctoolbox.utilities as utils
import sctoolbox.plotting as plot

# --------------------------- Batch correction methods -------------------------- #

def wrap_corrections(adata,
                     batch_key,
                     methods=["bbknn", "mnn"]):
    """
    Wrapper for calculating multiple batch corrections for adata using the 'batch_correction' function.

    Parameters
    ----------
    adata : anndata.AnnData
        An annotated data matrix object to apply corrections to.
    batch_key : str
        The column in adata.obs containing batch information.
    methods : list of str
        The method(s) to use for batch correction. Options are:
        - bbknn
        - mnn
        - harmony
        - scanorama
        - combat
    """

    # Ensure that methods can be looped over
    if isinstance(methods, str):
        methods = [methods]

    # Check the existance of packages before running batch_corrections
    required_packages = {"harmony": "harmonypy", "bbknn": "bbknn", "mnn": "mnnpy", "scanorama": "scanorama"}
    for method in methods:
        if method in required_packages:  # not all packages need external tools
            f = io.StringIO()
            with redirect_stderr(f):  # make the output of check_module silent; mnnpy prints ugly warnings
                utils.check_module(required_packages[method])

    # Collect batch correction per method
    anndata_dict = {}
    for method in methods:
        anndata_dict[method] = batch_correction(adata, batch_key, method)  # batch correction returns the corrected adata

    anndata_dict['uncorrected'] = adata

    print("Finished batch correction(s)!")

    return anndata_dict


def batch_correction(adata, batch_key, method, highly_variable=True):
    """
    Perform batch correction on the adata object using the 'method' given.

    Parameters
    -----------
    adata : anndata.AnnData
        An annotated data matrix object to apply corrections to.
    batch_key : str
        The column in adata.obs containing batch information.
    method : str
        Method for batch correction. Options are:
        - bbknn
        - mnn
        - harmony
        - scanorama
        - combat
    highly_variable : bool, default True
        Only for method 'mnn'. If True, only the highly variable genes (column 'highly_variable' in .var) will be used for batch correction.
    """

    method = method.lower()

    print(f"Running batch correction with '{method}'...")

    # Check that batch_key is in adata object
    if batch_key not in adata.obs.columns:
        raise ValueError(f"The given batch_key '{batch_key}' is not in adata.obs.columns")

    # Run batch correction depending on method
    if method == "bbknn":
        adata = sce.pp.bbknn(adata, batch_key=batch_key, copy=True)  # bbknn is an alternative to neighbors

    elif method == "mnn":

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
                                                    batch_categories=batch_categories, do_concatenate=False)

        # Join corrected adatas
        corrected_adatas = corrected_adatas[0]  # the output is a dict of list ([adata1, adata2, (...)], )
        adata = anndata.concat(corrected_adatas, join="outer", uns_merge="first")

        sc.pp.scale(adata)  # from the mnnpy github example
        sc.tl.pca(adata)  # rerun pca
        sc.pp.neighbors(adata)

    elif method == "harmony":
        adata = adata.copy()  # there is no copy option for harmony

        sce.pp.harmony_integrate(adata, key=batch_key)
        adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]
        sc.pp.neighbors(adata)

    elif method == "scanorama":
        adata = adata.copy()  # there is no copy option for scanorama

        # scanorama expect the batch key in a sorted format
        # therefore anndata.obs should be sorted based on batch column before this method.
        adata = adata[adata.obs[batch_key].argsort()]  # sort the whole adata to make sure obs is the same order as matrix

        sce.pp.scanorama_integrate(adata, key=batch_key)
        adata.obsm["X_pca"] = adata.obsm["X_scanorama"]
        sc.pp.neighbors(adata)

    elif method == "combat":

        corrected_mat = sc.pp.combat(adata, key=batch_key, inplace=False)

        adata = adata.copy()  # make sure adata is not modified
        adata.X = sparse.csr_matrix(corrected_mat)

        sc.pp.pca(adata)
        sc.pp.neighbors(adata)

    else:
        raise ValueError(f"Method '{method}' is not a valid batch correction method.")

    return adata  # the corrected adata object


# --------------------------- Automatic thresholds ------------------------- #

def establishing_cuts(data2, interval, skew_val, kurtosis_norm, df_cuts, param2, condi2):
    """
    Defining cutoffs for anndata.obs and anndata.var parameters

    TODO change param names to be more meaningful

    Parameters
    ----------
    data2 : pandas.core.series.Series
        Dataset to be have the cutoffs calculated.
    interval : int or float
        A percentage value from 0 to 1 or 100 to be used to calculate the confidence interval or percentile
    skew_val : int
        The skew value of the dataset under evaluated
    kurtosis_norm : int
        The normalized kurtosis value of the dataset under evaluated
    df_cuts : pandas.DataFrame
        An empty dataframe with data_to_evaluate, parameters, cutoff and strategy as columns to be filled.
        The rows will be the information of each sample analysed by this function.
    param2 : str
        The name of anndata.obs or anndata.var parameter to be evaluated by this function
    condi2 : str
        The name of anndata.obs sample to be evaluated by this function

    Returns
    -------
    pandas.DataFrame :
        Pandas dataframe with cutoffs for each parameter and dataset.

    Notes
    -----
    Author: Guilherme Valente
    """

    def filling_df_cut(df_cuts2, condi3, param3, lst_cuts, lst_dfcuts_cols):
        """ TODO add documentation """
        if condi3 is None:
            df_cuts2 = df_cuts2.append({lst_dfcuts_cols[0]: "_", lst_dfcuts_cols[1]: param3, lst_dfcuts_cols[2]: lst_cuts, lst_dfcuts_cols[3]: "filter_genes"}, ignore_index=True)
        else:
            df_cuts2 = df_cuts2.append({lst_dfcuts_cols[0]: condi3, lst_dfcuts_cols[1]: param3, lst_dfcuts_cols[2]: lst_cuts, lst_dfcuts_cols[3]: "filter_cells"}, ignore_index=True)
        return df_cuts2

    # Defining the types of distributions to be evaluated and organizing the data
    lst_distr, curves, directions = ["uniform", "expon", "powerlaw", "norm"], ["convex", "concave"], ["increasing", "decreasing"]
    lst_dfcuts_cols, np_data2, lst_data2 = df_cuts.columns.tolist(), data2.to_numpy(), data2.tolist()
    knns = list()

    # This is a normal distribution
    if skew_val == 0:
        cut_right, cut_left = np.percentile(np_data2, (interval * 100)), np.percentile(np_data2, 100 - (interval * 100))  # Percentile
        join_cuts = [cut_right, cut_left]
        if 0 in join_cuts:
            join_cuts = [max(join_cuts)]
        df_cutoffs = filling_df_cut(df_cuts, condi2, param2, join_cuts, lst_dfcuts_cols)

    # This is a mesokurtic skewed distributed (long tail and not sharp)
    elif skew_val != 0 and kurtosis_norm == 0:
        cut_right, cut_left = np.percentile(np_data2, (interval * 100)), np.percentile(np_data2, 100 - (interval * 100))  # Percentile
        join_cuts = [cut_right, cut_left]
        if 0 in join_cuts:
            join_cuts = [max(join_cuts)]
        df_cutoffs = filling_df_cut(df_cuts, condi2, param2, join_cuts, lst_dfcuts_cols)

    # This is a skewed distribution (long tail), and platykurtic (not extremely sharp, not so long tail) or leptokurtic (extremely sharp, long tail)
    elif skew_val != 0 and kurtosis_norm != 0:
        f = Fitter(np_data2, distributions=lst_distr)
        f.fit()
        best_fit = list(f.get_best().keys())  # Finding the best fit

        # This is the power law or exponential distributed data
        if "expon" in best_fit or "powerlaw" in best_fit:
            lst_data2.sort()
            histon2, bins_built = np.histogram(a=lst_data2, bins=int(len(lst_data2) / 100), weights=range(0, len(lst_data2), 1))
            for a in curves:
                for b in directions:
                    knns2 = KneeLocator(x=range(1, len(histon2) + 1), y=histon2, curve=a, direction=b)
                    knn2_converted = bins_built[knns2.knee - 1].item()
                    if knn2_converted > 0:
                        knns.append(knn2_converted)
            kn_selected = [min(knns)]
            df_cutoffs = filling_df_cut(df_cuts, condi2, param2, kn_selected, lst_dfcuts_cols)

        # This is the skewed shaped but not like exponential nor powerlaw
        else:
            cut_right, cut_left = np.percentile(np_data2, (interval * 100)), np.percentile(np_data2, 100 - (interval * 100))  # Percentile
            join_cuts = [cut_right, cut_left]
            if 0 in join_cuts:
                join_cuts = [max(join_cuts)]
            df_cutoffs = filling_df_cut(df_cuts, condi2, param2, join_cuts, lst_dfcuts_cols)

    return df_cutoffs


def qcmetric_calculator(anndata, control_var=False):
    """
    Calculating the qc metrics using the Scanpy

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object the quality metrics are added to.
    control_var : boolean, default False
        TODO this parameter is not used!!!

        If True, the adata.uns["infoprocess"]["gene_labeled"] will be used in the qc_var to control the metrics calculation
        The qc_var of sc.pp.calculate_qc_metrics will use this variable to control the qc metrics calculation (e.g. "is_mito").
        For details, see qc_vars at https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.calculate_qc_metrics.html

    Notes
    -----
    Author: Guilherme Valente

    Returns
    -------
    anndata.AnnData:
        Returns anndata object with added quality metrics.
    """
    # Message and others
    m1 = "pp.calculate_qc_metrics qc_vars: "
    m2 = "ID_"
    obs_info = ['total_counts', 'n_genes_by_counts', 'log1p_total_counts']
    var_info = ['n_cells_by_counts', 'mean_counts', 'log1p_mean_counts', 'pct_dropout_by_counts']

    var = anndata.uns["infoprocess"]["genes_labeled"]
    if var is not None:
        qc_metrics = sc.pp.calculate_qc_metrics(adata=anndata, qc_vars=var, inplace=False, log1p=True)
        for a in var:
            obs_info.append('total_counts_' + a)
            obs_info.append('pct_counts_' + a)
    else:
        qc_metrics = sc.pp.calculate_qc_metrics(adata=anndata, inplace=False, log1p=True)

    for a in list(qc_metrics[0].columns.values):
        if a in obs_info:
            anndata.obs[a] = qc_metrics[0][a]
    for a in list(qc_metrics[1].columns.values):
        if a in var_info:
            anndata.var[a] = qc_metrics[1][a]

    # Annotating into anndata.uns["infoprocess"] the qc_var parameter for the sc.pp.calculate_qc_metrics
    cr.build_infor(anndata, m1, var)

    # Storing the original counts
    for a in obs_info:
        go_to_id = anndata.obs[a].sum()
        cr.build_infor(anndata, m2 + "c_" + a, go_to_id)
    for a in var_info:
        go_to_id = anndata.var[a].sum()
        cr.build_infor(anndata, m2 + "g_" + a, go_to_id)

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


def evaluate_batch_effect(adata_dict, batch_key = 'batch'):
    """
    Evaluate batch effect methods using LISI.

    Parameters:
    -----------
    adata_dict : dict
        dictionary containing anndata objects as value and correction method as key.
    batch_key : str, default 'batch'
        The column in adata.obs containing batch information.
    
    Returns:
    --------
    AxesSubplot
        Boxplot containing lisi scores for every correction method.
    """
    # Load LISI
    from harmonypy.lisi import compute_lisi

    # setup empty pandas data
    merged_lisi_scores = pd.DataFrame()

    for method, adata in adata_dict.items():
        # run lisi for each adata.obsm['X_umap']. What about PCA? both?
        lisi_res = compute_lisi(adata.obsm['X_umap'], adata.obs, [batch_key])
        merged_lisi_scores[method] = lisi_res.flatten().tolist()

    # Plot
    return plot.boxplot(merged_lisi_scores)
