# Loading packages
import scanpy as sc
import sctoolbox.creators as cr
import sctoolbox.annotation as an
from fitter import Fitter
import numpy as np
from kneed import KneeLocator
import scipy.stats


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
    if not limit_on in ["upper", "lower", "both"]:
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
