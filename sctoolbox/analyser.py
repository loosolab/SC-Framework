# Loading packages
import scanpy as sc
import sctoolbox.creators as creators
from fitter import Fitter
import numpy as np
from kneed import KneeLocator
import scanpy.external as sce
from scipy import sparse


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
        The method(s) to use for batch correction. Options are: (...).
        Default: []
    """

    # TODO: check if methods are valid
    if isinstance(methods, str):
        methods = [methods]

    # Collect batch correction per method
    anndata_dict = {}
    for method in methods:
        anndata_dict[method] = batch_correction(adata, batch_key, method)  # batch correction returns the corrected adata

    anndata_dict['uncorrected'] = adata

    return anndata_dict


def batch_correction(adata, batch_key, method):
    """
    Perform batch correction on the adata object using the 'method' given.

    Parameters
    -----------
    adata : anndata.AnnData
        An annotated data matrix object to apply corrections to.
    batch_key : str
        The column in adata.obs containing batch information.
    method : str
        Method for batch correction. Options are: (....)
    """

    method = method.lower()

    # TODO: check that batch_key is in adata object

    # Run batch correction depending on method
    if method == "bbknn":
        adata = sce.pp.bbknn(adata, batch_key=batch_key, copy=True)  # bbknn is an alternative to neighbors

    elif method == "mnn":

        # split adata on batch_key
        batch_categories = list(set(adata.obs[batch_key]))
        adatas = [adata[adata.obs[batch_key] == category] for category in batch_categories]

        # TODO: enable var_subset as input

        # give individual adatas to mnn_correct
        adata, _, _ = sce.pp.mnn_correct(adatas, batch_key=batch_key, batch_categories=batch_categories, do_concatenate=True)
        
        # sc.pp.scale expect only adata object, which is the first element of the output list;
        # therfore:
        adata = adata[0][0]

        sc.pp.scale(adata)  # from the mnnpy github example
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
        adata.obs = adata.obs.sort_values(batch_key)

        sce.pp.scanorama_integrate(adata, key=batch_key)
        adata.obsm["X_pca"] = adata.obsm["X_scanorama"]
        sc.pp.neighbors(adata)

    elif method == "combat":

        corrected_mat = sc.pp.combat(adata, key=batch_key, inplace=False)

        adata = adata.copy()  # make sure adata is not modified
        adata.X = sparse.csr_matrix(corrected_mat)

    else:
        raise ValueError(f"Method '{method}' is not a valid batch correction method.")

    return adata  # the corrected adata object

# --------------------------- Automatic thresholds ------------------------- #


def establishing_cuts(DATA2, INTERVAL, SKEW_VAL, KURTOSIS_NORM, DF_CUTS, PARAM2, CONDI2):  # Ir para analyser.py
    '''
    Defining cutoffs for anndata.obs and anndata.var parameters

    Parameters
    ------------
    DATA2 : pandas.core.series.Series
        Dataset to be have the cutoffs calculated
    INTERVAL : Int or float
        A percentage value from 0 to 1 or 100 to be used to calculate the confidence interval or percentile
    SKEW_VAL : Int
        The skew value of the dataset under evaluated
    KURTOSIS_NORM : Int
        The normalized kurtosis value of the dataset under evaluated
    DF_CUTS : Pandas dataframe
        An empty dataframe with data_to_evaluate, parameters, cutoff and strategy as columns to be filled.
        The rows will be the information of each sample analysed by this function.
    PARAM2 : String
        The name of anndata.obs or anndata.var parameter to be evaluated by this function
    CONDI2 : String
        The name of anndata.obs sample to be evaluated by this function

    Return
    ------------
    Pandas dataframe with cutoffs for each parameter and dataset
    '''
    # Author: Guilherme Valente
    def filling_df_cut(DF_CUTS2, CONDI3, PARAM3, LST_CUTS, LST_DFCUTS_COLS):
        if CONDI3 is None:
            DF_CUTS2 = DF_CUTS2.append({LST_DFCUTS_COLS[0]: "_", LST_DFCUTS_COLS[1]: PARAM3, LST_DFCUTS_COLS[2]: LST_CUTS, LST_DFCUTS_COLS[3]: "filter_genes"}, ignore_index=True)
        else:
            DF_CUTS2 = DF_CUTS2.append({LST_DFCUTS_COLS[0]: CONDI3, LST_DFCUTS_COLS[1]: PARAM3, LST_DFCUTS_COLS[2]: LST_CUTS, LST_DFCUTS_COLS[3]: "filter_cells"}, ignore_index=True)
        return DF_CUTS2

    # Defining the types of distributions to be evaluated and organizing the data
    lst_distr, curves, directions = ["uniform", "expon", "powerlaw", "norm"], ["convex", "concave"], ["increasing", "decreasing"]
    lst_dfcuts_cols, np_data2, lst_data2 = DF_CUTS.columns.tolist(), DATA2.to_numpy(), DATA2.tolist()
    knns = list()

    # This is a normal distribution
    if SKEW_VAL == 0:
        cut_right, cut_left = np.percentile(np_data2, (INTERVAL * 100)), np.percentile(np_data2, 100 - (INTERVAL * 100))  # Percentile
        join_cuts = [cut_right, cut_left]
        if 0 in join_cuts:
            join_cuts = [max(join_cuts)]
        df_cutoffs = filling_df_cut(DF_CUTS, CONDI2, PARAM2, join_cuts, lst_dfcuts_cols)

    # This is a mesokurtic skewed distributed (long tail and not sharp)
    elif SKEW_VAL != 0 and KURTOSIS_NORM == 0:
        cut_right, cut_left = np.percentile(np_data2, (INTERVAL * 100)), np.percentile(np_data2, 100 - (INTERVAL * 100))  # Percentile
        join_cuts = [cut_right, cut_left]
        if 0 in join_cuts:
            join_cuts = [max(join_cuts)]
        df_cutoffs = filling_df_cut(DF_CUTS, CONDI2, PARAM2, join_cuts, lst_dfcuts_cols)

    # This is a skewed distribution (long tail), and platykurtic (not extremely sharp, not so long tail) or leptokurtic (extremely sharp, long tail)
    elif SKEW_VAL != 0 and KURTOSIS_NORM != 0:
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
            df_cutoffs = filling_df_cut(DF_CUTS, CONDI2, PARAM2, kn_selected, lst_dfcuts_cols)

        # This is the skewed shaped but not like exponential nor powerlaw
        else:
            cut_right, cut_left = np.percentile(np_data2, (INTERVAL * 100)), np.percentile(np_data2, 100 - (INTERVAL * 100))  # Percentile
            join_cuts = [cut_right, cut_left]
            if 0 in join_cuts:
                join_cuts = [max(join_cuts)]
            df_cutoffs = filling_df_cut(DF_CUTS, CONDI2, PARAM2, join_cuts, lst_dfcuts_cols)
    return(df_cutoffs)


def qcmetric_calculator(ANNDATA, control_var=False):
    '''
    Calculating the qc metrics using the Scanpy

    Parameters
    ------------
    ANNDATA : anndata object
        adata object
    control_var : Boolean
        If True, the adata.uns["infoprocess"]["gene_labeled"] will be used in the qc_var to control the metrics calculation
        The qc_var of sc.pp.calculate_qc_metrics will use this variable to control the qc metrics calculation (e.g. "is_mito").
        For details, see qc_vars at https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.calculate_qc_metrics.html
    '''
    # Author: Guilherme Valente
    # Message and others
    m1 = "pp.calculate_qc_metrics qc_vars: "
    m2 = "ID_"
    obs_info = ['total_counts', 'n_genes_by_counts', 'log1p_total_counts']
    var_info = ['n_cells_by_counts', 'mean_counts', 'log1p_mean_counts', 'pct_dropout_by_counts']

    VAR = ANNDATA.uns["infoprocess"]["genes_labeled"]
    if VAR is not None:
        qc_metrics = sc.pp.calculate_qc_metrics(adata=ANNDATA, qc_vars=VAR, inplace=False, log1p=True)
        for a in VAR:
            obs_info.append('total_counts_' + a)
            obs_info.append('pct_counts_' + a)
    else:
        qc_metrics = sc.pp.calculate_qc_metrics(adata=ANNDATA, inplace=False, log1p=True)

    for a in list(qc_metrics[0].columns.values):
        if a in obs_info:
            ANNDATA.obs[a] = qc_metrics[0][a]
    for a in list(qc_metrics[1].columns.values):
        if a in var_info:
            ANNDATA.var[a] = qc_metrics[1][a]

    # Annotating into anndata.uns["infoprocess"] the qc_var parameter for the sc.pp.calculate_qc_metrics
    creators.build_infor(ANNDATA, m1, VAR)

    # Storing the original counts
    for a in obs_info:
        go_to_id = ANNDATA.obs[a].sum()
        creators.build_infor(ANNDATA, m2 + "c_" + a, go_to_id)
    for a in var_info:
        go_to_id = ANNDATA.var[a].sum()
        creators.build_infor(ANNDATA, m2 + "g_" + a, go_to_id)

    return(ANNDATA)
