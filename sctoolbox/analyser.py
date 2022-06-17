#Loading packages
import scanpy as sc
import sys
from sctoolbox.creators import *
from sctoolbox.checker import *
from fitter import Fitter
import numpy as np
from scipy.stats import *
import scipy.stats as st
from kneed import KneeLocator
import os.path
from os import path

###################################################################################################################
def establishing_cuts(DATA2, INTERVAL, SKEW_VAL, KURTOSIS_NORM, DF_CUTS, PARAM2, CONDI2): #Ir para analyser.py
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
    #Author: Guilherme Valente
    def filling_df_cut(DF_CUTS2, CONDI3, PARAM3, LST_CUTS, LST_DFCUTS_COLS):
        if CONDI3 == None:
            DF_CUTS2=DF_CUTS2.append({LST_DFCUTS_COLS[0]: "_", LST_DFCUTS_COLS[1]: PARAM3, LST_DFCUTS_COLS[2]: LST_CUTS, LST_DFCUTS_COLS[3]: "filter_genes"}, ignore_index=True)
        else:
            DF_CUTS2=DF_CUTS2.append({LST_DFCUTS_COLS[0]: CONDI3, LST_DFCUTS_COLS[1]: PARAM3, LST_DFCUTS_COLS[2]: LST_CUTS, LST_DFCUTS_COLS[3]: "filter_cells"}, ignore_index=True)
        return DF_CUTS2
#Defining the types of distributions to be evaluated and organizing the data
    lst_distr, lst_dfcuts_cols, np_data2, lst_data2=["uniform", "expon", "powerlaw", "norm"], DF_CUTS.columns.tolist(), DATA2.to_numpy(), DATA2.tolist()
#This is a normal distribution
    if SKEW_VAL == 0:
        cut_right, cut_left=np.percentile(np_data2, (INTERVAL*100)), np.percentile(np_data2, 100-(INTERVAL*100)) #Percentile
        join_cuts=[cut_right, cut_left]
        if 0 in join_cuts:
            join_cuts=[max(join_cuts)]
        df_cutoffs = filling_df_cut(DF_CUTS, CONDI2, PARAM2, join_cuts, lst_dfcuts_cols)
#This is a mesokurtic skewed distributed (long tail and not sharp)
    elif SKEW_VAL != 0 and KURTOSIS_NORM == 0:
        cut_right, cut_left=np.percentile(np_data2, (INTERVAL*100)), np.percentile(np_data2, 100-(INTERVAL*100)) #Percentile
        join_cuts=[cut_right, cut_left]
        if 0 in join_cuts:
            join_cuts=[max(join_cuts)]
        df_cutoffs = filling_df_cut(DF_CUTS, CONDI2, PARAM2, join_cuts, lst_dfcuts_cols)
#This is a skewed distribution (long tail), and platykurtic (not extremely sharp, not so long tail) or leptokurtic (extremely sharp, long tail)
    elif SKEW_VAL != 0 and KURTOSIS_NORM != 0:
        f = Fitter(np_data2, distributions=lst_distr)
        f.fit()
        best_fit=list(f.get_best().keys()) #Finding the best fit
#This is the power law or exponential distributed data
        if  "expon" in best_fit or "powerlaw" in best_fit:
            lst_data2.sort()
            avg=st.tmean(lst_data2)
            histon2, bins_built = np.histogram(a=lst_data2, bins=int(len(lst_data2)/100), weights=range(0, len(lst_data2), 1))
            kni, knd = KneeLocator(x=range(1, len(histon2)+1), y=histon2, curve='convex', direction="increasing"), KneeLocator(x=range(1, len(histon2)+1), y=histon2, curve='convex', direction="decreasing")
            kni_converted, knd_converted = [bins_built[kni.knee-1]], [bins_built[knd.knee-1]]
            kni_avg, knd_avg = [kni_converted, avg], [knd_converted, avg]
            kni_avg_dif, knd_avg_dif = max(kni_avg) - min(kni_avg), max(knd_avg) - min(knd_avg)
            if kni_avg_dif > knd_avg_dif:
                kn_converted=kni_avg_dif
            else:
                kn_converted=knd_avg_dif
            df_cutoffs = filling_df_cut(DF_CUTS, CONDI2, PARAM2, kn_converted, lst_dfcuts_cols)
#This is the skewed shaped but not like exponential nor powerlaw
        else:
            cut_right, cut_left=np.percentile(np_data2, (INTERVAL*100)), np.percentile(np_data2, 100-(INTERVAL*100)) #Percentile
            join_cuts=[cut_right, cut_left]
            if 0 in join_cuts:
                join_cuts=[max(join_cuts)]
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
    #Author: Guilherme Valente
    #Message and others
    m1="pp.calculate_qc_metrics qc_vars: "
    m2="ID_"
    obs_info=['total_counts', 'n_genes_by_counts', 'log1p_total_counts']
    var_info=['n_cells_by_counts', 'mean_counts', 'log1p_mean_counts', 'pct_dropout_by_counts']

    VAR=ANNDATA.uns["infoprocess"]["genes_labeled"]
    if VAR != None:
        qc_metrics = sc.pp.calculate_qc_metrics(adata = ANNDATA, qc_vars = VAR, inplace = False, log1p=True)
        for a in VAR:
            obs_info.append('total_counts_' + a)
            obs_info.append('pct_counts_' + a)
    else:
        qc_metrics = sc.pp.calculate_qc_metrics(adata = ANNDATA, inplace = False, log1p=True)
    for a in list(qc_metrics[0].columns.values):
        if a in obs_info:
            ANNDATA.obs[a] = qc_metrics[0][a]
    for a in list(qc_metrics[1].columns.values):
        if a in var_info:
            ANNDATA.var[a] = qc_metrics[1][a]
    #Annotating into anndata.uns["infoprocess"] the qc_var parameter for the sc.pp.calculate_qc_metrics
    build_infor(ANNDATA, m1, VAR)
    #Storing the original counts
    for a in obs_info:
        go_to_id=ANNDATA.obs[a].sum()
        build_infor(ANNDATA, m2 + "c_" + a, go_to_id)
    for a in var_info:
        go_to_id=ANNDATA.var[a].sum()
        build_infor(ANNDATA, m2 + "g_" + a, go_to_id)
    return(ANNDATA)
