import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from scipy.stats import *
import scipy
from kneed import KneeLocator
import pandas as pd
from sctoolbox.plotting import *
from sctoolbox.creators import *
from sctoolbox.checker import *
from sctoolbox.analyser import *

########################STEP 1: DEFINING DEFAULT CUTOFFS###############################
#######################################################################################
def set_def_cuts(ANNDATA, only_plot=False, interval=None, file_name=None, save=None):
    '''
    1- Definining the cutoffs for QC and filtering steps.
    2- Ploting anndata obs or anndata var selected for the QC and filtering steps with the cutoffs or not.

    Parameters
    ------------
    ANNDATA : anndata object
        anndata object
    only_plot : Boolean
        If true, only a plot with the data without cutoff lines will be provided.
    interval : Int or Float. Default: None.
        The percentage (from 0 to 1 or 100) to be used to calculate the cutoffs.
    file_name : String
        If you wanna use a custom filename to save.
    save : Boolean
        True, save the figure to the path given in 'save'. Default: None (figure is not saved).

    Returns
    ------------
    - A pandas dataframe with the defined cutoff parameters.
    - Violin plots.
    '''
    #Author: Guilherme Valente
    #List, dfs, messages and others
    uns_condition_name=ANNDATA.uns["infoprocess"]["data_to_evaluate"] #The name of data to evaluate parameter, e.g., "condition"
    for_cells=[uns_condition_name, "n_genes_by_counts", "total_counts", "pct_counts_is_mitochondrial"] #List of obs variables to be analysed. The first item MUST be the data to be evaluated
    for_genes=["n_cells_by_counts", "mean_counts", "pct_dropout_by_counts"] #List of var variables to be analysed.
    for_cells_pd=ANNDATA.obs[ANNDATA.obs.columns.intersection(for_cells)]
    for_genes_pd=ANNDATA.var[ANNDATA.var.columns.intersection(for_genes)]
    cells_genes_list=for_cells[1:] + for_genes
    calulate_and_plot_filter=[] #Samples to be calculated cutoffs and further plotted with cutoff lines
    df_cuts=pd.DataFrame(columns=["data_to_evaluate", "parameters", "cutoff", "strategy"]) #Empty df to be filled with the cutoffs
    ANNDATA.uns[uns_condition_name + '_colors']=ANNDATA.uns["color_set"][:len(ANNDATA.obs[uns_condition_name].unique())]
    sta_cut_cells, sta_cut_genes=None, None
    #Setting colors for plot
    m2="The interval=[float or int] must be between 0 to 1 or 100."
    m3="violin_plot"
    m4="Defining and ploting cutoffs only for total_counts"
    m5="You choose not plot cutoffs"
    m6="To define cutoff demands to set the interval=[int or float] from 0 to 1 or 100."

    if only_plot == False:
        if interval == None: #The cutoff will be performed, but it is missing the interval
            sys.exit(m6)
        else:
            filename=m3 + "_cutoff"
#Checking if interval for cutoffs were properly defined
            if check_cuts(str(interval), 0, 100) == "invalid": #Means the interval is not a number
                    sys.exit(m2)
            else:
                if isinstance(interval, (int)) == True and interval >= 0 and interval <= 100: #Converting interval int for float from 0 to 1
                    interval=interval/100
#Checking if the total count was already filtered
            act_c_total_counts, id_c_total_counts=float(ANNDATA.obs["total_counts"].sum()), float(ANNDATA.uns["infoprocess"]["ID_c_total_counts"])
            if check_cuts(str(act_c_total_counts), id_c_total_counts, id_c_total_counts) == "valid": #Total_counts was not filtered yet, then only this parameter will be evaluated
                sta_cut_cells, sta_cut_genes=True, False
                calulate_and_plot_filter.append("total_counts")
                filename=m3 + "_total_count_cutoff"
                print(m4)
#Other parameters will be evaluated because plot was selected and total count is filtered
            elif check_cuts(str(act_c_total_counts), id_c_total_counts, id_c_total_counts) == "invalid": #Total counts was filtered yet.
                sta_cut_cells, sta_cut_genes=True, True
                filename=m3 + "_other_param_cutoff"
                for a in cells_genes_list:
                    if a != "total_counts":
                        calulate_and_plot_filter.append(a)
    else: #Only plots without cutoff lines will be provided
        filename=m3

#Defining filenames and path to save the figure
    if save == True:
        save_path=ANNDATA.uns["infoprocess"]["Anndata_path"]
        if type(file_name) == str: #The violing plot will have a custom filename
            filename=file_name
    else:
        save_path, filename=None, None

#Building the dataframe with default cutoffs stablished and the plots
#Here will be called the function establishing_cuts from the analyser.py module
    if only_plot == True: #Only plot without cutoffs
        print(m5)
        return qcf_ploting(for_cells_pd, for_genes_pd, ANNDATA.uns[for_cells[0] + "_colors"], df_cuts, PLOT=None, SAVE=save, SAVE_PATH=save_path, FILENAME=filename)
    elif only_plot == False: #Calculate cutoffs and plot in the violins
        if sta_cut_cells == True: #For cells
            for a in for_cells_pd[for_cells[0]].unique().tolist(): #Getting the conditions.
                adata_cond=for_cells_pd[for_cells_pd[for_cells[0]] == a].copy()
                for b in for_cells[1:]:
                    if b in calulate_and_plot_filter:
                        data=adata_cond[b]
                        skew_val, kurtosis_val=round(skew(data.to_numpy(), bias=False), 1), int(kurtosis(data.to_numpy(), fisher=False, bias=False)) #Calculating the skewness and kurtosis
                        kurtosis_val_norm=int(kurtosis_val - 3) #This is a normalization for kurtosis value to identify the excessive kurtosis. Cite: https://www.sciencedirect.com/topics/mathematics/kurtosis
                        df_cuts=establishing_cuts(data, interval, skew_val, kurtosis_val_norm, df_cuts, b, a)
        if sta_cut_genes == True: #For genes
            for a in for_genes:
                if a in calulate_and_plot_filter: #Calculate cutoffs only for selected parameters
                    data=for_genes_pd[a]
                    skew_val, kurtosis_val=round(skew(data.to_numpy(), bias=False), 1), int(kurtosis(data.to_numpy(), fisher=False, bias=False)) #Calculating the skewness and kurtosis
                    kurtosis_val_norm=int(kurtosis_val - 3) #This is a normalization for kurtosis value to identify the excessive kurtosis. Cite: https://www.sciencedirect.com/topics/mathematics/kurtosis
                    df_cuts=establishing_cuts(data, interval, skew_val, kurtosis_val_norm, df_cuts, a, None)
        display(df_cuts)
        qcf_ploting(for_cells_pd, for_genes_pd, ANNDATA.uns[for_cells[0] + "_colors"], df_cuts, PLOT=calulate_and_plot_filter, SAVE=save, SAVE_PATH=save_path, FILENAME=filename)
        return df_cuts

########################STEP 2: DEFINING CUSTOM CUTOFFS###############################
######################################################################################
def refining_cuts(ANNDATA, def_cut2):
    '''
    Refining the cutoffs to be used in the QC and filtering steps

    Parameters
    ------------
    ANNDATA : anndata object
        anndata object
    def_cut2 : Pandas dataframe.
        A dataframe with the default cutoffs calculated by the function set_def_cuts of qc_filter.py module

    Return
    -----------
    Pandas dataframe to be used for the QC and filtering steps
    '''
    #Author: Guilherme Valente
    #Dataframes and others
    dfcut=def_cut2.copy()
    new_df=dfcut[0:0]
    col0=dfcut.columns[0] #data_to_evaluate
    col1=dfcut.columns[1] #parameters
    col2=dfcut.columns[2] #cutoffs
    col3=dfcut.columns[3] #strategy
    cond_name=ANNDATA.uns["infoprocess"]["data_to_evaluate"]
    #Questions, and messages
    q2=": custom or def"
    q3=": minimun cutoff [FLOAT or INT]"
    q4=": maximum cutoff [FLOAT or INT]"
    q5=": custom, def or skip"
    m1="\n\nChoose y or n\n"
    m2="The default setting is:"
    m3="\n\n#########################################################\nNOTE. Choose: \n\t1) custom to define new cutoffs\n\t2) def to use the previously defined cutoffs\n#########################################################\n"
    m4="\n\nChoose custom or def\n"
    m5="\n\nType a positive number "
    m6="<= "
    m7="\n"
    m8="\n\n#########################################################\nNOTE. Choose: \n\t1) custom to define new cutoffs\n\t2) def to use the previously defined cutoffs\n\t3) skip to avoid filter this parameter\n#########################################################\n"
    m9="\n\nChoose custom, def or skip\n"
    #Checking if the total_counts was already filtered
    if ANNDATA.obs["total_counts"].sum() == ANNDATA.uns["infoprocess"]["ID_c_total_counts"]: #If False, means that the total_counts was not filtered yet
        df_tot_coun=dfcut[dfcut[col1]== "total_counts"].reset_index(drop=True)
        is_total_count_filt=False
    else:
        df_NOT_tot_coun=dfcut[dfcut[col1]!= "total_counts"].reset_index(drop=True)
        is_total_count_filt=True
    def loop_question(ANSWER, QUIT_M, CHECK): #Checking invalid outcome
        '''
        This core check the validity of user's answer.
        Parameters
        ---------------
        ANSWER : String
            User's answer
        QUIT_M : String
            Question to be print if the user's answer is invalid
        CHECK : String
            Types of questions. The options are "yn", "custdef", or float

        Returns
        ---------------
        The user's answer in lower case format
        '''
        #Author: Guilherme Valente
        #List and others
        opt1, opt2, opt3, opt4=["q", "quit"], ["y", "yes", "n", "no"], ["custom", "def"], ["custom", "def", "skip"] #Lists with possible answers for each question
        m1="Invalid choice! "
        #Executing
        ANSWER=input(ANSWER).lower()
        if ANSWER in opt1:
            sys.exit("You quit and lost all modifications :(")
        if CHECK == "yn": #Checking validity of Y or N question
            while check_options(ANSWER, opt1, opt2) == "invalid":
                print(m1 + QUIT_M)
                ANSWER=input()
                check_options(ANSWER, opt1, opt2)
        elif CHECK == "custdef": #Checking validity of custom or def question
            while check_options(ANSWER, opt1, opt3) == "invalid":
                print(m1 + QUIT_M)
                ANSWER=input()
                check_options(ANSWER, opt1, opt3)
        elif CHECK == "custdefskip": #Checking validity of custom, def or skip question
            while check_options(ANSWER, opt1, opt4) == "invalid":
                print(m1 + QUIT_M)
                ANSWER=input()
                check_options(ANSWER, opt1, opt4)
        elif type(CHECK) == float: #Checking validity of custom value: >=0 and <= a limit (the CHECK)
            while check_cuts(ANSWER, 0, CHECK) == "invalid":
                print(m1 + QUIT_M)
                ANSWER=input()
                check_cuts(ANSWER, 0, CHECK)
        return(ANSWER.lower())
#Executing the main function
    if is_total_count_filt == False: #It means that total_counts was not filtered. Then, the dataframe has only total_count information
        print(m2)
        display(df_tot_coun)
        print(m3)
        for idx, a in dfcut.iterrows():
            data, param, list_cuts, stra =a[col0], a[col1], a[col2], a[col3]
            answer=loop_question(data + " " + param + q2, m4, "custdef") #Custom or default?
            if answer == "def":
                new_df=new_df.append({col0: data, col1: param, col2: list_cuts, col3: stra}, ignore_index=True)
            elif answer == "custom":
                max_lim=max(ANNDATA.obs[ANNDATA.obs[cond_name] == data][param])
                min_cut=loop_question(data + " " + param + q3, m5 + m6 + " " + str(max_lim) + m7, float(max_lim))
                max_cut=loop_question(data + " " + param + q4, m5 + m6 + " " + str(max_lim) + m7, float(max_lim))
                list_cuts=sorted([float(min_cut), float(max_cut)], reverse=True)
                new_df=new_df.append({col0: data, col1: param, col2: list_cuts, col3: stra}, ignore_index=True)
    else: #It means that total_counts was filtered. Then, the dataframe only has parameters for other information
        print(m2)
        display(df_NOT_tot_coun)
        print(m8)
        for idx, a in dfcut.iterrows():
            data, param, list_cuts, stra =a[col0], a[col1], a[col2], a[col3]
            answer=loop_question(data + " " + param + q5, m9, "custdefskip") #Custom, default or skip?
            if answer == "def":
                new_df=new_df.append({col0: data, col1: param, col2: list_cuts, col3: stra}, ignore_index=True)
            elif answer == "custom":
                if stra == "filter_cells":
                    max_lim=max(ANNDATA.obs[ANNDATA.obs[cond_name] == data][param])
                elif stra == "filter_genes":
                    max_lim=max(ANNDATA.var[param])
                min_cut=loop_question(data + " " + param + q3, m5 + m6 + " " + str(max_lim) + m7, float(max_lim))
                max_cut=loop_question(data + " " + param + q4, m5 + m6 + " " + str(max_lim) + m7, float(max_lim))
                list_cuts=sorted([float(min_cut), float(max_cut)], reverse=True)
                new_df=new_df.append({col0: data, col1: param, col2: list_cuts, col3: stra}, ignore_index=True)
            elif answer == "skip":
                new_df=new_df.append({col0: data, col1: param, col2: "skip", col3: stra}, ignore_index=True)
    return new_df

########################STEP 3: APPLYING CUTOFFS###############################
###############################################################################
def anndata_filter(ANNDATA, GO_CUT):
    '''
    Parameters
    ----------
    ANNDATA : anndata object
        anndata object
    GO_CUT : Pandas dataframe
        Pandas dataframe with the samples, parameters, and cutoffs to be used in the filtering process

    Return
    ----------
    Filtered anndata
    Annotation of filtering parameters in the anndata.uns["infoprocess"]
    '''
    #Author: Guilherme Valente
    def concatenating(LISTA_ADATA): #Concatenating adata
        if len(LISTA_ADATA) > 1:
            adata_conc=LISTA_ADATA[0].concatenate(LISTA_ADATA[1:], join='inner', batch_key=None)
        else:
            adata_conc=LISTA_ADATA[0]
        return adata_conc

    #List, messages and others
    m1="" #Fill message to go to anndata.uns["infoprocess"]
    ANNDATA_CP=ANNDATA.copy()
    ANNDATA_CP2=ANNDATA_CP.copy()
    datamcol, paramcol, cutofcol, stratcol = GO_CUT.columns[0], GO_CUT.columns[1], GO_CUT.columns[2], GO_CUT.columns[3]
    act_params=GO_CUT[paramcol].unique().tolist()
    act_strate=GO_CUT[stratcol].unique().tolist()
    uns_cond=ANNDATA.uns["infoprocess"]["data_to_evaluate"]
    lst_adata_sub, go_info_cell, go_info_genes =list(), [], []

    #Creating the infoprocess
    if "Cell filter" not in ANNDATA_CP.uns["infoprocess"]:
        build_infor(ANNDATA_CP, "Cell filter", list())
    if "Gene filter" not in ANNDATA_CP.uns["infoprocess"]:
        build_infor(ANNDATA_CP, "Gene filter", list())

    #Filter mitochondrial content first
    if "pct_counts_is_mitochondrial" in act_params:
        raw_data=GO_CUT[GO_CUT[paramcol]=="pct_counts_is_mitochondrial"]
        for idx, a in raw_data.iterrows():
            data, param, cuts, stra =a[datamcol], a[paramcol], a[cutofcol], a[stratcol]
            if "skip" in cuts:
                lst_adata_sub.append(ANNDATA_CP[ANNDATA_CP.obs[uns_cond] == data,:])
            else:
                max_cut=max(cuts)
                m1=data + " " + param + " max_percent= " + str(max_cut)
                adata_sub=ANNDATA_CP[ANNDATA_CP.obs[uns_cond] == data,:]
                lst_adata_sub.append(adata_sub[adata_sub.obs["pct_counts_is_mitochondrial"] < max_cut, :])
                go_info_cell.append(m1)
        ANNDATA_CP2=concatenating(lst_adata_sub)
        lst_adata_sub=list()
    else:
        ANNDATA_CP2=ANNDATA_CP
        lst_adata_sub=list()
    #Filtering other cells
    raw_data=GO_CUT[GO_CUT[stratcol]=="filter_cells"]
    for idx, a in raw_data.iterrows():
        data, param, cuts, stra =a[datamcol], a[paramcol], a[cutofcol], a[stratcol]
        if param != "pct_counts_is_mitochondrial":
            adata_sub=ANNDATA_CP2[ANNDATA_CP2.obs[uns_cond] == data,:]
            if "skip" in cuts:
                lst_adata_sub.append(adata_sub)
            else:
                m1=data + " " + param + " "
                min_cut, max_cut=min(cuts), max(cuts)
                if param == "total_counts":
                    sc.pp.filter_cells(adata_sub, min_counts=min_cut, inplace=True)
                    sc.pp.filter_cells(adata_sub, max_counts=max_cut, inplace=True)
                    lst_adata_sub.append(adata_sub)
                    go_info_cell.append(m1 + "min_counts= " + str(min_cut))
                    go_info_cell.append(m1 + "max_counts= " + str(max_cut))
                elif param == "n_genes_by_counts":
                    sc.pp.filter_cells(adata_sub, min_genes=min_cut, inplace=True)
                    lst_adata_sub.append(adata_sub)
                    go_info_cell.append(m1 + "min_genes= " + str(min_cut))
    ANNDATA_CP2=concatenating(lst_adata_sub)
    lst_adata_sub=list()

    #Filtering genes
    if "filter_genes" in act_strate:
        raw_data=GO_CUT[GO_CUT[stratcol]=="filter_genes"]
        adata_sub=ANNDATA_CP2.copy()
        for idx, a in raw_data.iterrows():
            data, param, cuts, stra =a[datamcol], a[paramcol], a[cutofcol], a[stratcol]
            if "skip" in cuts:
                pass
            else:
                min_cut, max_cut=min(cuts), max(cuts)
                if param == "n_cells_by_counts":
                    sc.pp.filter_genes(adata_sub, min_cells=min_cut, inplace=True)
                    go_info_genes.append("n_cells_by_counts min_cells= " + str(min_cut))
                elif param == "mean_counts":
                    sc.pp.filter_genes(adata_sub, min_counts=min_cut, inplace=True)
                    go_info_genes.append("mean_counts min_counts= " + str(min_cut))
                elif param == "pct_dropout_by_counts":
                    print("pct_dropout_by_counts filtering is not implemented.")
        ANNDATA_CP2=adata_sub

#Annotating anndata.uns["infoprocess"]
    if len(ANNDATA_CP.uns["infoprocess"]["Cell filter"]) > 0:
        ANNDATA_CP.uns["infoprocess"]["Cell filter"] = ANNDATA_CP.uns["infoprocess"]["Cell filter"] + go_info_cell
    else:
        ANNDATA_CP.uns["infoprocess"]["Cell filter"] = go_info_cell
    if len(ANNDATA_CP.uns["infoprocess"]["Gene filter"]) > 0:
        ANNDATA_CP.uns["infoprocess"]["Gene filter"] = ANNDATA_CP.uns["infoprocess"]["Gene filter"] + go_info_genes
    else:
        ANNDATA_CP.uns["infoprocess"]["Gene filter"] = go_info_genes

    ANNDATA_CP2.uns = ANNDATA_CP.uns

    print(ANNDATA)
    print(ANNDATA_CP2)
    return ANNDATA_CP2.copy()

#####################################################################################
#####################################################################################

def filter_genes(adata, genes):
    """ Remove genes from adata object.

    Parameters
    -----------
    adata : anndata.AnnData
        Annotated data matrix object to filter
    genes : list of str
        A list of genes to remove from object.

    Returns
    --------
    adata : anndata.AnnData
        Anndata object with removed genes.
    """

    #Check if all genes are found in adata
    not_found = list(set(genes) - set(adata.var_names))
    if len(not_found) > 0:
        print("{0} genes were not found in adata and could therefore not be removed. These genes are: {1}".format(len(not_found), not_found))

    #Remove genes from adata
    n_before = adata.shape[1]
    adata = adata[:, ~adata.var_names.isin(genes)]
    n_after = adata.shape[1]
    print("Filtered out {0} genes from adata. New number of genes is: {1}.".format(n_before-n_after, n_after))

    return(adata)


def estimate_doublets(adata, threshold=0.25, inplace=True, **kwargs):
    """ Estimate doublet cells using scrublet. Adds additional columns "doublet_score" and "predicted_doublet" in adata.obs,
        as well as a "scrublet" key in adata.uns.

    Parameters
    ------------
    adata : anndata.AnnData
        Anndata object to estimate doublets for.
    threshold : float
        Threshold for doublet detection. Default is 0.25.
    inplace : bool
        Whether to estimate doublets inplace or not. Default is True.
    kwargs : arguments
        Additional arguments are passed to scanpy.external.pp.scrublet.

    Returns
    ---------
    If inplace is False, the function returns a copy of the adata object. 
    If inplace is True, the function returns None.
    """
    
    if inplace == False:
        adata = adata.copy()

    #Run scrublet on adata
    adata_scrublet = sc.external.pp.scrublet(adata, threshold=threshold, **kwargs)

    # Plot the distribution of scrublet scores
    sc.external.pl.scrublet_score_distribution(adata_scrublet)

    #Save scores to object
    adata.obs["doublet_score"] = adata_scrublet.obs["doublet_score"]
    adata.obs["predicted_doublet"] = adata_scrublet.obs["predicted_doublet"]
    adata.uns["scrublet"] = adata_scrublet.uns["scrublet"]

    if inplace == False:
        return adata

