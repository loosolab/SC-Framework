import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets
import sys
from scipy.stats import *
import scipy
from kneed import KneeLocator
import pandas as pd
import functools #for partial functions
from matplotlib.patches import Rectangle
import time

from sctoolbox.plotting import *
from sctoolbox.creators import *
from sctoolbox.checker import *
from sctoolbox.analyser import *

########################STEP 1: DEFINING DEFAULT CUTOFFS###############################
#######################################################################################
def set_def_cuts(ANNDATA, only_plot=False, interval=None, file_name="note2_violin_", save=False):
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
    file_name : String. Default is "note2_violin_"
        Define a name for save a custom filename to save.
        NOTE: use a sintax at least composing the "note2_". Do not add any file extension.
    save : Boolean. Default is False
        True, save the figure to the path given in 'save'.

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
    m1="file_name[STRING]"
    m2="The interval=[float or int] must be between 0 to 1 or 100."
    m4="Defining and ploting cutoffs only for total_counts"
    m5="You choose not plot cutoffs"
    m6="To define cutoff demands to set the interval=[int or float] from 0 to 1 or 100."
    pathway=ANNDATA.uns["infoprocess"]["Anndata_path"]

    #Creating filenames
    if save == True and file_name != "note2_violin_" and type(file_name) != str: #Here checking is custom name is proper
        sys.exist(m1)
    elif save == True and type(file_name) == str: #Custom name is proper. If custom is not provided, the default is used
        filename=pathway + file_name
        filename=filename.replace("//", "/")
    elif save == False:
        filename=""

    #Checking if interval for cutoffs were properly defined
    if only_plot == False and interval == None: #It is missing the interval for cutoff
            sys.exit(m6)
    elif only_plot == False and interval != None:
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
                filename=filename + "tot_count_cut"
                print(m4)
            #Other parameters will be evaluated because plot was selected and total count is filtered
            elif check_cuts(str(act_c_total_counts), id_c_total_counts, id_c_total_counts) == "invalid": #Total counts was filtered yet.
                sta_cut_cells, sta_cut_genes=True, True
                filename=filename + "other_param_cut"
                for a in cells_genes_list:
                    if a != "total_counts":
                        calulate_and_plot_filter.append(a)
    #Ploting with or without cutoffs
    if only_plot == True: #Only plots without cutoff lines will be provided
        filename=filename
        #Building the dataframe with default cutoffs stablished and the plots
        #Here will be called the function establishing_cuts from the analyser.py module
        print(m5)
        return qcf_ploting(for_cells_pd, for_genes_pd, ANNDATA.uns[for_cells[0] + "_colors"], df_cuts, PLOT=None, SAVE=save, FILENAME=filename)
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
        qcf_ploting(for_cells_pd, for_genes_pd, ANNDATA.uns[for_cells[0] + "_colors"], df_cuts, PLOT=calulate_and_plot_filter, SAVE=save, FILENAME=filename)
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
    adata_scrublet = sc.external.pp.scrublet(adata, threshold=threshold, copy=True, **kwargs)

    # Plot the distribution of scrublet scores
    sc.external.pl.scrublet_score_distribution(adata_scrublet)

    #Save scores to object
    adata.obs["doublet_score"] = adata_scrublet.obs["doublet_score"]
    adata.obs["predicted_doublet"] = adata_scrublet.obs["predicted_doublet"]
    adata.uns["scrublet"] = adata_scrublet.uns["scrublet"]

    if inplace == False:
        return adata
        



#####################################################################################
######################## More general QC filtering functions ########################
#####################################################################################

def _validate_minmax(d):
    """
    
    """
    allowed = set(["min", "max"])
    keys = set(d.keys())
    
    not_allowed = len(keys - allowed)
    if not_allowed > 0:
        raise ValueError("Keys {0} not allowed".format(not_allowed))
    
def validate_threshold_dict(table, thresholds, groupby=None):
    """ 
    Validate threshold dictionary. Thresholds can be in the format::
    
    thresholds = {"chrM_percent": {"min": 0, "max": 10},
                  "total_reads": {"min": 1000}}
              
    Or per group in 'groupy':

    thresholds = {"chrM_percent": {
                               "Sample1": {"max": 10},
                               "Sample2": {"max": 5}
                               },
                  "total_reads": {"min": 1000}}
    
    Parameters
    ----------
    table : pandas.DataFrame
        Table to validate thresholds for.
    thresholds : dict
        Dictionary of thresholds to validate.
    groupby : str, optional
        Column for grouping thresholds. Default: None (no grouping)
    """
    
    if groupby is not None:
        groups = table[groupby]
    
    #Check if all columns in thresholds are available
    threshold_columns = thresholds.keys()
    not_found = [col for col in threshold_columns if col not in table.columns]
    if len(not_found) > 0:
        raise ValueError("Column(s) '{0}' given in thresholds are not found in table".format(not_found))
        
    #Check the format of individual column thresholds
    for col in thresholds:
        
        if groupby is None: #expecting one threshold for all cells
            _validate_minmax(thresholds[col])
    
        else: #Expecting either one threshold or a threshold for each sample
            
            for key in thresholds[col]:
                if key in groups:
                    minmax_dict = thresholds[col][key]
                    _validate_minmax(minmax_dict)
                else: #this is a minmax threshold
                    _validate_minmax(thresholds[col])

def _link_sliders(sliders):
    """ Link the values between interactive sliders.
    
    Parameters
    ------------
    sliders : 


    """
    
    tup = [(slider, 'value') for slider in sliders]
    
    l_list = []
    for i in range(1,len(tup)):
        l = ipywidgets.link(*tup[i-1:i+1])
        l_list.append(l)
    
    return l_list

def _toggle_linkage(checkbox, l_list):
    """ Either link or unlink sliders depending on the new value of the checkbox.
    
    Parameters
    -----------
    checkbox: 
    
    l_list : ..     
    
    """
    
    check_bool = checkbox["new"]
    
    if check_bool == True:
        for l in l_list:
            l.link()
    elif check_bool == False:
        for l in l_list:
            l.unlink()
    
def _update_thresholds(slider, fig, min_line, min_shade, max_line, max_shade):
    """ Update the locations of thresholds in plot """
    
    tmin, tmax = slider["new"] #threshold values from slider
    
    #Update min line
    ydata = min_line.get_ydata()
    ydata = [tmin for _ in ydata]
    min_line.set_ydata(ydata)
    
    x,y = min_shade.get_xy()
    min_shade.set_height(tmin - y)
        
    #Update max line
    ydata = max_line.get_ydata()
    ydata = [tmax for _ in ydata]
    max_line.set_ydata(ydata)
    
    x,y = max_shade.get_xy()
    max_shade.set_height(tmax - y)
    
    #Draw figure after update
    fig.canvas.draw_idle()
    
    #Save figure
    #sctoolbox.utilities.save_figure(save)

    
def quality_violin(adata, columns, 
                                which="obs", 
                                groupby=None, 
                                ncols=2, 
                                header=None, 
                                color_list=None, 
                                title=None, 
                                thresholds=None,
                                sliders=True,
                                save=None):
    """
    A function to plot quality measurements for cells in an anndata object.

    Note
    ------
    Notebook needs to start with "%matplotlib widget" for the interactive sliders to work.

    Parameters
    -------------
    adata : anndata.AnnData
        Anndata object containing quality measures in .obs/.var
    columns : list
        A list of columns in .obs/.var to show measures for.
    which : str, optional
        Which table to show quality for. Either "obs" / "var". Default: "obs".
    groupby : str, optional
        A column in table to values on the x-axis, e.g. 'condition'. 
    ncols : int
        Number of columns in the plot. Default: 2.
    header : list, optional
        A list of custom headers for each measure given in columns. Default: None (headers are the column names)
    color_list : list, optional
        A list of colors to use for violins. Default: None (colors are chosen automatically)
    title : str, optional
        The title of the full plot. Default: None (no title).
    thresholds : dict, optional
        Dictionary containing initial min/max thresholds to show in plot.
    sliders : bool, optional
        Whether to show sliders for thresholding. Default: True.
    save : str, optional
        Save the figure to the path given in 'save'. Default: None (figure is not saved).
    """

    #---------------- Test input and get ready --------------#

    ncols = min(ncols, len(columns)) #Make sure ncols is not larger than the number of columns
    nrows = int(np.ceil(len(columns) / ncols))

    #Decide which table to use
    if which == "obs":
        table = adata.obs
    elif which == "var":
        table = adata.var
    else:
        raise ValueError()

    #Order of categories on x axis
    if groupby is not None:
        groups = table[groupby].cat.categories
        n_colors = len(groups)
    else:
        groups = None
        n_colors = 1

    #Setup colors to be used
    if color_list is None:
        color_list = sns.color_palette("Set1", n_colors)
    else:
        if int(n_colors) <= int(len(color_list)):
            raise ValueError("Increase the color_list variable to at least {} colors.".format(n_colors))
        else:
            color_list = color_list[:n_colors]

    #Setup headers to be used
    if header is None:
        header = columns
    else:
        #check that header has the right length
        if len(header) != len(columns):
            raise ValueError("Length of header does not match length of columns")
    
    #Setup thresholds if not given
    if thresholds is None:
        thresholds = {col: {} for col in columns}


    #---------------- Setup figure --------------#

    #Setting up output figure
    fig, axarr = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    axes_list = axarr.flatten()

    #Remove empty axes
    for ax in axes_list[len(columns):]:
        ax.axis('off')

    #Add title of full plot
    if title != None:
        fig.suptitle(title, fontsize=16)

    #Add title of individual plots
    for i in range(len(columns)):
        ax = axes_list[i]
        ax.set_title(header[i], fontsize=11)
  

    #------------- Plot data and add sliders ---------#

    #Plotting data
    slider_dict = {}
    accordion_content = []
    for i, column in enumerate(columns):
        ax = axes_list[i]
        slider_dict[column] = {}

        #Plot data from table
        sns.violinplot(data=table, x=groupby, y=column, ax=ax, order=groups, palette=color_list)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_ylabel("")
        ax.set_xlabel("")    

        ticks = ax.get_xticks()
        ymin, ymax = ax.get_ylim() #ylim before plotting any thresholds

        #Establish groups
        if groupby is not None:
            group_names = groups
        else:
            group_names = ["Threshold"] 

        #Plot thresholds per group
        data_min = table[column].min()
        data_max = table[column].max()  
        slider_list = []
        for j, group in enumerate(group_names):
            
            #Establish the threshold to plot
            if column not in thresholds: #no thresholds given
                tmin = data_min
                tmax = data_max
            elif group in thresholds[column]: #thresholds per group
                tmin = thresholds[column][group].get("min", data_min)
                tmax = thresholds[column][group].get("max", data_max)
            else:
                tmin = thresholds[column].get("min", data_min)
                tmax = thresholds[column].get("max", data_max)

            #Plot line and shading
            tick = ticks[j]
            x = [tick-0.5, tick+0.5]

            min_line = ax.plot(x, [tmin]*2, color="red", linestyle="--")[0]
            max_line = ax.plot(x, [tmax]*2, color="red", linestyle="--")[0]

            y1, y2, y3, y4 = sorted([ymin, tmin, tmax, ymax]) #sorted positions of axes and lines from small to large
            min_shade = ax.add_patch(Rectangle((x[0], y1), x[1]-x[0], y2-y1, color="grey", alpha=0.2, linewidth=0))
            max_shade = ax.add_patch(Rectangle((x[0], y4), x[1]-x[0], y3-y4, color="grey", alpha=0.2, linewidth=0)) #negative height

            #Add slider to control thresholds
            slider = ipywidgets.FloatRangeSlider(description=group, min=data_min, max=data_max, 
                                                value=[tmin, tmax],  #initial value
                                                continuous_update=False)

            slider.observe(functools.partial(_update_thresholds, fig=fig, min_line=min_line, 
                                                                    min_shade=min_shade,
                                                                    max_line=max_line,
                                                                    max_shade=max_shade), 
                                                                    names=["value"])

            slider_list.append(slider)
            if groupby is not None:
                slider_dict[column][group] = slider
            else:
                slider_dict[column] = slider

        #Link sliders together
        if len(slider_list) > 0:
            link_list = _link_sliders(slider_list)
        
            #Toggle linked sliders
            c = ipywidgets.Checkbox(value=True, description='Global threshold', disabled=False, indent=False)
            c.observe(functools.partial(_toggle_linkage, l_list=link_list), names=["value"])

            box = ipywidgets.VBox([c] + slider_list)
        
        else:
            box = ipywidgets.Vbox(slider_list)
        
        accordion_content.append(box)

    fig.tight_layout()
    sctoolbox.utilities.save_figure(save) #save plot; can be overwritten if thresholds are changed

    #Assemble accordion with different measures
    accordion = ipywidgets.Accordion(children=accordion_content, selected_index=None)
    for i in range(len(columns)):
        accordion.set_title(i, columns[i])

    #Setup box to hold all widgets
    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = False
    fig.canvas.resizable = True
    fig.canvas.width = "auto"

    if sliders == True:
        full_box = ipywidgets.HBox([accordion, fig.canvas])
    else:
        full_box = fig.canvas #ipywidgets.HBox([fig.canvas])

    return full_box, slider_dict

def get_slider_thresholds(slider_dict):
    """ Get thresholds from sliders.
    
    Parameters
    ----------
    slider_dict : dict
        Dictionary of sliders in the format 'slider_dict[column][group] = slider' or 'slider_dict[column] = slider' if no grouping.
    
    Returns
    -------
    dict in the format threshold_dict[column][group] = {"min": <min_threshold>, "max": <max_threshold>} or 
    threshold_dict[column] = {"min": <min_threshold>, "max": <max_threshold>} if no grouping

    """
    threshold_dict = {}
    for measure in slider_dict:
        threshold_dict[measure] = {}
        
        if isinstance(slider_dict[measure], dict): #thresholds for groups
            for group in slider_dict[measure]:
                slider = slider_dict[measure][group]
                threshold_dict[measure][group] = {"min": slider.value[0], "max": slider.value[1]}

            #Check if all groups have the same thresholds
            mins = set([d["min"] for d in threshold_dict[measure].values()])
            maxs = set([d["max"] for d in threshold_dict[measure].values()])

            #Set overall threshold if individual sliders are similar
            if len(mins) == 1 and len(maxs) == 1:
                threshold_dict[measure] = threshold_dict[measure][group] #takes the last group from the previous for loop

        else: #One threshold for measure
            slider = slider_dict[measure]
            threshold_dict[measure] = {"min": slider.value[0], "max": slider.value[1]}

    return threshold_dict



def apply_qc_thresholds(adata, thresholds, which="obs", groupby=None):
    """
    Apply QC thresholds to anndata object.

    Parameters
    -------------
    adata : AnnData
        Anndata object to filter.
    thresholds : dict
        Dictionary of thresholds to apply.
    which : str
       Which table to filter on. Must be one of "obs" / "var". Default: "obs".
    groupby : str
        Column in table to group by. Default: None.

    Returns
    ---------
    adata : AnnData
        Anndata object with QC thresholds applied.
    """

    table = adata.obs if which == "obs" else adata.var

    #Create copy of adata
    adata = adata.copy()

    #Check if all columns are found in adata
    not_found = list(set(thresholds) - set(table.columns))
    if len(not_found) > 0:
        print("{0} threshold columns were not found in adata and could therefore not be applied. These columns are: {1}".format(len(not_found), not_found))
    thresholds = {k: thresholds[k] for k in thresholds if k not in not_found}
    
    if len(thresholds) == 0:
        raise ValueError(f"The thresholds given do not match the columns given in adata.{which}. Please adjust the 'which' parameter if needed.")

    if groupby is not None:
        groups = table[groupby].cat.categories

    #Check that thresholds contain min/max
    for column, d in thresholds.items():
        if 'min' not in d and 'max' not in d:
            if groupby is not None:
                keys = d.keys()
                not_found = list(set(keys) - set(groups))
                if not_found > 0:
                    raise ValueError(f"{len(not_found)} groups from thresholds were not found in adata.obs[{groupby}]. These groups are: {not_found}")

            else:
                raise ValueError("Error in threshold format: Thresholds must contain min or max per column, or a threshold per group in groupby")

    #Apply thresholds
    for column, d in thresholds.items(): 

        #Update size of table
        table = adata.obs if which == "obs" else adata.var

        #Collect boolean array of rows to select of table
        excluded = np.array([False] * len(table))
        if groupby is not None:
            if "min" in d or "max" in d:
                global_threshold = True

            else:
                for group in d:
                    minmax_dict = d[group]

                    group_bool = table[groupby] == group

                    if "min" in minmax_dict:
                        excluded = excluded | (group_bool & (table[column] < minmax_dict["min"]))

                    if "max" in minmax_dict:
                        excluded = excluded | (group_bool & (table[column] > minmax_dict["max"]))
        else:
            global_threshold = True

        #Select using a global threshold
        if global_threshold == True:
            minmax_dict = d

            if "min" in minmax_dict:
                excluded = excluded | (table[column] < minmax_dict["min"]) #if already excluded, or if excluded by min

            if "max" in minmax_dict:
                excluded = excluded | (table[column] > minmax_dict["max"])

        #Apply filtering
        included = ~excluded
        if which == "obs":
            adata = adata[included,:]
            name = "cells"
        else: 
            adata = adata[:,included] #filter on var
            name = ".var features"
        print(f"Filtering based on '{column}' from {len(table)} -> {sum(included)} {name}")

    return adata
