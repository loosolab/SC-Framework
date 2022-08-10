import sys
from tabnanny import check
from scipy.stats import skew, kurtosis
import pandas as pd
from sctoolbox import plotting, creators, checker, analyser, utilities
import scanpy as sc
from IPython.display import display
import numpy as np

###############################################################################
#                      STEP 1: DEFINING DEFAULT CUTOFFS                       #
###############################################################################


def loop_question(answer, quit_m, check):
    """
    This core check the validity of user's answer.

    Parameters
    ----------
    answer : str
        User's answer
    quit_m : str
        Question to be print if the user's answer is invalid
    check : str
        Types of questions. The options are "yn", "custdef", or float

    Returns
    -------
    str :
        The user's answer in lower case format
    """
    # List and others
    opt1, opt2, opt3, opt4 = ["q", "quit"], ["y", "yes", "n", "no"], ["custom", "def"], ["custom", "def", "skip"]  # Lists with possible answers for each question
    options = {"yn": opt2, "custdef": opt3, "custdefskip": opt4}

    if checker.check_options(check, list(options.keys())) is False and type(check) != float:
        sys.exit("Insert a valid check: " + str(list(options.keys())) + " or float.")

    answer = input(answer).lower()
    # Check validity of options
    while checker.check_options(answer, options=opt1 + opt2 + opt3 + opt4) is False and utilities.is_str_numeric(answer) is False:
        print("Choose one of these options: " + str(opt1 + opt2 + opt3 + opt4))
        answer = input(answer).lower()

    # In case the input is valid, goes futher
    if type(check) != float:
        while checker.check_options(answer, options=opt1 + options[check]) is False:
            print("Invalid choice! " + quit_m)
            answer = input()
            checker.check_options(answer, opt1 + options[check])
    else:
        # Checking validity of custom value: >=0 and <= a limit (the CHECK)
        while checker.check_cuts(answer, 0, check) is False:
            print("Invalid choice! " + quit_m)
            answer = input()
            checker.check_cuts(answer, 0, check)

    return answer.lower()


def set_def_cuts(anndata, interval, var="all", obs="all", only_plot=False, file_name="note2_violin_", save=False):
    """
    Find thresholds for the given .obs and .var columns in anndata.

    1- Definining the cutoffs for QC and filtering steps.
    2- Ploting anndata obs or anndata var selected for the QC and filtering steps with the cutoffs or not.

    Parameters
    ----------
    anndata : anndata.AnnData
        anndata object
    interval : int or float
        The percentage (from 0 to 100) to be used to calculate the cutoffs.
    var : str or list of str, default 'all'
        Anndata.var columns to find thresholds for. If 'all' will select all numeric columns. Use None to disable.
    obs : str or list of str, default 'all'
        Anndata.obs columns to find thresholds for. If 'all' will select all numeric columns. Use None to disable.
    only_plot : bool, default False
        If true, only a plot with the data without cutoff lines will be provided.
    file_name : str, default "note2_violin_"
        Define a name for save a custom filename to save.
        NOTE: use a syntax at least composing the "note2_". Do not add any file extension.
    save : bool, default False
        True, save the figure to the path given in 'save'.

    Notes
    -----
    Author: Guilherme Valente

    Returns
    -------
    pandas.DataFrame or None:
        A pandas dataframe with the defined cutoff parameters. None for only_plot=True.
    """
    # -------------------- checks & setup ------------------- #
    # is interval valid?
    if not checker.in_range(interval, (0, 100)):
        raise ValueError(f"Parameter interval is {interval} but has to be in range 0 - 100!")

    # anything selected?
    if var is None and obs is None:
        raise ValueError("Parameters var & obs are empty. Set at least one of them.")

    # expand 'all'; get all numeric columns
    if var == "all":
        var = list(anndata.var.select_dtypes(np.number).columns)
    if obs == "all":
        obs = list(anndata.obs.select_dtypes(np.number).columns)

    # make iterable
    obs = obs if isinstance(obs, list) else [obs]
    var = var if isinstance(var, list) else [var]
    
    # invalid column name?
    invalid_obs = set(obs) - set(anndata.obs.columns)
    invalid_var = set(var) - set(anndata.var.columns)
    if invalid_obs or invalid_var:
        raise ValueError(f"""
                         Invalid names in obs and/ or var detected:
                         obs: {invalid_obs}
                         var: {invalid_var}
                         """)

    # columns numeric?
    not_num_obs = set(obs) - set(anndata.obs.select_dtypes(np.number).columns)
    not_num_var = set(var) - set(anndata.var.select_dtype(np.number).columns)
    if not_num_obs or not_num_var:
        raise ValueError(f"""
                         Selected columns have to be numeric. Not numeric column(s) received:
                         obs: {not_num_obs}
                         var: {not_num_var}
                         """)

    # List, dfs, messages and others
    uns_condition_name = anndata.uns["infoprocess"]["data_to_evaluate"]  # The name of data to evaluate parameter, e.g., "condition"
    for_cells = [uns_condition_name, "n_genes_by_counts", "total_counts", "pct_counts_is_mitochondrial"]  # List of obs variables to be analysed. The first item MUST be the data to be evaluated
    for_genes = ["n_cells_by_counts", "mean_counts", "pct_dropout_by_counts"]  # List of var variables to be analysed.
    for_cells_pd = anndata.obs[anndata.obs.columns.intersection(for_cells)]
    for_genes_pd = anndata.var[anndata.var.columns.intersection(for_genes)]
    cells_genes_list = for_cells[1:] + for_genes
    calulate_and_plot_filter = []  # Samples to be calculated cutoffs and further plotted with cutoff lines
    df_cuts = pd.DataFrame(columns=["data_to_evaluate", "parameters", "cutoff", "strategy"])  # Empty df to be filled with the cutoffs
    anndata.uns[uns_condition_name + '_colors'] = anndata.uns["color_set"][:len(anndata.obs[uns_condition_name].unique())]
    sta_cut_cells, sta_cut_genes = None, None

    # Setting colors for plot
    m1 = "file_name[STRING]"
    m4 = "Defining and ploting cutoffs only for total_counts"
    m5 = "You choose not plot cutoffs"
    pathway = anndata.uns["infoprocess"]["Anndata_path"]

    # Creating filenames
    if save is True and file_name != "note2_violin_" and type(file_name) != str:  # Here checking is custom name is proper
        sys.exist(m1)
    elif save is True and type(file_name) == str:  # Custom name is proper. If custom is not provided, the default is used
        filename = pathway + file_name
        filename = filename.replace("//", "/")
    elif save is False:
        filename = ""

    if not only_plot:
        # Checking if the total count was already filtered
        act_c_total_counts, id_c_total_counts = float(anndata.obs["total_counts"].sum()), float(anndata.uns["infoprocess"]["ID_c_total_counts"])
        if checker.check_cuts(str(act_c_total_counts), id_c_total_counts, id_c_total_counts) is True:  # Total_counts was not filtered yet, then only this parameter will be evaluated
            sta_cut_cells, sta_cut_genes = True, False
            calulate_and_plot_filter.append("total_counts")
            filename = filename + "tot_count_cut"
            print(m4)

        # Other parameters will be evaluated because plot was selected and total count is filtered
        elif checker.check_cuts(str(act_c_total_counts), id_c_total_counts, id_c_total_counts) is False:  # Total counts was filtered yet.
            sta_cut_cells, sta_cut_genes = True, True
            filename = filename + "other_param_cut"
            for a in cells_genes_list:
                if a != "total_counts":
                    calulate_and_plot_filter.append(a)

    # Ploting with or without cutoffs
    if only_plot is True:  # Only plots without cutoff lines will be provided
        filename = filename

        # Building the dataframe with default cutoffs stablished and the plots
        # Here will be called the function establishing_cuts from the analyser.py module
        print(m5)
        return plotting.qcf_ploting(for_cells_pd, for_genes_pd, anndata.uns[for_cells[0] + "_colors"], df_cuts, PLOT=None, SAVE=save, FILENAME=filename)

    elif only_plot is False:  # Calculate cutoffs and plot in the violins
        if sta_cut_cells is True:  # For cells
            for a in for_cells_pd[for_cells[0]].unique().tolist():  # Getting the conditions.
                adata_cond = for_cells_pd[for_cells_pd[for_cells[0]] == a].copy()
                for b in for_cells[1:]:
                    if b in calulate_and_plot_filter:
                        data = adata_cond[b]
                        skew_val, kurtosis_val = round(skew(data.to_numpy(), bias=False), 1), int(kurtosis(data.to_numpy(), fisher=False, bias=False))  # Calculating the skewness and kurtosis
                        kurtosis_val_norm = int(kurtosis_val - 3)  # This is a normalization for kurtosis value to identify the excessive kurtosis. Cite: https://www.sciencedirect.com/topics/mathematics/kurtosis
                        df_cuts = analyser.establishing_cuts(data, interval, skew_val, kurtosis_val_norm, df_cuts, b, a)

        if sta_cut_genes is True:  # For genes
            for a in for_genes:
                if a in calulate_and_plot_filter:  # Calculate cutoffs only for selected parameters
                    data = for_genes_pd[a]
                    skew_val, kurtosis_val = round(skew(data.to_numpy(), bias=False), 1), int(kurtosis(data.to_numpy(), fisher=False, bias=False))  # Calculating the skewness and kurtosis
                    kurtosis_val_norm = int(kurtosis_val - 3)  # This is a normalization for kurtosis value to identify the excessive kurtosis. Cite: https://www.sciencedirect.com/topics/mathematics/kurtosis
                    df_cuts = analyser.establishing_cuts(data, interval, skew_val, kurtosis_val_norm, df_cuts, a, None)

        plotting.qcf_ploting(for_cells_pd, for_genes_pd, anndata.uns[for_cells[0] + "_colors"], df_cuts, PLOT=calulate_and_plot_filter, SAVE=save, FILENAME=filename)

        return df_cuts

######################################################################################
#                         STEP 2: DEFINING CUSTOM CUTOFFS                            #
######################################################################################


def refining_cuts(anndata, def_cut2):
    """
    Refining the cutoffs to be used in the QC and filtering steps

    Parameters
    ----------
    anndata : anndata.AnnData
        anndata object
    def_cut2 : pandas.DataFrame
        A dataframe with the default cutoffs calculated by the function set_def_cuts of qc_filter.py module

    Notes
    -----
    Author: Guilherme Valente

    Returns
    -------
    pandas.DataFrame :
        Pandas dataframe to be used for the QC and filtering steps
    """
    # Dataframes and others
    dfcut = def_cut2.copy()
    new_df = dfcut[0:0]
    col0 = dfcut.columns[0]  # data_to_evaluate
    col1 = dfcut.columns[1]  # parameters
    col2 = dfcut.columns[2]  # cutoffs
    col3 = dfcut.columns[3]  # strategy
    cond_name = anndata.uns["infoprocess"]["data_to_evaluate"]

    # Questions, and messages
    q2 = ": custom or def"
    q3 = ": minimun cutoff [FLOAT or INT]"
    q4 = ": maximum cutoff [FLOAT or INT]"
    q5 = ": custom, def or skip"
    m2 = "The default setting is:"
    m3 = "\n\n#########################################################\nNOTE. Choose: \n\t1) custom to define new cutoffs\n\t2) def to use the previously defined cutoffs\n#########################################################\n"
    m4 = "\n\nChoose custom or def\n"
    m5 = "\n\nType a positive number "
    m6 = "<= "
    m7 = "\n"
    m8 = "\n\n#########################################################\nNOTE. Choose: \n\t1) custom to define new cutoffs\n\t2) def to use the previously defined cutoffs\n\t3) skip to avoid filter this parameter\n#########################################################\n"
    m9 = "\n\nChoose custom, def or skip\n"

    # Checking if the total_counts was already filtered
    if anndata.obs["total_counts"].sum() == anndata.uns["infoprocess"]["ID_c_total_counts"]:  # If False, means that the total_counts was not filtered yet
        df_tot_coun = dfcut[dfcut[col1] == "total_counts"].reset_index(drop=True)
        is_total_count_filt = False
    else:
        df_NOT_tot_coun = dfcut[dfcut[col1] != "total_counts"].reset_index(drop=True)
        is_total_count_filt = True

    # Executing the main function
    if is_total_count_filt is False:  # It means that total_counts was not filtered. Then, the dataframe has only total_count information
        print(m2)
        display(df_tot_coun)
        print(m3)
        for idx, a in dfcut.iterrows():
            data, param, list_cuts, stra = a[col0], a[col1], a[col2], a[col3]
            answer = loop_question(data + " " + param + q2, m4, "custdef")  # Custom or default?
            if answer == "def":
                new_df = new_df.append({col0: data, col1: param, col2: list_cuts, col3: stra}, ignore_index=True)
            elif answer == "custom":
                max_lim = max(anndata.obs[anndata.obs[cond_name] == data][param])
                min_cut = loop_question(data + " " + param + q3, m5 + m6 + " " + str(max_lim) + m7, float(max_lim))
                max_cut = loop_question(data + " " + param + q4, m5 + m6 + " " + str(max_lim) + m7, float(max_lim))
                list_cuts = sorted([float(min_cut), float(max_cut)], reverse=True)
                new_df = new_df.append({col0: data, col1: param, col2: list_cuts, col3: stra}, ignore_index=True)

    else:  # It means that total_counts was filtered. Then, the dataframe only has parameters for other information
        print(m2)
        display(df_NOT_tot_coun)
        print(m8)
        for idx, a in dfcut.iterrows():
            data, param, list_cuts, stra = a[col0], a[col1], a[col2], a[col3]
            answer = loop_question(data + " " + param + q5, m9, "custdefskip")  # Custom, default or skip?
            if answer == "def":
                new_df = new_df.append({col0: data, col1: param, col2: list_cuts, col3: stra}, ignore_index=True)
            elif answer == "custom":
                if stra == "filter_cells":
                    max_lim = max(anndata.obs[anndata.obs[cond_name] == data][param])
                elif stra == "filter_genes":
                    max_lim = max(anndata.var[param])
                min_cut = loop_question(data + " " + param + q3, m5 + m6 + " " + str(max_lim) + m7, float(max_lim))
                max_cut = loop_question(data + " " + param + q4, m5 + m6 + " " + str(max_lim) + m7, float(max_lim))
                list_cuts = sorted([float(min_cut), float(max_cut)], reverse=True)
                new_df = new_df.append({col0: data, col1: param, col2: list_cuts, col3: stra}, ignore_index=True)
            elif answer == "skip":
                new_df = new_df.append({col0: data, col1: param, col2: "skip", col3: stra}, ignore_index=True)

    return new_df

###############################################################################
#                           STEP 3: APPLYING CUTOFFS                          #
###############################################################################


def anndata_filter(anndata, go_cut):
    """
    TODO

    Parameters
    ----------
    anndata : anndata.AnnData
        anndata object
    go_cut : pandas.DataFrame
        Pandas dataframe with the samples, parameters, and cutoffs to be used in the filtering process

    Notes
    -----
    Author: Guilherme Valente

    Returns
    -------
    anndata.AnnData :
        Filtered anndata
        Annotation of filtering parameters in the anndata.uns["infoprocess"]
    """
    def concatenating(LISTA_ADATA):  # Concatenating adata
        """ TODO """
        if len(LISTA_ADATA) > 1:
            adata_conc = LISTA_ADATA[0].concatenate(LISTA_ADATA[1:], join='inner', batch_key=None)
        else:
            adata_conc = LISTA_ADATA[0]
        return adata_conc

    # List, messages and others
    anndata_CP = anndata.copy()
    anndata_CP2 = anndata_CP.copy()
    datamcol, paramcol, cutofcol, stratcol = go_cut.columns[0], go_cut.columns[1], go_cut.columns[2], go_cut.columns[3]
    act_params = go_cut[paramcol].unique().tolist()
    act_strate = go_cut[stratcol].unique().tolist()
    uns_cond = anndata.uns["infoprocess"]["data_to_evaluate"]
    lst_adata_sub, go_info_cell, go_info_genes = list(), [], []

    # Creating the infoprocess
    if "Cell filter" not in anndata_CP.uns["infoprocess"]:
        creators.build_infor(anndata_CP, "Cell filter", list())
    if "Gene filter" not in anndata_CP.uns["infoprocess"]:
        creators.build_infor(anndata_CP, "Gene filter", list())

    # Filter mitochondrial content first
    if "pct_counts_is_mitochondrial" in act_params:
        raw_data = go_cut[go_cut[paramcol] == "pct_counts_is_mitochondrial"]
        for idx, a in raw_data.iterrows():
            data, param, cuts, _ = a[datamcol], a[paramcol], a[cutofcol], a[stratcol]
            if "skip" in cuts:
                lst_adata_sub.append(anndata_CP[anndata_CP.obs[uns_cond] == data, :])
            else:
                max_cut = max(cuts)
                m1 = data + " " + param + " max_percent= " + str(max_cut)
                adata_sub = anndata_CP[anndata_CP.obs[uns_cond] == data, :]
                lst_adata_sub.append(adata_sub[adata_sub.obs["pct_counts_is_mitochondrial"] < max_cut, :])
                go_info_cell.append(m1)
        anndata_CP2 = concatenating(lst_adata_sub)
        lst_adata_sub = list()
    else:
        anndata_CP2 = anndata_CP
        lst_adata_sub = list()

    # Filtering other cells
    raw_data = go_cut[go_cut[stratcol] == "filter_cells"]
    for idx, a in raw_data.iterrows():
        data, param, cuts, _ = a[datamcol], a[paramcol], a[cutofcol], a[stratcol]
        if param != "pct_counts_is_mitochondrial":
            adata_sub = anndata_CP2[anndata_CP2.obs[uns_cond] == data, :]
            if "skip" in cuts:
                lst_adata_sub.append(adata_sub)
            else:
                m1 = data + " " + param + " "
                min_cut, max_cut = min(cuts), max(cuts)
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

    anndata_CP2 = concatenating(lst_adata_sub)
    lst_adata_sub = list()

    # Filtering genes
    if "filter_genes" in act_strate:
        raw_data = go_cut[go_cut[stratcol] == "filter_genes"]
        adata_sub = anndata_CP2.copy()
        for idx, a in raw_data.iterrows():
            data, param, cuts, _ = a[datamcol], a[paramcol], a[cutofcol], a[stratcol]
            if "skip" in cuts:
                pass
            else:
                min_cut, max_cut = min(cuts), max(cuts)
                if param == "n_cells_by_counts":
                    sc.pp.filter_genes(adata_sub, min_cells=min_cut, inplace=True)
                    go_info_genes.append("n_cells_by_counts min_cells= " + str(min_cut))
                elif param == "mean_counts":
                    sc.pp.filter_genes(adata_sub, min_counts=min_cut, inplace=True)
                    go_info_genes.append("mean_counts min_counts= " + str(min_cut))
                elif param == "pct_dropout_by_counts":
                    print("pct_dropout_by_counts filtering is not implemented.")
        anndata_CP2 = adata_sub

    # Annotating anndata.uns["infoprocess"]
    if len(anndata_CP.uns["infoprocess"]["Cell filter"]) > 0:
        anndata_CP.uns["infoprocess"]["Cell filter"] = anndata_CP.uns["infoprocess"]["Cell filter"] + go_info_cell
    else:
        anndata_CP.uns["infoprocess"]["Cell filter"] = go_info_cell
    if len(anndata_CP.uns["infoprocess"]["Gene filter"]) > 0:
        anndata_CP.uns["infoprocess"]["Gene filter"] = anndata_CP.uns["infoprocess"]["Gene filter"] + go_info_genes
    else:
        anndata_CP.uns["infoprocess"]["Gene filter"] = go_info_genes

    anndata_CP2.uns = anndata_CP.uns

    print(anndata)
    print(anndata_CP2)
    return anndata_CP2.copy()

#####################################################################################
#####################################################################################


def filter_genes(adata, genes):
    """
    Remove genes from adata object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object to filter
    genes : list of str
        A list of genes to remove from object.

    Returns
    -------
    anndata.AnnData :
        Anndata object with removed genes.
    """
    # Check if all genes are found in adata
    not_found = list(set(genes) - set(adata.var_names))
    if len(not_found) > 0:
        print("{0} genes were not found in adata and could therefore not be removed. These genes are: {1}".format(len(not_found), not_found))

    # Remove genes from adata
    n_before = adata.shape[1]
    adata = adata[:, ~adata.var_names.isin(genes)]
    n_after = adata.shape[1]
    print("Filtered out {0} genes from adata. New number of genes is: {1}.".format(n_before - n_after, n_after))

    return adata


def estimate_doublets(adata, threshold=0.25, inplace=True, **kwargs):
    """
    Estimate doublet cells using scrublet. Adds additional columns "doublet_score" and "predicted_doublet" in adata.obs,
    as well as a "scrublet" key in adata.uns.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object to estimate doublets for.
    threshold : float, default 0.25
        Threshold for doublet detection.
    inplace : bool, default True
        Whether to estimate doublets inplace or not.
    **kwargs :
        Additional arguments are passed to scanpy.external.pp.scrublet.

    Returns
    -------
    anndata.Anndata or None :
        If inplace is False, the function returns a copy of the adata object.
        If inplace is True, the function returns None.
    """
    if inplace is False:
        adata = adata.copy()

    # Run scrublet on adata
    adata_scrublet = sc.external.pp.scrublet(adata, threshold=threshold, copy=True, **kwargs)

    # Plot the distribution of scrublet scores
    sc.external.pl.scrublet_score_distribution(adata_scrublet)

    # Save scores to object
    adata.obs["doublet_score"] = adata_scrublet.obs["doublet_score"]
    adata.obs["predicted_doublet"] = adata_scrublet.obs["predicted_doublet"]
    adata.uns["scrublet"] = adata_scrublet.uns["scrublet"]

    if inplace is False:
        return adata
