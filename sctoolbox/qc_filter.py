import pandas as pd
from sctoolbox import plotting, creators, checker, analyser, utilities
import scanpy as sc
from IPython.display import display
import numpy as np
import os
import click

###############################################################################
#                      STEP 1: DEFINING DEFAULT CUTOFFS                       #
###############################################################################


def find_thresholds(anndata, interval, var="all", obs="all", var_color_by=None, obs_color_by=None, show_thresholds=True, output=None):
    """
    Find thresholds for the given .obs (cell) and .var (gene) columns in anndata.

    Parameters
    ----------
    anndata : anndata.AnnData
        anndata object
    interval : int or float
        The percentage (from 0 to 100) to be used to calculate the cutoffs.
    var : str or list of str, default 'all'
        Anndata.var (gene) columns to find thresholds for. If 'all' will select all numeric columns. Use None to disable.
    obs : str or list of str, default 'all'
        Anndata.obs (cell) columns to find thresholds for. If 'all' will select all numeric columns. Use None to disable.
    var_color_by : str, default None
        Split anndata.var related violins into color groups using .var column of the given name.
    obs_color_by : str, default None
        Split anndata.obs related violins into color groups using .obs column of the given name.
    show_thresholds : bool, default True
        If true, compute thresholds and show threshold lines. Returned DataFrame won't contain thresholds if False.
    output : str or bool, default None
        Path + filename to save plot to. If True instead of str will save plot to "<project_folder>/qc_violin.pdf".

    Returns
    -------
    pandas.DataFrame or None:
        A pandas dataframe with the defined cutoff parameters. Won't contain thresholds for show_thresholds=False.
    """
    # -------------------- checks & setup ------------------- #
    # is interval valid?
    if not checker.in_range(interval, (0, 100)):
        raise ValueError(f"Parameter interval is {interval} but has to be in range 0 - 100!")

    # anything selected?
    if var is None and obs is None:
        raise ValueError("Parameters var & obs are empty. Set at least one of them.")

    # check valid obs & var color_by parameters
    if not var_color_by is None and not var_color_by in anndata.var.columns:
        raise ValueError("Couldn't find value of var_color_by in anndata.var column names.")

    if not obs_color_by is None and not obs_color_by in anndata.obs.columns:
        raise ValueError("Couldn't find value of obs_color_by in anndata.obs column names.")

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
    not_num_var = set(var) - set(anndata.var.select_dtypes(np.number).columns)
    if not_num_obs or not_num_var:
        raise ValueError(f"""
                         Selected columns have to be numeric. Not numeric column(s) received:
                         obs: {not_num_obs}
                         var: {not_num_var}
                         """)

    # TODO check if data was filtered before

    # Creating filenames
    if output is True:
        # TODO think of a better way instead of hardcoding this.
        output = os.path.join(anndata.uns["infoprocess"]["Anndata_path"], "qc_violin.pdf")

    # setup thresholds data frame
    thresholds = {'index': [], 'threshold': [], 'color_by': []}
    for column in obs + var:
        thresholds['index'].append(column)
        thresholds['threshold'].append(None)
        thresholds['color_by'].append(obs_color_by if column in anndata.obs.columns else var_color_by)

    thresholds = pd.DataFrame.from_dict(thresholds).set_index("index")

    # Plotting with or without cutoffs
    if show_thresholds:
        # Calculate cutoffs to plot in the violins
        for column in obs + var:
            # compute cutoffs for each column
            if column in anndata.obs.columns:
                data = anndata.obs[column]
            else:
                data = anndata.var[column]

            # compute threshold
            thresholds.at[column, "threshold"] = analyser.get_threshold(data=data.to_list(), interval=interval, limit_on="both")

    # create violinplot
    plotting.qc_violins(anndata, thresholds, colors=None, filename=output)

    # TODO return anndata containing threshold table instead
    return thresholds

######################################################################################
#                         STEP 2: DEFINING CUSTOM CUTOFFS                            #
######################################################################################


def refine_thresholds(thresholds, inplace=False):
    """
    Interactive method to refine the cutoffs used in the QC and filtering steps.

    Parameters
    ----------
    anndata : anndata.AnnData
        anndata object
    thresholds : pandas.DataFrame
        A dataframe with the default cutoffs produced by the function `find_thresholds`.
    inplace : bool, default False
        Adjust the thresholds dataframe inplace.

    Returns
    -------
    pandas.DataFrame or None :
        Pandas dataframe to be used for the QC and filtering steps
    """
    if not inplace:
        thresholds = thresholds.copy()

    # quit value
    quit = ["Quit"]

    # temporary reset index for easier handling
    index_name = thresholds.index.name
    thresholds.reset_index(inplace=True)

    def convert_type(val):
        """ Evaluate string if possible else return string. """
        try:
            return eval(val)
        except:
            return val

    # start interactive loop
    while True:
        # start menu
        print("""
              1. Add new threshold row
              2. Edit threshold row
              3. Remove threshold row
              4. Show table
              5. Quit
              """)
        selection = int(click.prompt("What do you want to do?", type=click.Choice(["1", "2", "3", "4", "5"]), show_choices=False))

        # add new
        if selection == 1:
            print("Create new row.")
            # create row
            new = {}
            for column in thresholds.columns:
                new[column] = click.prompt(f"{column}")

            # add row
            thresholds = pd.concat([thresholds, pd.DataFrame(new, index=[0])], ignore_index=True)

        # edit row
        elif selection == 2:
            print("Edit row.")

            options = list(thresholds[index_name])
            for row in options + quit:
                print(f"    - {row}")

            index = click.prompt("Select row to update", type=click.Choice(options + quit), show_choices=False)

            if index != quit[0]:
                for column in thresholds.columns:
                    # skip index column
                    if column == index_name:
                        continue

                    # update value
                    index_num = thresholds[thresholds[index_name] == index].index[0]
                    thresholds.at[index_num, column] = click.prompt(f"Update {column} leave empty to keep former value", default=thresholds.at[index_num, column], value_proc=convert_type)

        # remove row
        elif selection == 3:
            options = list(thresholds[index_name]) + quit

            # show options
            for opt in options:
                print(f"    - {opt}")

            selection = click.prompt("Select row to remove", type=click.Choice(options), show_choices=False)

            if selection != quit:
                index_num = thresholds[thresholds[index_name] == selection].index[0]

                # remove row
                thresholds.drop(index=index_num, inplace=True)

        # show table
        elif selection == 4:
            if utilities._is_notebook():
                display(thresholds)
            else:
                print(thresholds)

        # quit
        elif selection == 5:
            if click.confirm("Are you sure you want to quit?"):
                break

        if selection != 4:
            # clear output
            utilities.clear()

    # re-add index
    thresholds.set_index(index_name, inplace=True)

    return thresholds if not inplace else None

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
