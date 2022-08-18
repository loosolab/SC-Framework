import pandas as pd
from sctoolbox import plotting, checker, analyser, utilities
import scanpy as sc
import numpy as np
import os

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
    if var_color_by is not None and var_color_by not in anndata.var.columns:
        raise ValueError("Couldn't find value of var_color_by in anndata.var column names.")

    if obs_color_by is not None and obs_color_by not in anndata.obs.columns:
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
    thresholds : pandas.DataFrame
        A dataframe with the default cutoffs produced by the function `find_thresholds`.
    inplace : bool, default False
        Adjust the thresholds dataframe inplace.

    Returns
    -------
    pandas.DataFrame or None :
        Pandas dataframe to be used for the QC and filtering steps
    """
    utilities.check_module("click")
    import click

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
        except Exception:
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
                utilities.check_module("IPython")
                from IPython.display import display

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


def filter_threshold(anndata, on, min, max, inplace=False):
    """
    Filter anndata.obs or anndata.var column to given range.

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object to filter.
    on : str
        Anndata.obs or anndata.var column to apply filter to.
    min : float
        Minimum allowed value. Set None for no threshold.
    max : float
        Maximum allowed value. Set None for no threshold.
    inplace : bool, default False
        If anndata should be modified inplace.

    Returns
    -------
    anndata.AnnData or None :
        Filtered anndata object.
    """
    # check if filter column exists
    if on not in anndata.obs.columns and on not in anndata.var.columns:
        raise ValueError(f"Invalid parameter on=`{on}`. Neither found as anndata.obs nor anndata.var column name.")

    # find out where the column is
    col_in = "obs" if on in anndata.obs.columns else "var"
    table = anndata.obs if col_in == "obs" else anndata.var

    # don't filter if min or max is None
    min = np.min(table[on]) if min is None else min
    max = np.max(table[on]) if max is None else max

    # list of entries to keep
    filter_bool = (table[on] >= min) & (table[on] <= max)

    # TODO add logging (add info to anndata.uns["infoprocess"])

    if inplace:
        # NOTE: these are privat anndata functions so they might change without warning!
        if "obs":
            anndata._inplace_subset_obs(filter_bool)
        else:
            anndata._inplace_subset_var(filter_bool)
    else:
        if "obs":
            return anndata[filter_bool]
        else:
            return anndata[:, filter_bool]


def anndata_filter(anndata, thresholds, inplace=False):
    """
    Filter anndata based on provided threshold table.

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object to filter.
    thresholds : pandas.DataFrame
        Pandas dataframe with filter thresholds for anndata.var & anndata.obs columns. Produced by `find_thresholds()`.
        Structure (columns):
            index     = anndata.obs & anndata.var column names
            threshold = number of list of numbers with thresholds
            color_by  = corresponding to index .obs or .var column name with categories to split the data. (Not used)
    inplace : bool, default False
        If anndata should be modified inplace.

    Returns
    -------
    anndata.AnnData or None :
        Filtered anndata object.
        TODO Annotation of filtering parameters in the anndata.uns["infoprocess"]
    """
    if not inplace:
        anndata = anndata.copy()

    # iterate over thresholds and filter anndata object
    for index, (threshold, _) in thresholds.iterrows():
        if threshold:
            filter_threshold(anndata=anndata, on=index, min=threshold[0], max=threshold[1], inplace=True)

    if not inplace:
        return anndata

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
