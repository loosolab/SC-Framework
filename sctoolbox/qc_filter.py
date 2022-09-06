import os
import numpy as np
import pandas as pd
import functools  # for partial functions
import scanpy as sc

# for plotting
from matplotlib.patches import Rectangle

import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator

# toolbox functions
import sctoolbox
from sctoolbox import plotting, checker, analyser, utilities


###############################################################################
#                        PRE-CALCULATION OF QC METRICS                        #
###############################################################################

def estimate_doublets(adata, threshold=0.25, inplace=True, plot=True, batch_key=None, **kwargs):
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
    plot : bool, default True
        Whether to plot the doublet score distribution.
    batch_key : str, default None
        Key in adata.obs to use for batching during doublet estimation. This option is passed to scanpy.external.pp.scrublet.
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
    adata_scrublet = sc.external.pp.scrublet(adata, threshold=threshold, copy=True, batch_key=batch_key, **kwargs)

    # Plot the distribution of scrublet scores
    if plot is True:
        sc.external.pl.scrublet_score_distribution(adata_scrublet)

    # Save scores to object
    adata.obs["doublet_score"] = adata_scrublet.obs["doublet_score"]
    adata.obs["predicted_doublet"] = adata_scrublet.obs["predicted_doublet"]
    adata.uns["scrublet"] = adata_scrublet.uns["scrublet"]

    if inplace is False:
        return adata


###############################################################################
#                      STEP 1: FINDING AUTOMATIC CUTOFFS                      #
###############################################################################

def get_thresholds(data,
                   max_mixtures=5,
                   n_std=3,
                   plot=True):
    """
    Get automatic min/max thresholds for input data array. The function will fit a gaussian mixture model, and find the threshold
    based on the mean and standard deviation of the largest mixture in the model.

    Parameters
    ----------
    data : numpy.ndarray
        Array of data to find thresholds for.
    max_mixtures : int, default 5
        Maximum number of gaussian mixtures to fit.
    n_std : int or float, default 3
        Number of standard deviations from distribution mean to set as min/max thresholds.
    plot : bool, default True
        If True, will plot the distribution of BIC and the fit of the gaussian mixtures to the data.

    Returns
    --------
    thresholds : dict
        Dictionary with min and max thresholds.
    """

    # Get numpy array if input was pandas series or list
    data_type = type(data).__name__
    if data_type == "Series":
        data = data.values
    elif data_type == "list":
        data = np.array(data)

    # Attempt to reshape values
    data = data.reshape(-1, 1)

    # Fit data with gaussian mixture
    n_list = list(range(1, max_mixtures + 1))  # 1->max mixtures per model
    models = [None] * len(n_list)
    for i, n in enumerate(n_list):
        models[i] = GaussianMixture(n).fit(data)

    # Evaluate quality of models
    # AIC = [m.aic(data) for m in models]
    BIC = [m.bic(data) for m in models]

    # Choose best number of mixtures
    try:
        kn = KneeLocator(n_list, BIC, curve='convex', direction='decreasing')
        M_best = models[kn.knee - 1]  # -1 to get index

    except Exception:
        # Knee could not be found; use the normal distribution estimated using one gaussian
        M_best = models[0]

    # Which is the largest component? And what are the mean/variance of this distribution?
    weights = M_best.weights_
    i = np.argmax(weights)
    dist_mean = M_best.means_[i][0]
    dist_std = np.sqrt(M_best.covariances_[i][0][0])

    # Threshold estimation
    thresholds = {"min": dist_mean - dist_std * n_std,
                  "max": dist_mean + dist_std * n_std}

    # ------ Plot if chosen -------#
    if plot:

        fig, axarr = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
        axarr = axarr.ravel()

        # Plot distribution of BIC
        # plt.plot(n_list, AIC, color="red", label="AIC")
        axarr[0].plot(n_list, BIC, color="blue")
        axarr[0].set_xlabel("Number of mixtures")
        axarr[0].set_ylabel("BIC")

        # Plot distribution of gaussian mixtures
        min_x = min(data)
        max_x = max(data)
        x = np.linspace(min_x, max_x, 1000).reshape(-1, 1)
        logprob = M_best.score_samples(x)
        responsibilities = M_best.predict_proba(x)
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

        axarr[1].hist(data, density=True)
        axarr[1].set_xlabel("Value")
        axarr[1].set_ylabel("Density")
        for i in range(M_best.n_components):
            w = weights[i] * 100
            axarr[1].plot(x, pdf_individual[:, i], label=f"Component {i+1} ({w:.0f}%)")

        axarr[1].axvline(thresholds["min"], color="red", linestyle="--")
        axarr[1].axvline(thresholds["max"], color="red", linestyle="--")
        axarr[1].legend(bbox_to_anchor=(1.05, 1), loc=2)  # locate legend outside of plot

    return thresholds


def automatic_thresholds(adata, which="obs", groupby=None, columns=None):
    """
    Get automatic thresholds for multiple data columns in adata.obs or adata.var.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object to find thresholds for.
    which : str, default "obs"
        Which data to find thresholds for. Either "obs" or "var".
    groupby : str, default None
        Group rows by the column given in 'groupby' to find thresholds independently per group
    columns : list, default None
        Columns to calculate automatic thresholds for. If None, will take all numeric columns.

    Returns
    --------
    dict
        A dict containing thresholds for each data column, either grouped by groupby or directly containing "min" and "max" per column.
    """

    # Find out which data to find thresholds for
    if which == "obs":
        table = adata.obs
    elif which == "var":
        table = adata.var
    else:
        raise ValueError("'which' must be either 'obs' or 'var'.")

    # Establish which columns to find thresholds for
    if columns is None:
        columns = list(table.select_dtypes(np.number).columns)

    # Check groupby
    if groupby is not None:
        if groupby not in table.columns:
            raise ValueError()

    # Get threshold per data column (and groupby if chosen)
    thresholds = {}
    for col in columns:

        if groupby is None:
            data = table[col].values
            d = get_thresholds(data, plot=False)
            thresholds[col] = d

        else:
            thresholds[col] = {}  # initialize to fill in per group
            for group, subtable in table.groupby(groupby):
                data = subtable[col].values
                d = get_thresholds(data, plot=False)
                thresholds[col][group] = d

    return thresholds


def thresholds_as_table(threshold_dict):
    """ Show the threshold dictionary as a table.

    Parameters
    ----------
    threshold_dict : dict
        Dictionary with thresholds.

    Returns
    -------
    pandas.DataFrame
    """

    rows = []
    for column in threshold_dict:

        if "min" in threshold_dict[column] or "max" in threshold_dict[column]:
            row = [column, np.nan, threshold_dict[column].get("min", np.nan), threshold_dict[column].get("max", np.nan)]
            rows.append(row)
        else:
            for group in threshold_dict[column]:
                row = [column, group, threshold_dict[column][group].get("min", np.nan), threshold_dict[column][group].get("max", np.nan)]
                rows.append(row)

    # Assemble table
    df = pd.DataFrame(rows)
    df.columns = ["Parameter", "Group", "Minimum", "Maximum"]

    # Remove group column if no thresholds had groups
    if df["Group"].isnull().sum() == df.shape[0]:
        df.drop(columns="Group", inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    return df


######################################################################################
#                     STEP 2: PLOT AND DEFINE CUSTOM CUTOFFS                         #
######################################################################################

def _validate_minmax(d):
    """
    Validate that the dict 'd' contains the keys 'min' and 'max'.
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

    Raises
    ------
    ValueError
        If the threshold dict is not valid.
    """

    if groupby is not None:
        groups = table[groupby]

    # Check if all columns in thresholds are available
    threshold_columns = thresholds.keys()
    not_found = [col for col in threshold_columns if col not in table.columns]
    if len(not_found) > 0:
        raise ValueError("Column(s) '{0}' given in thresholds are not found in table".format(not_found))

    # Check the format of individual column thresholds
    for col in thresholds:

        if groupby is None:  # expecting one threshold for all cells
            _validate_minmax(thresholds[col])

        else:  # Expecting either one threshold or a threshold for each sample

            for key in thresholds[col]:
                if key in groups:
                    minmax_dict = thresholds[col][key]
                    _validate_minmax(minmax_dict)
                else:  # this is a minmax threshold
                    _validate_minmax(thresholds[col])


def _link_sliders(sliders):
    """ Link the values between interactive sliders.

    Parameters
    ------------
    sliders : list of ipywidgets.widgets.Slider
        List of sliders to link.

    Returns
    --------
    list : list of ipywidgets.widgets.link
        List of links between sliders.
    """

    tup = [(slider, 'value') for slider in sliders]

    linkage_list = []
    for i in range(1, len(tup)):
        link = ipywidgets.link(*tup[i - 1:i + 1])
        linkage_list.append(link)

    return linkage_list


def _toggle_linkage(checkbox, linkage_dict, slider_list, key):
    """
    Either link or unlink sliders depending on the new value of the checkbox.

    Parameters
    -----------
    checkbox : ipywidgets.widgets.Checkbox
        Checkbox to toggle linkage.
    linkage_dict : dict
        Dictionary of links to link or unlink.
    slider_list : list of ipywidgets.widgets.Slider
        List of sliders to link or unlink.
    key : str
        Key in linkage_dict for fetching and updating links.
    """

    check_bool = checkbox["new"]

    if check_bool is True:
        if linkage_dict[key] is None:  # link sliders if they have not been linked yet
            linkage_dict[key] = _link_sliders(slider_list)  # overwrite None with the list of links

        for linkage in linkage_dict[key]:
            linkage.link()

    elif check_bool is False:

        if linkage_dict[key] is not None:  # only unlink if there are links to unlink
            for linkage in linkage_dict[key]:
                linkage.unlink()


def _update_thresholds(slider, fig, min_line, min_shade, max_line, max_shade):
    """ Update the locations of thresholds in plot """

    tmin, tmax = slider["new"]  # threshold values from slider

    # Update min line
    ydata = min_line.get_ydata()
    ydata = [tmin for _ in ydata]
    min_line.set_ydata(ydata)

    x, y = min_shade.get_xy()
    min_shade.set_height(tmin - y)

    # Update max line
    ydata = max_line.get_ydata()
    ydata = [tmax for _ in ydata]
    max_line.set_ydata(ydata)

    x, y = max_shade.get_xy()
    max_shade.set_height(tmax - y)

    # Draw figure after update
    fig.canvas.draw_idle()

    # Save figure
    # sctoolbox.utilities.save_figure(save)


def quality_violin(adata, columns,
                   which="obs",
                   groupby=None,
                   ncols=2,
                   header=None,
                   color_list=None,
                   title=None,
                   thresholds=None,
                   global_threshold=True,
                   interactive=True,
                   save=None):
    """
    A function to plot quality measurements for cells in an anndata object.

    Note
    ------
    Notebook needs "%matplotlib widget" before the call for the interactive sliders to work.

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
    global_threshold : bool, default True
        Whether to use global thresholding as the initial setting. If False, thresholds are set per group.
    interactive : bool, Default True
        Whether to show interactive sliders. If False, the static matplotlib plot is shown.
    save : str, optional
        Save the figure to the path given in 'save'. Default: None (figure is not saved).

    Returns
    -----------
    tuple of box, dict
        box contains the sliders and figure to show in notebook, and the dictionary contains the sliders determined by sliders
    """

    is_interactive = sctoolbox.utilities._is_interactive()

    # ---------------- Test input and get ready --------------#

    ncols = min(ncols, len(columns))  # Make sure ncols is not larger than the number of columns
    nrows = int(np.ceil(len(columns) / ncols))

    # Decide which table to use
    if which == "obs":
        table = adata.obs
    elif which == "var":
        table = adata.var
    else:
        raise ValueError()

    # Order of categories on x axis
    if groupby is not None:
        groups = table[groupby].cat.categories
        n_colors = len(groups)
    else:
        groups = None
        n_colors = 1

    # Setup colors to be used
    if color_list is None:
        color_list = sns.color_palette("Set1", n_colors)
    else:
        if int(n_colors) <= int(len(color_list)):
            raise ValueError("Increase the color_list variable to at least {} colors.".format(n_colors))
        else:
            color_list = color_list[:n_colors]

    # Setup headers to be used
    if header is None:
        header = columns
    else:
        # check that header has the right length
        if len(header) != len(columns):
            raise ValueError("Length of header does not match length of columns")

    # Setup thresholds if not given
    if thresholds is None:
        thresholds = {col: {} for col in columns}

    # ---------------- Setup figure --------------#

    # Setting up output figure
    plt.ioff()  # prevent plot from showing twice in notebook

    if is_interactive:
        figsize = (ncols * 3, nrows * 3)
    else:
        figsize = (ncols * 4, nrows * 4)  # static plot can be larger

    fig, axarr = plt.subplots(nrows, ncols, figsize=figsize)
    axes_list = axarr.flatten()

    # Remove empty axes
    for ax in axes_list[len(columns):]:
        ax.axis('off')

    # Add title of full plot
    if title is not None:
        fig.suptitle(title)
        fontsize = fig._suptitle._fontproperties._size * 1.2  # increase fontsize of title
        plt.setp(fig._suptitle, fontsize=fontsize)

    # Add title of individual plots
    for i in range(len(columns)):
        ax = axes_list[i]
        ax.set_title(header[i], fontsize=11)

    # ------------- Plot data and add sliders ---------#

    # Plotting data
    slider_dict = {}
    linkage_dict = {}  # one link list per column
    accordion_content = []
    for i, column in enumerate(columns):
        ax = axes_list[i]
        slider_dict[column] = {}

        # Plot data from table
        sns.violinplot(data=table, x=groupby, y=column, ax=ax, order=groups, palette=color_list)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_ylabel("")
        ax.set_xlabel("")

        ticks = ax.get_xticks()
        ymin, ymax = ax.get_ylim()  # ylim before plotting any thresholds

        # Establish groups
        if groupby is not None:
            group_names = groups
        else:
            group_names = ["Threshold"]

        # Plot thresholds per group
        y_range = ymax - ymin
        nothresh_min = ymin - y_range * 0.1  # just outside of y axis range
        nothresh_max = ymax + y_range * 0.1

        data_min = table[column].min()
        data_max = table[column].max()
        slider_list = []
        for j, group in enumerate(group_names):

            # Establish the threshold to plot
            if column not in thresholds:  # no thresholds given
                tmin = nothresh_min
                tmax = nothresh_max
            elif group in thresholds[column]:  # thresholds per group
                tmin = thresholds[column][group].get("min", nothresh_min)
                tmax = thresholds[column][group].get("max", nothresh_max)
            else:
                tmin = thresholds[column].get("min", nothresh_min)
                tmax = thresholds[column].get("max", nothresh_max)

            # Plot line and shading
            tick = ticks[j]
            x = [tick - 0.5, tick + 0.5]

            min_line = ax.plot(x, [tmin] * 2, color="red", linestyle="--")[0]
            max_line = ax.plot(x, [tmax] * 2, color="red", linestyle="--")[0]

            min_shade = ax.add_patch(Rectangle((x[0], ymin), x[1] - x[0], tmin - ymin, color="grey", alpha=0.2, linewidth=0))  # starting at lower left with positive height
            max_shade = ax.add_patch(Rectangle((x[0], ymax), x[1] - x[0], tmax - ymax, color="grey", alpha=0.2, linewidth=0))  # starting at upper left with negative height

            # Add slider to control thresholds
            if is_interactive:

                slider = ipywidgets.FloatRangeSlider(description=group, min=data_min, max=data_max,
                                                     value=[tmin, tmax],  # initial value
                                                     continuous_update=False)

                slider.observe(functools.partial(_update_thresholds,
                                                 fig=fig,
                                                 min_line=min_line,
                                                 min_shade=min_shade,
                                                 max_line=max_line,
                                                 max_shade=max_shade), names=["value"])

                slider_list.append(slider)
                if groupby is not None:
                    slider_dict[column][group] = slider
                else:
                    slider_dict[column] = slider

        ax.set_ylim(ymin, ymax)  # set ylim back to original after plotting thresholds

        # Link sliders together
        if is_interactive:

            if len(slider_list) > 1:

                # Toggle linked sliders
                c = ipywidgets.Checkbox(value=global_threshold, description='Global threshold', disabled=False, indent=False)
                linkage_dict[column] = _link_sliders(slider_list) if global_threshold is True else None

                c.observe(functools.partial(_toggle_linkage,
                                            linkage_dict=linkage_dict,
                                            slider_list=slider_list,
                                            key=column), names=["value"])

                box = ipywidgets.VBox([c] + slider_list)

            else:
                box = ipywidgets.VBox(slider_list)  # no tickbox needed if there is only one slider per column

            accordion_content.append(box)

    fig.tight_layout()
    sctoolbox.utilities.save_figure(save)  # save plot; can be overwritten if thresholds are changed

    # Assemble accordion with different measures
    if is_interactive:

        accordion = ipywidgets.Accordion(children=accordion_content, selected_index=None)
        for i in range(len(columns)):
            accordion.set_title(i, columns[i])

        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False
        fig.canvas.resizable = True
        fig.canvas.width = "auto"

        # Hack to force the plot to show
        # reference: https://github.com/matplotlib/ipympl/issues/290
        fig.canvas._handle_message(fig.canvas, {'type': 'send_image_mode'}, [])
        fig.canvas._handle_message(fig.canvas, {'type': 'refresh'}, [])
        fig.canvas._handle_message(fig.canvas, {'type': 'initialized'}, [])
        fig.canvas._handle_message(fig.canvas, {'type': 'draw'}, [])

        fig.canvas.draw()
        figure = ipywidgets.HBox([accordion, fig.canvas])  # Setup box to hold all widgets

    else:
        figure = fig  # non interactive figure

    return figure, slider_dict


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

        if isinstance(slider_dict[measure], dict):  # thresholds for groups
            for group in slider_dict[measure]:
                slider = slider_dict[measure][group]
                threshold_dict[measure][group] = {"min": slider.value[0], "max": slider.value[1]}

            # Check if all groups have the same thresholds
            mins = set([d["min"] for d in threshold_dict[measure].values()])
            maxs = set([d["max"] for d in threshold_dict[measure].values()])

            # Set overall threshold if individual sliders are similar
            if len(mins) == 1 and len(maxs) == 1:
                threshold_dict[measure] = threshold_dict[measure][group]  # takes the last group from the previous for loop

        else:  # One threshold for measure
            slider = slider_dict[measure]
            threshold_dict[measure] = {"min": slider.value[0], "max": slider.value[1]}

    return threshold_dict


###############################################################################
#                           STEP 3: APPLYING CUTOFFS                          #
###############################################################################

def apply_qc_thresholds(adata, thresholds, which="obs", groupby=None, inplace=True):
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

    # Cells or genes? For naming in log prints
    if which == "obs":
        name = "cells"
    else:
        name = ".var features"

    # Check if all columns are found in adata
    not_found = list(set(thresholds) - set(table.columns))
    if len(not_found) > 0:
        print("{0} threshold columns were not found in adata and could therefore not be applied. These columns are: {1}".format(len(not_found), not_found))
    thresholds = {k: thresholds[k] for k in thresholds if k not in not_found}

    if len(thresholds) == 0:
        raise ValueError(f"The thresholds given do not match the columns given in adata.{which}. Please adjust the 'which' parameter if needed.")

    if groupby is not None:
        groups = table[groupby].cat.categories

    # Check that thresholds contain min/max
    for column, d in thresholds.items():
        if 'min' not in d and 'max' not in d:
            if groupby is not None:
                keys = d.keys()
                not_found = list(set(keys) - set(groups))
                if len(not_found) > 0:
                    raise ValueError(f"{len(not_found)} groups from thresholds were not found in adata.obs[{groupby}]. These groups are: {not_found}")

            else:
                raise ValueError("Error in threshold format: Thresholds must contain min or max per column, or a threshold per group in groupby")

    # Apply thresholds
    for column, d in thresholds.items():

        # Update size of table
        table = adata.obs if which == "obs" else adata.var

        # Collect boolean array of rows to select of table
        global_threshold = False  # can be overwritten if thresholds are global
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

        # Select using a global threshold
        if global_threshold is True:
            minmax_dict = d

            if "min" in minmax_dict:
                excluded = excluded | (table[column] < minmax_dict["min"])  # if already excluded, or if excluded by min

            if "max" in minmax_dict:
                excluded = excluded | (table[column] > minmax_dict["max"])

        # Apply filtering
        included = ~excluded

        if inplace:
            # NOTE: these are privat anndata functions so they might change without warning!
            if which == "obs":
                adata._inplace_subset_obs(included)
            else:
                adata._inplace_subset_var(included)
        else:
            if which == "obs":
                adata = adata[included]
            else:
                adata = adata[:, included]  # filter on var

        print(f"Filtering based on '{column}' from {len(table)} -> {sum(included)} {name}")

    if inplace is False:
        return adata


###############################################################################
#                         STEP 4: ADDITIONAL FILTERING                        #
###############################################################################

def _filter_object(adata, filter, which="obs", remove_bool=True, inplace=True):
    """
    Filter an adata object based on a filter on either obs (cells) or var (genes). Is called by filter_cells and filter_genes.
    """

    # Decide which element type (genes/cells) we are dealing with
    if which == "obs":
        table = adata.obs
        table_name = "adata.obs"
        element_name = "cells"
    else:
        table = adata.var
        table_name = "adata.var"
        element_name = "genes"

    n_before = len(table)

    # genes is either a string (column in .var table) or a list of genes to remove
    if isinstance(filter, str):
        if filter not in table.columns:
            raise ValueError(f"Column {filter} not found in {table_name}.columns")

        boolean = table[filter].values
        if remove_bool is True:
            boolean = ~boolean

    else:
        # Check if all genes/cells are found in adata
        not_found = list(set(filter) - set(table.index))
        if len(not_found) > 0:
            print(f"{len(not_found)} {element_name} were not found in adata and could therefore not be removed. These genes are: {not_found}")

        boolean = ~table.index.isin(filter).values

    # Remove genes from adata
    if inplace:
        if which == "obs":
            adata._inplace_subset_obs(boolean)
        elif which == "var":
            adata._inplace_subset_var(boolean)  # boolean is the included genes
    else:
        if which == "obs":
            adata = adata[boolean]
        elif which == "var":
            adata = adata[:, boolean]

    n_after = adata.shape[0] if which == "obs" else adata.shape[1]
    filtered = n_before - n_after
    print(f"Filtered out {filtered} {element_name} from adata. New number of {element_name} is: {n_after}")

    if inplace is False:
        return adata


def filter_cells(adata, cells, remove_bool=True, inplace=True):
    """
    Remove cells from anndata object.

    Parameters
    -----------
    adata : AnnData
        Anndata object to filter.
    cells : str or list of str
        A column in .obs containing boolean indicators or a list of cells to remove from object .obs table.
    remove_bool : bool, default True
        Is used if genes is a column in .obs table. If True, remove cells that are True. If False, remove cells that are False.
    inplace : bool, default True
        If True, filter inplace. If False, return filtered adata object.

    Returns
    -------
    anndata.AnnData or None
        If inplace is False, returns the filtered Anndata object. If inplace is True, returns None.
    """

    ret = _filter_object(adata, cells, which="obs", remove_bool=remove_bool, inplace=inplace)

    return ret  # adata objec or None


def filter_genes(adata, genes, remove_bool=True, inplace=True):
    """
    Remove genes from adata object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object to filter
    genes : str or list of str
        A column containing boolean indicators or a list of genes to remove from object .var table.
    remove_bool : bool, default True
        Is used if genes is a column in .var table. If True, remove genes that are True. If False, remove genes that are False.
    inplace : bool, default True
        If True, filter inplace. If False, return filtered adata object.

    Returns
    -------
    anndata.AnnData or None
        If inplace is False, returns the filtered Anndata object. If inplace is True, returns None.
    """

    ret = _filter_object(adata, genes, which="var", remove_bool=remove_bool, inplace=inplace)

    return ret  # adata objec or None


###############################################################################
#                   Functions leftover after refactoring                      #
###############################################################################

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

    # get index name(s) and column name(s)
    index_names = [thresholds.index.name] if thresholds.index.nlevels <= 1 else list(thresholds.index.names)
    column_names = list(thresholds.columns)

    def convert_type(val):
        """ Evaluate string if possible else return string. """
        try:
            return eval(val)
        except Exception:
            return val

    def select_row(table, quit="Quit"):
        """ Select a row to work on. Returns index tuple or False if quit. """
        index_names = [table.index.name] if table.index.nlevels <= 1 else table.index.names

        # select row by giving every level of index
        selected_indices = []
        for index in index_names:
            options = list(set(table.index.get_level_values(index)))

            print(f"Select {index}:")
            [print(f"    - {o}") for o in options + [quit]]

            selected_indices.append(
                click.prompt("", type=click.Choice(options + [quit]), show_choices=False, prompt_suffix="")
            )

            # go back to main menu if 'quit'
            if quit == selected_indices[-1]:
                return False

        return tuple(selected_indices)

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
            for column in index_names + column_names:
                new[column] = [click.prompt(f"Enter value for '{column}' column")]

            # add row
            thresholds = pd.concat([thresholds, pd.DataFrame(new).set_index(index_names)])

        # edit row
        elif selection == 2:
            print("Edit row.")

            # select row by giving every level of index
            selection = select_row(table=thresholds, quit=quit[0])
            print(selection)

            # update selected row
            if selection is not False:
                for column in thresholds.columns:
                    # update value
                    thresholds.at[selection, column] = click.prompt(f"Update {column} leave empty to keep former value", default=thresholds.at[selection, column], value_proc=convert_type)

        # remove row
        elif selection == 3:
            print("Select row to remove:")

            # select row by giving every level of index
            selection = select_row(table=thresholds, quit=quit[0])

            if selection:
                # remove row
                thresholds.drop(index=selection, inplace=True)

        # show table
        elif selection == 4:
            # clear output
            utilities.clear()

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

    return thresholds if not inplace else None


def find_thresholds(anndata, interval=None, var="all", obs="all", var_color_by=None, obs_color_by=None, output=None):
    """
    Find thresholds for the given .obs (cell) and .var (gene) columns in anndata.

    Parameters
    ----------
    anndata : anndata.AnnData
        anndata object
    interval : int or float, default None
        The percentage (from 0 to 100) to be used to calculate the cutoffs. None to show plots without threshold.
    var : str or list of str, default 'all'
        Anndata.var (gene) columns to find thresholds for. If 'all' will select all numeric columns. Use None to disable.
    obs : str or list of str, default 'all'
        Anndata.obs (cell) columns to find thresholds for. If 'all' will select all numeric columns. Use None to disable.
    var_color_by : str, default None
        Split anndata.var related violins into color groups using .var column of the given name.
    obs_color_by : str, default None
        Split anndata.obs related violins into color groups using .obs column of the given name.
    output : str or bool, default None
        Path + filename to save plot to. If True instead of str will save plot to "<project_folder>/qc_violin.pdf".

    Returns
    -------
    pandas.DataFrame or None:
        A pandas dataframe with the defined cutoff parameters. None if no interval is set.
    """
    # -------------------- checks & setup ------------------- #
    # is interval valid?
    if interval and not checker.in_range(interval, (0, 100)):
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
    if obs:
        obs = obs if isinstance(obs, list) else [obs]
    else:
        obs = []
    if var:
        var = var if isinstance(var, list) else [var]
    else:
        var = []

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
    thresholds = {'name': [], 'origin': [], 'threshold': [], 'color_by': []}
    for name, origin in zip(obs + var, ["obs"] * len(obs) + ["var"] * len(var)):
        thresholds['name'].append(name)
        thresholds['origin'].append(origin)
        thresholds['threshold'].append(None)
        thresholds['color_by'].append(obs_color_by if origin == "obs" else var_color_by)

    thresholds = pd.DataFrame.from_dict(thresholds).set_index(["name", "origin"])

    # Plotting with or without cutoffs
    if interval:
        # Calculate cutoffs to plot in the violins
        for name, origin in thresholds.index:
            # compute cutoffs for each column
            if origin == "obs":
                data = anndata.obs[name]
            else:
                data = anndata.var[name]

            # compute threshold
            thresholds.at[(name, origin), "threshold"] = analyser.get_threshold(data=data.to_list(), interval=interval, limit_on="both")

    # create violinplot
    plotting.qc_violins(anndata, thresholds, colors=None, filename=output)

    # TODO return anndata containing threshold table instead
    if interval:
        return thresholds


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
