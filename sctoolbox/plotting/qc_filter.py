from math import ceil
import pandas as pd
import copy
import numpy as np
import ipywidgets
import functools  # for partial functions

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sctoolbox.utils as utils
from sctoolbox.plotting.general import violinplot, _save_figure


########################################################################################
# ---------------------------- Plots for counting cells ------------------------------ #
########################################################################################

def _n_cells_pieplot(adata, groupby,
                     figsize=None):
    """
    Plot number of cells per group in a pieplot.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    groupby : str
        Name of the column in adata.obs to group by.

    Returns
    -------


    Examples
    --------


    """

    # Get counts
    counts = adata.obs[groupby].value_counts()
    counts
    # in progress


def n_cells_barplot(adata, x, groupby=None, stacked=True, save=None, figsize=None,
                    add_labels=False,
                    **kwargs):
    """
    Plot number and percentage of cells per group in a barplot.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    x : str
        Name of the column in adata.obs to group by on the x axis.
    groupby : str, default None
        Name of the column in adata.obs to created stacked bars on the y axis. If None, the bars are not split.
    stacked : bool, default True
        Whether to stack the bars or not.
    save : str, default None
        Path to save the plot. If None, the plot is not saved.
    figsize : tuple, default None
        Size of figure, e.g. (4, 8). If None, size is determined automatically depending on whether groupby is None or not.
    add_labels : bool, default False
        Whether to add labels to the bars giving the number/percentage of cells.
    **kwargs : arguments
        Additional arguments passed to pandas.DataFrame.plot.bar.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.n_cells_barplot(adata, x="louvain")

    .. plot::
        :context: close-figs

        pl.n_cells_barplot(adata, x="louvain", groupby="condition")

    """

    # Get cell counts for groups or all
    tables = []
    if groupby is not None:
        for i, frame in adata.obs.groupby(groupby):
            count = frame.value_counts(x).to_frame(name="count").reset_index()
            count["groupby"] = i
            tables.append(count)
        counts = pd.concat(tables)

    else:
        counts = adata.obs[x].value_counts().to_frame(name="count").reset_index()
        counts.rename(columns={"index": x}, inplace=True)
        counts["groupby"] = "all"

    # Format counts
    counts_wide = counts.pivot(index=x, columns="groupby", values="count")
    counts_wide_percent = counts_wide.div(counts_wide.sum(axis=1), axis=0) * 100

    # Plot barplots
    if figsize is None:
        figsize = (5 + 5 * (groupby is not None), 3)  # if groupby is not None, add 5 to width

    if groupby is not None:
        _, axarr = plt.subplots(1, 2, figsize=figsize)
    else:
        _, axarr = plt.subplots(1, 1, figsize=figsize)  # axarr is a single axes
        axarr = [axarr]

    counts_wide.plot.bar(stacked=stacked, ax=axarr[0], legend=False, **kwargs)
    axarr[0].set_title("Number of cells")
    axarr[0].set_xticklabels(axarr[0].get_xticklabels(), rotation=45, ha="right")
    axarr[0].grid(False)

    if groupby is not None:
        counts_wide_percent.plot.bar(stacked=stacked, ax=axarr[1], **kwargs)
        axarr[1].set_title("Percentage of cells")
        axarr[1].set_xticklabels(axarr[1].get_xticklabels(), rotation=45, ha="right")
        axarr[1].grid(False)

        # Set location of legend
        axarr[1].legend(title=groupby, bbox_to_anchor=(1, 1), frameon=False,
                        handlelength=1, handleheight=1  # make legend markers square
                        )

        # Draw line at 100% if values are stacked
        if stacked is True:
            axarr[1].axhline(100, color='black', linestyle='--', linewidth=0.5, zorder=0)

    # Add labels to bars
    if add_labels:
        for i, ax in enumerate(axarr):
            for c in ax.containers:
                labels = [v.get_height() if v.get_height() > 0 else '' for v in c]  # no label if segment is 0
                if i == 0:
                    labels = [str(int(v)) for v in labels]  # convert to int
                else:
                    labels = ["%.1f" % v + "%" for v in labels]  # round and add % sign
                    labels = [label.replace(".0", "") for label in labels]  # remove .0 from 100.0%
                ax.bar_label(c, labels=labels, label_type='center')

    # Remove spines
    for ax in axarr:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    _save_figure(save)

    return axarr


def group_correlation(adata, groupby, method="spearman", save=None):
    """
    Plot correlation matrix between groups in `groupby`.
    The function expects the count data in .X to be normalized across cells.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    groupby : str
        Name of the column in adata.obs to group cells by.
    method : str, default "spearman"
        Correlation method to use. See pandas.DataFrame.corr for options.

    Returns
    -------
    ClusterGrid object

    Example
    --------
    .. plot::
        :context: close-figs

        import scanpy as sc
        import sctoolbox.plotting as pl

    .. plot::
        :context: close-figs

        adata = sc.datasets.pbmc68k_reduced()

    .. plot::
        :context: close-figs

        pl.group_correlation(adata, "phase", method="spearman", save=None)
    """

    # Calculate correlation of groups
    count_table = utils.pseudobulk_table(adata, groupby=groupby)
    corr = count_table.corr(numeric_only=False, method=method)

    # Plot clustermap
    g = sns.clustermap(corr, figsize=(4, 4),
                       xticklabels=True,
                       yticklabels=True,
                       cmap="Reds",
                       cbar_kws={'orientation': 'horizontal', 'label': method})
    g.ax_heatmap.set_facecolor("grey")

    # Adjust cbar
    n = len(corr)
    pos = g.ax_heatmap.get_position()
    cbar_h = pos.height / n / 2
    g.ax_cbar.set_position([pos.x0, pos.y0 - 3 * cbar_h, pos.width, cbar_h])

    # Final adjustments
    g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.xaxis.tick_top()
    _ = g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="left")

    _save_figure(save)

    return g


def qc_violins(anndata, thresholds, colors=None, save=None, ncols=3, figsize=None, dpi=300):
    """
    Grid of violinplots with optional cutoffs.

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object providing violin data.
    thresholds : pandas.DataFrame
        Dataframe with anndata.var & anndata.obs column names as index, and threshold column with lists of cutoff lines to draw.
        Note: Row order defines plot order.
        Structure:

            - index (two columns)
                - Name of anndata.var or anndata.obs column.
                - Name of origin. Either "obs" or "var".
            - 1st column: Threshold number(s) defining violinplot lines. Either None, single number or list of numbers.
            - 2nd column: Name of anndata.var or anndata.obs column used for color grouping or None to disable.
    colors : list of str, default None
        List of colors for the violins.
    save : str, default None
        Path and name of file to be saved.
    ncols : int, default 3
        Number of violins per row.
    figsize : int tuple, default None
        Size of figure in inches.
    dpi : int, default 300
        Dots per inch.
    """
    # test if threshold indexes are column names in .obs or .var
    invalid_index = set(thresholds.index.get_level_values(0)) - set(anndata.obs.columns) - set(anndata.var.columns)
    if invalid_index:
        raise ValueError(f"Threshold table indices need to be column names of anndata.obs or anndata.var. Indices not found: {invalid_index}")

    # create subplot grid
    nrows = ceil(len(thresholds) / ncols)
    figsize = figsize if figsize is not None else (ncols * 4, nrows * 4)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, dpi=dpi, figsize=figsize, constrained_layout=True)
    axs = axs.flatten()  # flatten to 1d array per row

    # iterate over threshold rows
    for ((name, origin), row), ax in zip(thresholds.iterrows(), axs):
        # find out if in obs or var
        table = anndata.var if origin == "var" else anndata.obs

        # create violin
        violinplot(table=table, y=name, hlines=row[0], color_by=row[1], colors=colors, ax=ax, title=f"{origin}: {name}", ylabel=False)

    # delete unused subplots
    for i in range(len(thresholds), len(axs)):
        fig.delaxes(axs[i])

    # Save plot
    _save_figure(save)


#####################################################################
# --------------------------- Insertsize -------------------------- #
#####################################################################

def plot_insertsize(adata, barcodes=None):
    """
    Plot insertsize distribution for barcodes in adata. Requires adata.uns["insertsize_distribution"] to be set.

    Parameters
    -----------
    adata : AnnData
        AnnData object containing insertsize distribution in adata.uns["insertsize_distribution"].
    barcodes : list of str, default None
        Subset of barcodes to plot information for. If None, all barcodes are used.

    Returns
    --------
    ax : matplotlib.Axes
        Axes object containing the plot.
    """

    if "insertsize_distribution" not in adata.uns:
        raise ValueError("adata.uns['insertsize_distribution'] not found!")

    insertsize_distribution = copy.deepcopy(adata.uns['insertsize_distribution'])
    insertsize_distribution.columns = insertsize_distribution.columns.astype(int)

    # Subset barcodes if a list is given
    if barcodes is not None:
        # Convert to list if only barcode is given
        if isinstance(barcodes, str):
            barcodes = [barcodes]
        table = insertsize_distribution.loc[barcodes].sum(axis=0)
    else:
        table = insertsize_distribution.sum(axis=0)

    # Plot
    ax = sns.lineplot(x=table.index, y=table.values)
    ax.set_xlabel("Insertsize (bp)")
    ax.set_ylabel("Count")

    return ax


###########################################################################
# ----------------- Interactive quality control plot -------------------- #
###########################################################################


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

    is_interactive = utils._is_interactive()

    # ---------------- Test input and get ready --------------#

    ncols = min(ncols, len(columns))  # Make sure ncols is not larger than the number of columns
    nrows = int(np.ceil(len(columns) / ncols))

    # Decide which table to use
    if which == "obs":
        table = adata.obs
    elif which == "var":
        table = adata.var
    else:
        raise ValueError("'which' must be either 'obs' or 'var'.")
    
    # Check that columns are in table
    invalid_columns = set(columns) - set(table.columns)
    if invalid_columns:
        raise ValueError(f"The following columns from 'columns' were not found in '{which}' table: {invalid_columns}")

    # Order of categories on x axis
    if groupby is not None:
        # Convert to category
        if table[groupby].dtype.name != "category":
            table[groupby] = table[groupby].astype('category')
        groups = list(table[groupby].cat.categories)
        n_colors = len(groups)
    else:
        groups = None
        n_colors = 1

    # Setup colors to be used
    if color_list is None:
        color_list = sns.color_palette("Set1", n_colors)
    else:
        if int(n_colors) > int(len(color_list)):
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
    axes_list = [axarr] if type(axarr).__name__ == "Axes" else axarr.flatten()

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
        sns.violinplot(data=table, x=groupby, y=column, ax=ax, order=groups, palette=color_list, cut=0)
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

            # Replace None with nothresh
            tmin = nothresh_min if tmin is None else tmin
            tmax = nothresh_max if tmax is None else tmax

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
    _save_figure(save)  # save plot; can be overwritten if thresholds are changed

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
