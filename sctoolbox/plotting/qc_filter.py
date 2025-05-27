"""Functions for plotting QC-related figures e.g. number of cells per group and violins."""

import pandas as pd
import copy
import numpy as np
import ipywidgets
import traitlets
import functools  # for partial functions
import glob
import scanpy as sc
import warnings

import upsetplot
import seaborn as sns
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sctoolbox.utils as utils
from sctoolbox.plotting.general import _save_figure
import sctoolbox.utils.decorator as deco

# type hint imports
from beartype.typing import Tuple, Dict, Optional, Literal, Callable, Any  # , Union, List
from beartype import beartype
from numpy.typing import NDArray

from sctoolbox._settings import settings
logger = settings.logger


########################################################################################
# ------------------------------ QC plots for starsolo ------------------------------- #
########################################################################################

@beartype
def _read_starsolo_summary(folder: str) -> pd.DataFrame:
    """Get summary table from an output folder containing multiple starsolo runs.

    Parameters
    ----------
    folder : str
        Path to a folder, e.g. "path/to/starsolo_output", which contains folders "solorun1", "solorun2", etc.

    Raises
    ------
    ValueError
        If no summary files are found in the folder.

    Returns
    -------
    summary_table : pd.DataFrame
        Table with summary statistics from all runs.
    """

    summary_files = glob.glob(folder + "/**/solo/Gene/Summary.csv")
    if len(summary_files) == 0:
        raise ValueError(f"No STARsolo summary files found in folder '{folder}'. Please check the path and try again.")

    # Read statistics from summary files
    names = utils.general.clean_flanking_strings(summary_files)
    summary_tables = []
    for name, f in zip(names, summary_files):
        star_table = pd.read_csv(f, index_col=0, header=None, names=[name])
        summary_tables.append(star_table)
    summary_table = pd.concat(summary_tables, axis=1)

    return summary_table


@beartype
def plot_starsolo_quality(folder: str,
                          measures: list[str] = ["Number of Reads", "Reads Mapped to Genome: Unique",
                                                 "Reads Mapped to Gene: Unique Gene", "Fraction of Unique Reads in Cells",
                                                 "Median Reads per Cell", "Median Gene per Cell"],
                          ncol: int = 3,
                          order: Optional[list[str]] = None,
                          save: Optional[str] = None,
                          **kwargs: Any) -> NDArray[Axes]:
    """Plot quality measures from starsolo as barplots per condition.

    Parameters
    ----------
    folder : str
        Path to a folder, e.g. "path/to/starsolo_output", which contains folders "solorun1", "solorun2", etc.
    measures : list[str], default ["Number of Reads", "Reads Mapped to Genome: Unique", "Reads Mapped to Gene: Unique Gene", "Fraction of Unique Reads in Cells", "Median Reads per Cell", "Median Gene per Cell"]
        List of measures to plot. Must be available in the solo summary table.
    ncol : int, default 3
        Number of columns in the plot.
    order : Optional[list[str]], default None
        Order of conditions in the plot. If None, the order is alphabetical.
    save : Optional[str], default None
        Path to save the plot. If None, the plot is not saved.
    **kwargs : Any
        Additional arguments passed to seaborn.barplot.

    Returns
    -------
    axes : NDArray[Axes]
        Array of axes objects containing the plot(s).

    Raises
    ------
    KeyError
        If a measure is not available in the solo summary table.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.qc_filter.plot_starsolo_quality("data/quant/")
    """

    # Prepare functions for converting labels
    def format_million(label):
        return '{:,.0f} M'.format(int(label) / 10**6)

    def format_thousand(label):
        return '{:,.0f} K'.format(int(label) / 10**3)

    def format_percent(label):
        return '{:,.0f}%'.format(float(label) * 100)

    # Get summary table
    summary_table = _read_starsolo_summary(folder)
    available_measures = summary_table.index.tolist()

    if order is None:
        order = sorted(summary_table.columns.tolist())
        summary_table = summary_table[order]
    else:
        summary_table = summary_table[order]

    # Setup plot
    ncol = min(ncol, len(measures))
    row = int(np.ceil(len(measures) / ncol))
    fig, axes = plt.subplots(row, ncol, figsize=(ncol * 4, row * 4))
    axes = axes.flatten() if len(measures) > 1 else np.array([axes])  # axes is a list of axes objects
    _ = [ax.axis('off') for ax in axes[len(measures):]]  # hide additional plots

    # Fill in plot per measure
    for i, measure in enumerate(measures):
        if measure not in available_measures:
            raise KeyError(f"Measure '{measure}' not found in summary table. Available measures: {available_measures}")

        # Plot data to barplot
        ax = axes[i]
        data = summary_table.loc[measure].astype(float)
        sns.barplot(x=data.index, y=data.values, ax=ax, edgecolor="black", **kwargs)
        ax.set_title(measure)

        # Format yticklabels
        if data.max() < 1:  # convert to %
            ax.set_ylim(0, 1)
            ax.set_yticks(ax.get_yticks(), [format_percent(value) for value in ax.get_yticks()])
        elif data.max() < 10000:
            pass  # no format; show raw values
        elif data.max() < 10**6:  # convert to thousands
            ax.set_yticks(ax.get_yticks(), [format_thousand(value) for value in ax.get_yticks()])
        else:  # convert to millions
            ax.set_yticks(ax.get_yticks(), [format_million(value) for value in ax.get_yticks()])

        ax.set_xticks(ax.get_xticks())  # prevent locator error
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    _save_figure(save)

    return axes


@beartype
def plot_starsolo_UMI(folder: str,
                      ncol: int = 3,
                      save: Optional[str] = None) -> NDArray[Axes]:
    """Plot UMI distribution for each condition in a folder.

    Parameters
    ----------
    folder : str
        Path to a folder, e.g. "path/to/starsolo_output", which contains folders "solorun1", "solorun2", etc.
    ncol : int, default 3
        Number of columns in the plot.
    save : Optional[str], default None
        Path to save the plot. If None, the plot is not saved.

    Returns
    -------
    axes : NDArray[Axes]
        Array of axes objects containing the plot(s).

    Raises
    ------
    ValueError
        If no UMI files ('UMIperCellSorted.txt') are found in the folder.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.qc_filter.plot_starsolo_UMI("data/quant/", ncol=2)
    """

    summary_table = _read_starsolo_summary(folder)
    umi_files = glob.glob(folder + "/**/solo/Gene/UMIperCellSorted.txt")

    if len(umi_files) == 0:
        raise ValueError("No UMI files found in folder. Please check the path and try again.")

    names = utils.general.clean_flanking_strings(umi_files)

    # Setup plot
    ncol = min(len(names), ncol)
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 4))
    axes = axes.flatten() if len(names) > 1 else np.array([axes])  # axes is a list of axes objects
    _ = [ax.axis('off') for ax in axes[len(names):]]  # hide additional plots

    for i, f in enumerate(umi_files):

        ax = axes[i]
        name = names[i]

        df_knee = pd.read_table(f, header=None, names=[name])
        cut = int(summary_table.loc["Estimated Number of Cells", name])

        df_knee.plot.line(logx=True, logy=True, legend=False, ax=ax)
        df_knee[:cut].plot.line(logx=True, legend=False, ax=ax, color='red')
        ax.axvline(x=cut, color='grey', linestyle='-')

        vmax = df_knee.iloc[0, 0]
        ax.text(cut * 1.2, vmax, str(cut) + ' cells', verticalalignment='center')
        ax.set_title(name)
        ax.set_xlabel('Barcodes')
        ax.set_ylabel('UMI count')

    fig.tight_layout()
    _save_figure(save)

    return axes


########################################################################################
# ---------------------------- Plots for counting cells ------------------------------ #
########################################################################################

@deco.log_anndata
@beartype
def _n_cells_pieplot(adata: sc.AnnData,
                     groupby: str,
                     figsize: Optional[Tuple[int | float, int | float]] = None):
    """
    Plot number of cells per group in a pieplot.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    groupby : str
        Name of the column in adata.obs to group by.
    figsize : tuple, default None
        Size of figure, e.g. (4, 8).
    """

    # Get counts
    counts = adata.obs[groupby].value_counts()
    counts
    # in progress


@deco.log_anndata
@beartype
def n_cells_barplot(adata: sc.AnnData,
                    x: str,
                    groupby: Optional[str] = None,
                    stacked: bool = True,
                    save: Optional[str] = None,
                    figsize: Optional[Tuple[int | float, int | float]] = None,
                    add_labels: bool = False,
                    **kwargs: Any) -> NDArray[Axes]:
    """
    Plot number and percentage of cells per group in a barplot.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    x : str
        Name of the column in adata.obs to group by on the x axis.
    groupby : Optional[str], default None
        Name of the column in adata.obs to created stacked bars on the y axis. If None, the bars are not split.
    stacked : bool, default True
        Whether to stack the bars or not.
    save : Optional[str], default None
        Path to save the plot. If None, the plot is not saved.
    figsize : Optional[Tuple[int | float, int | float]], default None
        Size of figure, e.g. (4, 8). If None, size is determined automatically depending on whether groupby is None or not.
    add_labels : bool, default False
        Whether to add labels to the bars giving the number/percentage of cells.
    **kwargs : Any
        Additional arguments passed to pandas.DataFrame.plot.bar.

    Returns
    -------
    axarr : NDArray[Axes]
        Array of axes objects containing the plot(s).

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.qc_filter.n_cells_barplot(adata, x="louvain")

    .. plot::
        :context: close-figs

        pl.qc_filter.n_cells_barplot(adata, x="louvain", groupby="condition")
    """

    # Get cell counts for groups or all
    tables = []
    if groupby is not None:
        for i, frame in adata.obs.groupby(groupby, observed=False):
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
        axarr = np.array([axarr])

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


@deco.log_anndata
@beartype
def group_correlation(adata: sc.AnnData,
                      groupby: str,
                      method: Literal["spearman", "pearson", "kendall"] | Callable = "spearman",
                      save: Optional[str] = None,
                      **kwargs: Any) -> sns.matrix.ClusterGrid:
    """Plot correlation matrix between groups in `groupby`.

    The function expects the count data in .X to be normalized across cells.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    groupby : str
        Name of the column in adata.obs to group cells by.
    method : Literal["spearman", "pearson", "kendall"] | Callable, default "spearman"
        Correlation method to use. See pandas.DataFrame.corr for options.
    save : Optional[str], default None
        Path to save the plot. If None, the plot is not saved.
    **kwargs : Any
        Additional arguments passed to seaborn.clustermap.

    Returns
    -------
    sns.matrix.ClusterGrid

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.qc_filter.group_correlation(adata, "phase", method="spearman", save=None)
    """

    # Calculate correlation of groups
    count_table = utils.bioutils.pseudobulk_table(adata, groupby=groupby)
    corr = count_table.corr(numeric_only=False, method=method)

    clustermap_kwargs = {"figsize": (4, 4),
                         "cmap": "Reds",
                         "xticklabels": True,
                         "yticklabels": True,
                         "cbar_kws": {'orientation': 'horizontal', 'label': method}}  # defaults
    clustermap_kwargs.update(kwargs)    # overwrite defaults with user input

    # Plot clustermap
    g = sns.clustermap(corr, **clustermap_kwargs)
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


#####################################################################
# --------------------------- Insertsize -------------------------- #
#####################################################################

@deco.log_anndata
@beartype
def plot_insertsize(adata: sc.AnnData,
                    barcodes: Optional[list[str]] = None,
                    **kwargs: Any) -> Axes:
    """
    Plot insertsize distribution for barcodes in adata. Requires adata.uns["insertsize_distribution"] to be set.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing insertsize distribution in adata.uns["insertsize_distribution"].
    barcodes : Optional[list[str]], default None
        Subset of barcodes to plot information for. If None, all barcodes are used.
    **kwargs : Any
        Additional arguments passed to seaborn.lineplot.

    Returns
    -------
    ax : Axes
        Axes object containing the plot.

    Raises
    ------
    ValueError
        If adata.uns["insertsize_distribution"] is not set.
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
    ax = sns.lineplot(x=table.index, y=table.values, **kwargs)
    ax.set_xlabel("Insertsize (bp)")
    ax.set_ylabel("Count")

    return ax


###########################################################################
# ----------------- Interactive quality control plot -------------------- #
###########################################################################


@beartype
def _link_sliders(sliders: list[ipywidgets.widgets.FloatRangeSlider]) -> list[ipywidgets.link]:
    """Link the values between interactive sliders.

    Parameters
    ----------
    sliders : list[ipywidgets.widgets.FloatRangeSlider]
        List of sliders to link.

    Returns
    -------
    list[ipywidgets.link]
        List of links between sliders.
    """

    tup = [(slider, 'value') for slider in sliders]

    linkage_list = []
    for i in range(1, len(tup)):
        link = ipywidgets.link(*tup[i - 1:i + 1])
        linkage_list.append(link)

    return linkage_list


@beartype
def _toggle_linkage(checkbox: ipywidgets.widgets.Checkbox | traitlets.utils.bunch.Bunch,  # after first check, checkbox is a bunch object
                    linkage_dict: dict,
                    slider_list: list,
                    key: str) -> None:
    """
    Either link or unlink sliders depending on the new value of the checkbox.

    Parameters
    ----------
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
    """Update the locations of thresholds in plot."""

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


@deco.log_anndata
@beartype
def quality_violin(adata: sc.AnnData,
                   columns: list[str],
                   which: Literal["obs", "var"] = "obs",
                   groupby: Optional[str] = None,
                   ncols: int = 2,
                   header: Optional[list[str]] = None,
                   color_list: Optional[list[str | Tuple[float | int, float | int, float | int]]] = None,
                   title: Optional[str] = None,
                   thresholds: Optional[dict[str, dict[str, dict[Literal["min", "max"], int | float]] | dict[Literal["min", "max"], int | float]]] = None,
                   global_threshold: bool = True,
                   interactive: bool = True,
                   save: Optional[str] = None,
                   **kwargs: Any
                   ) -> Tuple[Any, Dict[str, Any]]:
    """
    Plot quality measurements for cells/features in an anndata object.

    Notes
    -----
    Notebook needs "%matplotlib widget" before the call for the interactive sliders to work.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object containing quality measures in .obs/.var
    columns : list[str]
        A list of columns in .obs/.var to show measures for.
    which : Literal["obs", "var"], default "obs"
        Which table to show quality for. Either "obs" / "var".
    groupby :  Optional[str], default "condition"
        A column in table to values on the x-axis.
    ncols : int, default 2
        Number of columns in the plot.
    header : Optional[list[str]], defaul None
        A list of custom headers for each measure given in columns.
    color_list : Optional[list[str]], default None
        A list of colors to use for violins. If None, colors are chosen automatically.
    title : Optional[str], default None
        The title of the full plot.
    thresholds : Optional[dict[str, dict[str, dict[Literal["min", "max"], int | float]] | dict[Literal["min", "max"], int | float]]], default None
        Dictionary containing initial min/max thresholds to show in plot.
    global_threshold : bool, default True
        Whether to use global thresholding as the initial setting. If False, thresholds are set per group.
    interactive : bool, default True
        Whether to show interactive sliders. If False, the static matplotlib plot is shown.
    save : Optional[str], optional
        Save the figure to the path given in 'save'. Default: None (figure is not saved).
    **kwargs : Any
        Additional arguments passed to seaborn.violinplot.

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        Tuple[Union[matplotlib.figure.Figure, ipywidgets.HBox], Dict[str, Union[List[ipywidgets.FloatRangeSlider.observe], Dict[str, ipywidgets.FloatRangeSlider.observe]]]]
        First element contains figure (static) or figure and sliders (interactive). The second element is a nested dict of slider values that are continously updated.

    Raises
    ------
    ValueError
        If 'which' is not 'obs' or 'var' or if columns are not in table.
    """

    is_interactive = utils.checker._is_interactive()

    # ---------------- Test input and get ready --------------#

    ncols = min(ncols, len(columns))  # Make sure ncols is not larger than the number of columns
    nrows = int(np.ceil(len(columns) / ncols))

    # Decide which table to use
    if which == "obs":
        table = adata.obs
    elif which == "var":
        table = adata.var

    # Check that columns are in table
    invalid_columns = set(columns) - set(table.columns)
    if invalid_columns:
        raise ValueError(f"The following columns from 'columns' were not found in '{which}' table: {invalid_columns}")

    # Order of categories on x axis
    if groupby is not None:
        groups = list(table[groupby].astype('category').cat.categories)
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
    axes_list = [axarr] if type(axarr).__name__.startswith("Axes") else axarr.flatten()

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
        sns.violinplot(data=table,
                       x=groupby,
                       hue=groupby,
                       y=column,
                       ax=ax,
                       order=groups,
                       palette=color_list if len(color_list) > 1 else None,  # fixes palette without hue warning
                       color=color_list[0] if len(color_list) == 1 else None,  # ^
                       cut=0,
                       legend=False,
                       **kwargs)
        ax.set_xticks(ax.get_xticks())  # get rid of userwarning https://stackoverflow.com/a/68794383/19870975
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

        accordion = ipywidgets.Accordion(children=accordion_content, selected_index=None, titles=columns)

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

    return (figure, slider_dict)


@beartype
def get_slider_thresholds(slider_dict: dict) -> dict:
    """Get thresholds from sliders.

    Parameters
    ----------
    slider_dict : dict
        Dictionary of sliders in the format 'slider_dict[column][group] = slider' or 'slider_dict[column] = slider' if no grouping.

    Returns
    -------
    dict
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

########################################################################################
# --------------------------- UpSet Plot Threshold Impacts --------------------------- #
########################################################################################


def _upset_select_cells(adata: sc.AnnData,
                        thresholds: dict[str, dict[str, dict[Literal["min", "max"], int | float]] | dict[Literal["min", "max"], int | float]],
                        groupby: Optional[str] = None) -> pd.DataFrame:
    """
    Select cells based on thresholds for UpSet Plot.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    thresholds : dict[str, dict[str, dict[Literal["min", "max"], int | float]] | dict[Literal["min", "max"], int | float]]
        Dictionary containing thresholds for each column. If groupby is given, thresholds are set per group.
    groupby : Optional[str], default None
        Name of the column in adata.obs to group cells by.

    Returns
    -------
    selection : pd.DataFrame
        DataFrame containing boolean values for each cell based on thresholds.
    """
    selection = {}
    # loop over all columns
    for column_name, values in thresholds.items():
        # loop over all groups
        if groupby:
            # initialize an array of False values
            accumulate_results = np.zeros(adata.obs.shape[0], dtype=bool)
            # loop over all samples
            for sample, cutoffs in values.items():
                # select cells based on the sample
                sample_selection = np.array(adata.obs[groupby] == sample)
                # select cells based on the cutoffs
                cutoff_selection = np.array(
                    (adata.obs[column_name] < cutoffs['min']) | (adata.obs[column_name] > cutoffs['max']))
                # add up the selected cells
                accumulate_results = accumulate_results + np.logical_and(sample_selection, cutoff_selection)
            # add the results to the selection
            selection[column_name] = accumulate_results

        else:
            # select cells based on the cutoffs
            selection[column_name] = (adata.obs[column_name] < values['min']) | (adata.obs[column_name] > values['max'])
    # convert to DataFrame
    selection = pd.DataFrame(selection)

    # reset index
    selection.reset_index(inplace=True, drop=True)

    return selection


def upset_plot_filter_impacts(adata: sc.AnnData,
                              thresholds: dict[str, dict[str, dict[Literal["min", "max"], int | float]] | dict[Literal["min", "max"], int | float]],
                              limit_combinations: Optional[int] = None,
                              groupby: Optional[int] = None) -> Optional[dict]:
    """
    Plot the impact of filtering cells based on thresholds in an UpSet Plot.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    thresholds : dict[str, dict[str, dict[Literal["min", "max"], int | float]] | dict[Literal["min", "max"], int | float]]
        Dictionary containing thresholds for each column. If groupby is given, thresholds are set per group.
    limit_combinations : Optional[int], default None
        Limit the number of combinations to show in the plot.
    groupby : Optional[str], default None
        Name of the column in adata.obs to group cells by.

    Returns
    -------
    plot_result : Optional[dict]
    """
    if len(thresholds) <= 1:
        logger.info("Skipping UpSet Plot as only one threshold is given.")

        return None

    selection = _upset_select_cells(adata, thresholds, groupby)

    # Number of variables
    n = len(selection.columns)

    # Generate all combinations of True/False
    raw_combinations = np.array(np.meshgrid(*[[False, True]] * n)).T.reshape(-1, n)

    # Sort the combinations first by the number of True values, then lexicographically
    sorted_combinations = np.array(sorted(raw_combinations, key=lambda x: (np.sum(x), list(x))))

    # exclude empty combinations
    mask = np.sum(sorted_combinations, axis=1) != 0
    combinations = sorted_combinations[mask]

    # make a dataframe
    combinations_df = pd.DataFrame(combinations, columns=selection.columns)

    results = []

    # loop over all combinations
    for _, combination in combinations_df.iterrows():
        # get a list of the column names selected
        metric_combination = list(selection.columns[combination])
        # initialize a results array for the combination with len n-cells
        single_result = np.zeros(selection.shape[0], dtype=bool)
        # loop over the selected combination
        for metric in metric_combination:
            # add up the selected cells
            single_result = single_result + selection[metric]
        # append result of the combination
        results.append(sum(single_result))
    # add to dataframe
    combinations_df['counts'] = results

    # limit combinations
    if limit_combinations:
        # select all combinations with a grade less or equal the limit
        limit_mask = np.array(np.sum(combinations_df[selection.columns], axis=1) <= limit_combinations)
        # always include the total counts
        limit_mask[-1] = True
        # index by the mask
        combinations_df = combinations_df[limit_mask]

    # set the combinations as index
    combinations_df.set_index(list(selection.columns), inplace=True)

    with warnings.catch_warnings():  # TODO remove when this is merged https://github.com/jnothman/UpSetPlot/pull/278
        warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.")

        # plot the UpSet Plot
        plot_result = upsetplot.plot(combinations_df['counts'], totals_plot_elements=0)

    plot_result["intersections"].set_ylabel("Cells Filtered")
    plt.show()

    return plot_result
