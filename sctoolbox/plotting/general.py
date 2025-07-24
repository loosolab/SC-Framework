"""General plotting functions for sctoolbox, e.g. general plots for wrappers, and saving and adding titles to figures."""

import pandas as pd
import numpy as np
import warnings

import seaborn as sns
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib_venn import venn2, venn3
import scipy.cluster.hierarchy as sciclust
import seaborn
from pathlib import Path

from beartype import beartype
from beartype.typing import Optional, Literal, Tuple, Union, Any, Dict, List
from numpy.typing import NDArray

from sctoolbox import settings
logger = settings.logger


########################################################################################
# -------------------- General helper functions for plotting ------------------------- #
########################################################################################

@beartype
def _save_figure(path: Optional[str],
                 dpi: Optional[int | float] = None,
                 report: bool = False,
                 max_pixle: int = 2**16,
                 **kwargs: Any) -> None:
    """Save the current figure to a file.

    Parameters
    ----------
    path : Optional[str]
        Path to the file to be saved. NOTE: Uses the internal 'sctoolbox.settings.figure_dir' + 'sctoolbox.settings.figure_prefix' as prefix.
        Add the extension (e.g. .tiff) you want save your figure in to the end of the path, e.g., /some/path/plot.tiff.
        The lack of extension indicates the figure will be saved as .png.
    dpi : Optional[int, float]
        Dots per inch. Higher value increases resolution. Uses either `sctoolbox.settings.dpi` or `sctoolbox.settings.report_dpi` if not set.
    report : bool, default False
        Set true to silently add plot to sctoolbox.settings.report_dir instead of 'figure_dir' + 'figure_prefix' (see above).
    max_pixle : int, default 2**16
        The maximum of pixles a figure can have in each direction. Figures exceeding this value will be resized to this maximum and a warning will be shown.
        2**16 is the maximum the jpeg-format can handle.
    **kwargs : Any
        Additional arguments to pass to matplotlib.pyplot.savefig.
    """

    savefig_kwargs = {"bbox_inches": "tight", "facecolor": "white"}  # defaults
    savefig_kwargs.update(kwargs)

    if dpi is None:
        if report:
            dpi = settings.report_dpi
        else:
            dpi = settings.dpi

    # 'path' can be None if _save_figure was used within a plotting function, and the internal 'save' was "None".
    # This moves the checking to the _save_figure function rather than each plotting function.
    if path is not None:
        # calculate the figure dimensions
        fig = plt.gcf()  # get current figure
        w, h = (int(s * dpi) for s in fig.get_size_inches())

        if w > max_pixle or h > max_pixle:
            warnings.warn(f"Image size of {w}x{h} pixels is too large. It must be less than {max_pixle} in each direction. Shrinking image...")

            if w > max_pixle:
                w = max_pixle - 1
            if h > max_pixle:
                h = max_pixle - 1
            fig.set_size_inches(w / dpi, h / dpi)

        # save either at report or standard figure location
        if not report:
            output_path = settings.full_figure_prefix + path
            logger.info(f"Saving figure to {output_path}")
        else:
            output_path = Path(settings.report_dir) / path
        plt.savefig(output_path, dpi=dpi, **savefig_kwargs)


@beartype
def _make_square(ax: Axes) -> None:
    """Force a plot to be square using aspect ratio regardless of the x/y ranges."""

    xrange = np.diff(ax.get_xlim())[0]
    yrange = np.diff(ax.get_ylim())[0]

    aspect = xrange / yrange
    ax.set_aspect(aspect)


@beartype
def _add_figure_title(axarr: list[Axes] | dict[str, Axes] | Axes | seaborn.matrix.ClusterGrid,
                      title: str,
                      y: float | int = 1.3,
                      fontsize: int = 16) -> None:
    """Add a figure title to the top of a multi-axes figure.

    Parameters
    ----------
    axarr : list[Axes] | dict[str, Axes] | Axes | seaborn.matrix.ClusterGrid
        List of axes to add the title to.
    title : str
        Title to add at the top of plot.
    y : float | int, default 1.3
        Vertical position of the title in relation to the content. Larger number moves the title further up.
    fontsize : int, default 16
        Font size of the title.

    Examples
    --------
    .. plot::
        :context: close-figs

        axes = sc.pl.umap(adata, color=["louvain", "condition"], show=False)
        pl.general._add_figure_title(axes, "UMAP plots", fontsize=20)
    """

    # If only one axes is passed, convert to list
    if type(axarr).__name__.startswith("Axes"):
        axarr = [axarr]

    try:
        axarr[0]
    except Exception:

        if isinstance(axarr, dict):
            ax_dict = axarr   # e.g. scanpy dotplot
        else:
            ax_dict = axarr.__dict__   # seaborn clustermap, etc.

        axarr = [ax_dict[key] for key, value in ax_dict.items() if type(value).__name__.startswith("Axes")]

    # Get figure
    fig = plt.gcf()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get bounding box of axes in relation to first axes
    trans_data_inv = axarr[0].transData.inverted()  # from display to data
    bbox_list = [ax.get_window_extent(renderer=renderer).transformed(trans_data_inv) for ax in axarr]

    # Find y/x positions based on bboxes
    ty = np.max([bbox.y1 for bbox in bbox_list])
    ty *= y

    xmin = np.min([bbox.x0 for bbox in bbox_list])
    xmax = np.max([bbox.x1 for bbox in bbox_list])
    tx = np.mean([xmin, xmax])

    # Add text
    _ = axarr[0].text(tx, ty, title, va="bottom", ha="center", fontsize=fontsize)


@beartype
def _add_labels(data: pd.DataFrame,
                x: str,
                y: str,
                label_col: Optional[str] = None,
                ax: Optional[Axes] = None,
                **kwargs: Any) -> list:
    """Add labels to a scatter plot.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the coordinates of points to label.
    x : str
        Name of the column in data to use for x axis coordinates.
    y : str
        Name of the column in data to use for y axis coordinates.
    label_col : str, default None
        Name of the column in data to use for labels. If `None`, the index of data is used.
    ax : Axes, default None
        Axis to plot on. If `None`, the current open figure axis is used.
    **kwargs : Any
        Additional arguments to pass to Axes.annotate.

    Returns
    -------
    list
        List of matplotlib.text.Annotation objects.
    """

    if ax is None:
        ax = plt.gca()

    x_coords = data[x]
    y_coords = data[y]
    labels = data.index if label_col is None else data[label_col]

    texts = []
    for i, label in enumerate(labels):

        text = ax.annotate(label, (x_coords[i], y_coords[i]), **kwargs)
        texts.append(text)

    # Adjust text positions
    # to-do

    return texts


#############################################################################
# ------------------------------ Dotplot ---------------------------------- #
#############################################################################


@beartype
def clustermap_dotplot(table: pd.DataFrame,
                       x: str,
                       y: str,
                       size: str,
                       hue: str,
                       cluster_on: Literal["hue", "size"] = "hue",
                       fillna: float | int = 0,
                       title: Optional[str] = None,
                       figsize: Optional[Tuple[int | float, int | float]] = None,
                       dend_height: float | int = 2,
                       dend_width: float | int = 2,
                       palette: str = "vlag",
                       x_rot: int = 45,
                       show_grid: bool = False,
                       save: Optional[str] = None,
                       report: Optional[str] = None,
                       **kwargs: Any) -> NDArray[Axes]:
    """
    Plot a heatmap with dots (instead of squares), which can contain the dimension of "size".

    Parameters
    ----------
    table : pd.DataFrame
        Table in long-format. Has to have at least four columns as given by x, y, size and hue.
    x : str
        Column in table to plot on the x-axis.
    y : str
        Column in table to plot on the y-axis.
    size : str
        Column in table to use for the size of the dots.
    hue : str
        Column in table to use for the color of the dots.
    cluster_on : Literal["hue", "size"], default hue
        Decide which values to use for creating the dendrograms. Either "hue" or "size".
    fillna : float | int, default 0
        Replace NaN with given value.
    title : Optional[str], default None
        Title of the dotplot.
    figsize : Optional[Tuple[int | float, int | float]], default None
        Figure size in inches. Default is estimated from the number of rows/columns (ncols/3, nrows/3).
    dend_height : float | int, default 2
        Height of the x-axis dendrogram in counts of row elements, e.g. 2 represents a height of 2 rows in the dotplot.
    dend_width : float | int, default 2
        Width of the y-axis dendrogram in counts of column elements, e.g. 2 represents a width of 2 columns in the dotplot.
    palette : str, default vlag
        Color palette for hue colors.
    x_rot : int, default 45
        Rotation of xticklabels in degrees.
    show_grid : bool, default False
        Show grid behind dots in plot.
    save : Optional[str], default None
        Save the figure to this path.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    **kwargs : Any
        Additional arguments to pass to seaborn.scatterplot.

    Returns
    -------
    NDArray[Axes]
        Array of Axes objects containing the dotplot and the dendrogram(s).

    Examples
    --------
    .. plot::
        :context: close-figs

        table = adata.obs.reset_index()[:10]

    .. plot::
        :context: close-figs

        pl.general.clustermap_dotplot(
            table=table,
            x="bulk_labels",
            y="index",
            hue="n_genes",
            size="n_counts",
            palette="viridis"
        )
    """

    table = table.copy()

    # long table to wide format for hue and size
    wide_hue = pd.pivot(data=table, index=y, columns=x, values=hue).fillna(fillna)
    wide_size = pd.pivot(data=table, index=y, columns=x, values=size).fillna(fillna)
    nrows, ncols = wide_hue.shape  # same shape as wide_size

    # decide what dendrograms are possible
    x_dend_possible = len((wide_hue if cluster_on == "hue" else wide_size).columns) > 1
    y_dend_possible = len(wide_hue if cluster_on == "hue" else wide_size) > 1

    # Set figsize automatically
    if figsize is None:
        figsize = (ncols / 3, nrows / 3)

    # Create figure
    fig, ax = plt.subplots(1, figsize=figsize)
    axes = [ax]

    # Prepare shape of dotplot
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(-0.5, nrows - 0.5)
    ax.set_xticks(np.arange(ncols))
    ax.set_aspect(1)

    # x-axis dendrogram
    if x_dend_possible:
        x_link = sciclust.linkage(wide_hue.T if cluster_on == "hue" else wide_size.T)

        # Plot dendrogram
        col_dend_ax = ax.inset_axes([0, 1, 1, dend_height / nrows])  # column dendrogram
        axes.append(col_dend_ax)
        x_dend = sciclust.dendrogram(x_link,
                                     orientation="top",
                                     labels=wide_hue.columns if cluster_on == "hue" else wide_size.columns,
                                     no_labels=True,
                                     link_color_func=lambda x: "black",  # disable cluster colors
                                     ax=col_dend_ax)

        col_dend_ax.axis("off")

        # order after dendrogram
        # (sharey parameter is bugged)
        # https://towardsdatascience.com/how-to-do-a-custom-sort-on-pandas-dataframe-ac18e7ea5320
        x_order = pd.CategoricalDtype(
            reversed(x_dend["ivl"]),
            ordered=True
        )
        table[x] = table[x].astype(x_order)

    # y-axis dendrogram
    if y_dend_possible:
        y_link = sciclust.linkage(wide_hue if cluster_on == "hue" else wide_size)

        # Plot dendrogram
        row_dend_ax = ax.inset_axes([1, 0, dend_width / ncols, 1])  # row dendrogram
        axes.append(row_dend_ax)
        y_dend = sciclust.dendrogram(y_link,
                                     orientation="right",
                                     labels=wide_hue.T.columns if cluster_on == "hue" else wide_size.T.columns,
                                     no_labels=True,
                                     link_color_func=lambda x: "black",  # disable cluster colors
                                     ax=row_dend_ax)

        row_dend_ax.axis("off")

        # order after dendrogram
        # (sharey parameter is bugged)
        # https://towardsdatascience.com/how-to-do-a-custom-sort-on-pandas-dataframe-ac18e7ea5320
        y_order = pd.CategoricalDtype(
            reversed(y_dend["ivl"]),
            ordered=True
        )
        table[y] = table[y].astype(y_order)

    # sort matrix according to the dendrogram orders
    table = table.sort_values([x, y])

    # Fill in axes with dotplot
    plot = sns.scatterplot(data=table,
                           y=y,
                           x=x,
                           size=size,
                           sizes=(10, 200),
                           hue=hue,
                           palette=palette,
                           ax=ax,
                           zorder=100,  # place points above grid
                           **kwargs
                           )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rot, ha="right" if x_rot != 0 else "center")

    # Move legend to right side
    x_anchor = 1 if nrows == 1 else 1 + dend_width / ncols
    sns.move_legend(plot, loc='upper left', bbox_to_anchor=(x_anchor, 1, 0, 0))

    # Show gridlines
    if show_grid:
        ax.grid()

    # Title above plot
    title_ax = col_dend_ax if x_dend_possible else ax
    title_ax.set_title(title)

    # Save figure
    _save_figure(save)

    if settings.report_dir and report:
        _save_figure(report, report=True)

    return np.array(axes)


########################################################################################
#                                       Barplot                                        #
########################################################################################

@beartype
def bidirectional_barplot(df: pd.DataFrame,
                          title: Optional[str] = None,
                          colors: Optional[dict[str, str]] = None,
                          figsize: Optional[Tuple[int | float, int | float]] = None,
                          save: Optional[str] = None) -> Axes:
    """Plot a bidirectional barplot.

    A vertical barplot where each position has one bar going left and one going right (bidirectional).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the following mandatory column names:
            - left_label
            - right_label
            - left_value
            - right_value
    title : Optional[str], default None
        Title of the plot.
    colors : Optional[dict[str, str]], default None
        Dictionary with label names as keys and colors as values.
    figsize : Optional[Tuple[int | float, int | float]], default None
        Figure size.
    save : Optional[str], default None
        If given, the figure will be saved to this path.

    Returns
    -------
    Axes
        Axes containing the plot.

    Raises
    ------
    KeyError
        If df does not contain the required columns.
    """

    # Check that df contains columns left/right_label and left/right value
    required_columns = ["left_label", "right_label", "left_value", "right_value"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column {col} not found in dataframe.")

    # Example data
    labels_left = df["left_label"].tolist()
    labels_right = df["right_label"].tolist()
    values_left = -np.abs(df["left_value"])
    values_right = df["right_value"]

    if colors is None:
        all_labels = list(set(labels_left + labels_right))
        colors = {label: sns.color_palette()[i] for i, label in enumerate(all_labels)}

    # Create figure and axis objects
    if figsize is None:
        figsize = (5, len(labels_left))  # 5 wide, n bars tall
    fig, ax = plt.subplots(figsize=figsize)

    # Set the position of the y-axis ticks
    n_bars = len(labels_left)
    yticks = np.arange(n_bars)[::-1]

    # Plot the positive values as blue bars
    right_colors = [colors[label] for label in labels_right]
    right_bars = ax.barh(yticks, values_right, color=right_colors)

    # Plot the negative values as red bars
    left_colors = [colors[label] for label in labels_left]
    left_bars = ax.barh(yticks, values_left, color=left_colors)

    # Set the x-axis limits to include both positive and negative values
    ax.set_xlim([min(values_left) * 1.1, max(values_right) * 1.1])

    # Add a vertical line at x=0 to indicate the zero point
    ax.axvline(x=0, color='k')

    # Add text labels and values to right bars
    for i, bar in enumerate(right_bars):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, " " + str(labels_right[i]), ha='left', va='center')  # adding a space before to ensure space between bars and labels
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(values_right[i]), ha='center', va='center')

    # Add text labels and values to left bars
    for i, bar in enumerate(left_bars):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, str(labels_left[i]) + " ", ha='right', va='center')
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(np.abs(values_left[i])), ha='center', va='center')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', 0))

    ax.set_yticks([])
    ax.set_yticklabels([])

    # Set the x-axis tick labels to be positive numbers
    ticks = ax.get_xticks().tolist()
    ax.set_xticks(ticks)  # prevent "FixedFormatter should only be used together with FixedLocator"
    ax.set_xticklabels([int(abs(tick)) for tick in ticks])

    # Add a legend
    # ax.legend(['Positive', 'Negative'], loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.legend(['Positive', 'Negative'], loc='lower right')

    if title is not None:
        ax.set_title(title)

    # Save figure
    _save_figure(save)

    return ax


########################################################################################
# -----------------------------  Boxplot / violinplot -------------------------------- #
########################################################################################

@beartype
def boxplot(dt: pd.DataFrame,
            show_median: bool = True,
            ax: Optional[Axes] = None,
            **kwargs: Any) -> Axes:
    """Generate one plot containing one box per column. The median value is shown.

    Parameters
    ----------
    dt : pd.DataFrame
        pandas datafame containing numerical values in every column.
    show_median : boolean, default True
        If True show median value as small box inside the boxplot.
    ax : Optional[Axes], default None
        Axes object to plot on. If None, a new figure is created.
    **kwargs : Any
        Additional arguments to pass to seaborn.boxplot.

    Returns
    -------
    Axes
        containing boxplot for every column.

    Examples
    --------
    .. plot::
        :context: close-figs

        import pandas as pd
        dt = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

    .. plot::
        :context: close-figs

        pl.general.boxplot(dt, show_median=True, ax=None)
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        # TODO: check if ax is an ax object
        pass

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message="iteritems is deprecated*")

        dt_melt = dt.melt()
        ax = sns.boxplot(data=dt_melt, x="variable", y="value", ax=ax, **kwargs)
        ax.set_xlabel("")
        ax.set_ylabel("")

    if show_median:
        # From:
        # https://stackoverflow.com/questions/49554139/boxplot-of-multiple-columns-of-a-pandas-dataframe-on-the-same-figure-seaborn
        lines = ax.get_lines()
        categories = ax.get_xticks()

        # Add median label
        for cat in categories:
            y = round(lines[4 + cat * 6].get_ydata()[0], 2)
            ax.text(cat, y, f'{y}', ha='center', va='center', fontweight='bold', size=10, color='white',
                    bbox=dict(facecolor='#445A64'))

    return ax


@beartype
def violinplot(table: pd.DataFrame,
               y: str,
               color_by: Optional[str] = None,
               hlines: Optional[Union[float | int,
                                      list[float | int],
                                      dict[str, Union[float | int, list[float | int]]]]] = None,
               colors: Optional[list[str]] = None,
               ax: Optional[Axes] = None,
               title: Optional[str] = None,
               ylabel: bool = True,
               **kwargs: Any) -> Axes:
    """Plot a violinplot with optional horizontal lines for each violin.

    Parameters
    ----------
    table : pd.DataFrame
        Values to create the violins from.
    y : str
        Column name of table. Values that will be shown on y-axis.
    color_by : Optional[str], default None
        Column name of table. Used to color group violins.
    hlines : Optional[Union[float | int, list[float | int],
                            dict[str, Union[float | int, list[float | int]]]]], default None
        Define horizontal lines for each violin.
    colors : Optional[list[str]], default None
        List of colors to use for violins.
    ax : Optional[Axes], default None
        Axes object to draw the plot on. Otherwise use current axes.
    title : Optional[str], default None
        Title of the plot.
    ylabel : bool | str, default True
        Boolean if ylabel should be shown. Or str for custom ylabel.
    **kwargs : Any
        Additional arguments to pass to seaborn.violinplot.

    Returns
    -------
    Axes
        Object containing the violinplot.

    Raises
    ------
    ValueError
        If y or color_by is not a column name of table. Or if hlines is not a number or list of numbers for color_by=None.

    Examples
    --------
    .. plot::
        :context: close-figs

        import seaborn as sns
        table = sns.load_dataset("titanic")

    .. plot::
        :context: close-figs

        pl.general.violinplot(table, "age", color_by="class", hlines=None, colors=None, ax=None, title=None, ylabel=True)
    """

    # check if valid column name
    if y not in table.columns:
        raise ValueError(f"{y} not found in column names of table! Use one of {list(table.columns)}.")

    # check if color_by is valid volumn name
    if color_by is not None and color_by not in table.columns:
        raise ValueError(f"Color grouping '{color_by}' not found in column names of table! Use one of {list(table.columns)}")

    # set violin order
    color_group_order = set(table[color_by]) if color_by is not None else color_by

    # hlines has to be number of list if color_by=None
    if hlines is not None and color_by is None and not isinstance(hlines, (list, tuple, int, float)):
        raise ValueError(f"Parameter hlines has to be number or list of numbers for color_by=None. Got type {type(hlines)}.")

    # check valid groups in hlines dict
    if isinstance(hlines, dict):
        invalid_keys = set(hlines.keys()) - set(table.columns)
        if invalid_keys:
            raise ValueError(f"Invalid dict keys in hlines parameter. Key(s) have to match table column names. Invalid keys: {invalid_keys}")

    # create violinplot
    plot = sns.violinplot(data=table, y=y, x=color_by, order=color_group_order, color=colors, ax=ax, **kwargs)

    # add horizontal lines
    if hlines:
        # add color_group_order placeholder
        if color_by is None:
            color_group_order = [None]

        # make iterable
        hlines = hlines if isinstance(hlines, (list, tuple, dict)) else [hlines]

        # make hlines dict
        hlines_dict = hlines if isinstance(hlines, dict) else {}

        # horizontal line length computation
        violin_width = 1 / len(color_group_order)
        line_length = violin_width - 2 * violin_width * 0.1  # subtract 10% padding
        half_length = line_length / 2

        # draw line(s) for each violin
        for i, violin_name in enumerate(color_group_order):
            violin_center = violin_width * (i + 1) - violin_width / 2

            # ensure iterable
            line_heights = hlines_dict.setdefault(violin_name, hlines)
            line_heights = line_heights if isinstance(line_heights, (list, tuple)) else [line_heights]
            for line_height in line_heights:
                # skip if invalid line_height
                if not isinstance(line_height, (int, float)):
                    continue

                # add to right axes
                tmp_ax = ax if ax else plot

                tmp_ax.axhline(y=line_height,
                               xmin=violin_center - half_length,
                               xmax=violin_center + half_length,
                               color="orange",
                               ls="dashed",
                               lw=3)

    # add title
    if title:
        plot.set(title=title)

    # adjust y-label
    if not ylabel:
        plot.set(ylabel=None)
    elif isinstance(ylabel, str):
        plot.set(ylabel=ylabel)

    # remove x-axis ticks if color_by=None
    if color_by is None:
        plot.tick_params(axis="x", which="both", bottom=False)

    return plot


########################################################################################
# --------------------------------  Venn diagrams ------------------------------------ #
########################################################################################

@beartype
def plot_venn(groups_dict: dict[str, list[Any]],
              title: Optional[str] = None,
              save: Optional[str] = None,
              **kwargs: Any) -> None:
    """Plot a Venn diagram from a dictionary of 2-3 groups of lists.

    Parameters
    ----------
    groups_dict : dict[str, list[Any]]
        A dictionary where the keys are group names (strings) and the values
        are lists of items belonging to that group (e.g. {'Group A': ['A', 'B', 'C'], ...}).
    title : Optional[str], default None
        Title of the plot.
    save : Optional[str], default None
        Filename to save the plot to.
    **kwargs : Any
        Additional arguments to pass to matplotlib_venn.venn2 or matplotlib_venn.venn3.

    Raises
    ------
    ValueError
        If number of groups in groups_dict is not 2 or 3.

    Examples
    --------
    .. plot::
        :context: close-figs

        venn2_example = { 'Group A': [1, 2, 3, 4],
                          'Group B': [3, 4, 5, 6]
                        }

    .. plot::
        :context: close-figs

        pl.general.plot_venn(venn2_example, "Simple Venn2 plot")

    .. plot::
        :context: close-figs

        venn3_example = { 'Fruits A': ['Lemon', 'Orange', 'Blueberry', 'Grapefruit'],
                          'Fruits B': ['Pineapple', 'Mango', 'Banana', 'Papaya', 'Blueberry', 'Strawberry'],
                          'Fruits C': ['Strawberry', 'Blueberry', 'Raspberry', 'Orange', 'Mango']
                        }

    .. plot::
        :context: close-figs

        pl.general.plot_venn(venn3_example, "Simple Venn3 plot")

    """

    # Extract the lists of items from the dictionary and convert them to sets
    group_sets = [set(groups_dict[group]) for group in groups_dict]

    plt.figure()

    # Plot the Venn diagram using matplotlib_venn
    if len(group_sets) == 2:
        venn2(group_sets, set_labels=list(groups_dict.keys()), **kwargs)
    elif len(group_sets) == 3:
        venn3(group_sets, set_labels=list(groups_dict.keys()), **kwargs)
    else:
        raise ValueError("Only 2 or 3 groups are supported.")

    # Add a title to the plot
    if title is not None:
        plt.title(title)

    # Show the plot
    _save_figure(save)


########################################################################################
# -------------------------------- Scatter plots ------------------------------------- #
########################################################################################

@beartype
def pairwise_scatter(table: pd.DataFrame,
                     columns: list[str],
                     thresholds: Optional[dict[str, dict[Literal["min", "max"], int | float]]] = None,
                     save: Optional[str] = None,
                     report: Optional[str] = None,
                     **kwargs: Any) -> NDArray[Axes]:
    """Plot a grid of scatterplot comparing column values pairwise.

    If thresholds are given, lines are drawn for each threshold and points outside of the thresholds are colored red.

    Parameters
    ----------
    table : pd.DataFrame
        Dataframe containing the data to plot.
    columns : list[str]
        List of column names in table to plot.
    thresholds : Optional[dict[str, dict[Literal["min", "max"], int | float]]], default None
        Dictionary containing thresholds for each column. Keys are column names and values are dictionaries with keys "min" and "max".
    save : Optional[str], default None
        If given, the figure will be saved to this path.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    **kwargs : Any
        Additional arguments to pass to Axes.scatter.

    Returns
    -------
    NDArray[Axes]
        Array of Axes objects.

    Raises
    ------
    ValueError
        1. If columns contains less than two columns.
        2. If one of the given columns is not a table column

    Examples
    --------
    .. plot::
        :context: close-figs

        columns = ["percent_mito", "n_counts", "S_score"]

        thresholds = {"n_counts": {"min": 2500, "max": 8000},
                      "percent_mito": {"max": 0.03},
                      "S_score": {"max": 0.5}}

        pl.general.pairwise_scatter(adata.obs, columns, thresholds=thresholds)
    """

    if len(columns) < 2:
        raise ValueError("'columns' must contain at least two columns to compare.")

    for col in columns:
        if col not in table.columns:
            raise ValueError(f"Column '{col}' not found in table.")

    if thresholds is None:
        thresholds = {}

    # Initialize plot
    fig, axarr = plt.subplots(nrows=len(columns), ncols=len(columns),
                              figsize=(len(columns) * 3, len(columns) * 3))

    # Fill in plots
    excluded_flag = False
    for i_row in range(len(columns)):  # iterate over rows
        for i_col in range(len(columns)):   # iterate over columns

            c_col, c_row = columns[i_col], columns[i_row]
            ax = axarr[i_row, i_col]

            if i_row == i_col:  # plot histogram
                sns.histplot(table[c_col], ax=ax, color="black")
                ax.set_xlabel("")  # labels are set afterwards
                ax.set_ylabel("")  # labels are set afterwards
            else:

                # Establish coloring using thresholds
                included = np.ones(len(table), dtype=bool)
                for col in [c_col, c_row]:
                    if col in thresholds:
                        included = included & (table[col] >= thresholds[col].get("min", table[col].min())) & (table[col] <= thresholds[col].get("max", table[col].max()))
                colors = np.where(included, "black", "red")

                ax.scatter(table[c_col], table[c_row], s=1, c=colors, **kwargs)  # x=columns, y=rows

                excluded_flag = excluded_flag or not np.all(included)  # set flag if any points are excluded

    # Plot threshold lines
    for i, col in enumerate(columns):
        if col in thresholds:
            for key in ["min", "max"]:
                if key in thresholds[col]:

                    # plot vertical lines in row
                    for ax in axarr[:, i]:
                        ax.axvline(thresholds[col][key], color="darkgrey", lw=1, linestyle="--")

                    # plot horizontal lines in scatterplots
                    scatter_idx = [i_col for i_col in range(len(columns)) if i_col != i]  # all but current column
                    for ax in axarr[i, scatter_idx]:
                        ax.axhline(thresholds[col][key], color="darkgrey", lw=1, linestyle="--")

    # Fix y-axis legends for first histogram
    ax = axarr[0, 0].twinx()  # create new axis for correct y-values
    ax.set_ylim(axarr[0, 1].get_ylim())
    ax.yaxis.set_label_position('left')
    ax.yaxis.set_ticks_position('left')
    axarr[0, 0].set_yticks([])  # remove original axis
    axarr[0, 0] = ax

    # Set labels
    for i, col in enumerate(columns):
        axarr[i, 0].set_ylabel(col)     # left column contains y labels
        axarr[-1, i].set_xlabel(col)    # bottom row contains x labels

    # Remove ticklabels from middle plots
    _ = [ax.axes.yaxis.set_ticklabels([]) for ax in axarr[:, 1:].flatten()]  # remove y ticklabels from all but first column
    _ = [ax.axes.xaxis.set_ticklabels([]) for ax in axarr[:-1, :].flatten()]  # remove x ticklabels from all but last row

    # Add legend if any points are excluded
    if excluded_flag:
        point = Line2D([0], [0], marker='o', markersize=np.sqrt(20), color='r', linestyle='None')
        axarr[0, -1].legend([point], ["Excluded"], loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

    plt.subplots_adjust(wspace=0.08, hspace=0.08)

    # Save plot
    _save_figure(save)

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True)

    return axarr


########################################################################################
# --------------------------------------- Table -------------------------------------- #
########################################################################################

@beartype
def plot_table(table: pd.DataFrame,
               ax: Optional[Axes] = None,
               save: Optional[str] = None,
               report: Optional[str] = None,
               save_kwargs: Dict = {},
               col_width: int | float | List[int | float] = 3,
               row_height: int | float = 0.625,
               row_colors: str | List[str] = ['#f1f1f2', 'white'],
               edge_color: str = 'white',
               bbox: Tuple[int | float, int | float, int | float, int | float] = (0, 0, 1, 1),
               index_color: str = '#40466e',
               show_index: bool = True,
               show_header: bool = True,
               fontsize: int = 14,
               crop: Optional[int] = 10,
               round: Optional[int] = None,
               **kwargs: Any) -> Axes:
    """
    Plot a pandas DataFrame.

    Template: https://stackoverflow.com/a/39358722

    Parameters
    ----------
    table : pd.DataFrame
        The table to plot.
    ax : Optional[Axes]
        Add the plot to this ax-object or create a new one.
    save : Optional[str]
        Path to save the plot. Uses `sctoolbox.settings.figure_dir`.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    save_kwargs : Dict, default {}
        Additional saving arguments. Will be used by `save` and `report`.
    col_width : int | float | List[int | float], default 3
        Width of each column in inches. Use a list to define individual column widths. The list has to be of length 'number of columns' + 'number of index columns'.
        Ignored if `ax` is used.
    row_height : int | float, default 0.625
        Height of each row in inches. Ignored if `ax` is used.
    row_colors : str | List[str], default ['#f1f1f2', 'white']
        The row background color(s). Multiple colors will be in alternating fashion.
    edge_color : str, default 'white'
        Color of the border of each cell.
    bbox : Tuple[int | float, int | float, int | float, int | float], default (0, 0, 1, 1)
        A matplotlib bounding box. Forwarded to `matplotlib.pyplot.table`. Defines the spacing and position of the table.
        Provide a Tuple of (xmin, ymin, width, height). See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.table.html
    index_color : str, default '#40466e'
        Background color for the column and row index cells.
    show_index : bool, default True
        Whether to show the row index.
    show_header : bool, default True
        Whether to show the column header.
    fontsize : int, default 14
        The table fontsize.
    crop : Optional[int], default 10
        Crop the table to the `crop / 2` top and bottom rows.
    round : Optional[int]
        The number of decimal places each number in the table should be rounded to.
    **kwargs
        Additional arguments are forwarded to `matplotlib.pyplot.table`

    Raises
    ------
    ValueError
        When the number of col_widths doesn't match the number of columns + indices.

    Returns
    -------
    Axes
        Object containing the plot.
    """
    table = table.copy()  # ensure not to change the original table

    index_num = table.index.nlevels if show_index else 0
    if isinstance(col_width, list) and len(col_width) != len(table.columns) + index_num:
        raise ValueError(f"The table has {len(table.columns) + index_num} column(s){'+index' if index_num else ''} but {len(col_width)} widths where given.")

    if round is not None:
        table = table.round(round)

    if crop and len(table) > crop:
        top = table.head(crop // 2)
        bottom = table.tail(crop // 2)

        # create row of '...'
        sep_row = pd.DataFrame([['...'] * len(table.columns)], columns=table.columns, index=['...'])
        sep_row.index.name = top.index.name  # NOTE: doesn't work with multi-index
        # combine top, a row of '...' and bottom entries to a truncated table
        table = pd.concat([top, sep_row, bottom])

    # if the table is empty but has an index
    # convert index to a normal column
    if len(table.columns) < 1:
        show_index = False
        table.reset_index(inplace=True)

    # set column widths
    if not isinstance(col_width, list):
        col_width = [1 / (len(table.columns) + index_num)] * (len(table.columns) + index_num)

    if ax is None:
        # compute figure size based on column and row width
        size = (sum(col_width),
                (len(table.index) + table.columns.nlevels) * row_height)

        _, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=table.values,
                         rowLabels=table.index if show_index else None,
                         colLabels=table.columns if show_header else None,
                         bbox=bbox,
                         colWidths=col_width,
                         **kwargs)

    # set fontsize
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(fontsize)

    # color cell background
    for (row, col), cell in mpl_table.get_celld().items():
        cell.set_edgecolor(edge_color)

        # col ids start to count at 0 but the index has negative numbers
        # row ids start to count at 0 and include the index (no negative numbers)
        if row < table.index.nlevels and show_header or col < 0:
            cell.set_text_props(weight='bold', color='w')  # TODO make header text options accessible
            cell.set_facecolor(index_color)
        else:
            cell.set_facecolor(row_colors[row % len(row_colors)])

    _save_figure(save, **save_kwargs)

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True, **save_kwargs)

    plt.close()

    return ax
