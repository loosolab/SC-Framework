"""General plotting functions for sctoolbox, e.g. general plots for wrappers, and saving and adding titles to figures."""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib_venn import venn2, venn3

from sctoolbox import settings


########################################################################################
# -------------------- General helper functions for plotting ------------------------- #
########################################################################################

def _save_figure(path, dpi=600):
    """Save the current figure to a file.

    Parameters
    ----------
    path : str
        Path to the file to be saved. NOTE: Uses the internal 'sctoolbox.settings.figure_dir' + 'sctoolbox.settings.figure_prefix' as prefix.
        Add the extension (e.g. .tiff) you want save your figure in to the end of the path, e.g., /some/path/plot.tiff.
        The lack of extension indicates the figure will be saved as .png.
    dpi : int, default 600
        Dots per inch. Higher value increases resolution.
    """

    # 'path' can be None if _save_figure was used within a plotting function, and the internal 'save' was "None".
    # This moves the checking to the _save_figure function rather than each plotting function.
    if path is not None:
        output_path = settings.full_figure_prefix + path
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")


def _make_square(ax):
    """Force a plot to be square using aspect ratio regardless of the x/y ranges."""

    xrange = np.diff(ax.get_xlim())[0]
    yrange = np.diff(ax.get_ylim())[0]

    aspect = xrange / yrange
    ax.set_aspect(aspect)


def _add_figure_title(axarr, title, y=1.3, fontsize=16):
    """Add a figure title to the top of a multi-axes figure.

    Parameters
    ----------
    axarr : `list` of `matplotlib.axes.Axes`
        List of axes to add the title to.
    title : `str`
        Title to add at the top of plot.
    y : `float`, optional (default: `1.3`)
        Vertical position of the title in relation to the content. Larger number moves the title further up.
    fontsize : `int`, optional (default: `16`)
        Font size of the title.

    Examples
    --------
    .. plot::
        :context: close-figs

        axes = sc.pl.umap(adata, color=["louvain", "condition"], show=False)
        pl.add_figure_title(axes, "UMAP plots", fontsize=20)
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


def _add_labels(data, x, y, label_col=None, ax=None, **kwargs) -> list:
    """Add labels to a scatter plot.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the coordinates of points to label.
    x : str
        Name of the column in data to use for x axis coordinates.
    y : str
        Name of the column in data to use for y axis coordinates.
    label_col : str, optional (default: `None`)
        Name of the column in data to use for labels. If `None`, the index of data is used.
    ax : matplotlib.axes.Axes, optional (default: `None`)
        Axis to plot on. If `None`, the current open figure axis is used.
    **kwargs : arguments
        Additional arguments to pass to matplotlib.axes.Axes.annotate.

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


def _scale_values(array, mini, maxi) -> np.ndarray:
    """Small utility to scale values in array to a given range.

    Parameters
    ----------
    array : np.ndarray
        Array to scale.
    mini : float
        Minimum value of the scale.
    maxi : float
        Maximum value of the scale.

    Returns
    -------
    np.ndarray
        Scaled array values.
    """
    val_range = array.max() - array.min()
    a = (array - array.min()) / val_range
    return a * (maxi - mini) + mini


def _plot_size_legend(ax, val_min, val_max, radius_min, radius_max, title):
    """Fill in an axis with a legend for the dotplot size scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot the legend on.
    val_min : float
        Minimum value of the scale.
    val_max : float
        Maximum value of the scale.
    radius_min : float
        Minimum radius of the dots.
    radius_max : float
        Maximum radius of the dots.
    title : str
        Title of the legend.
    """

    # Current issue: The sizes start at 0, which means there are dots for 0 values.
    # the majority of this code is from the scanpy dotplot function

    n_dots = 4
    radius_list = np.linspace(radius_min, radius_max, n_dots)
    value_list = np.linspace(val_min, val_max, n_dots)

    # plot size bar
    x_list = np.arange(n_dots) + 0.5
    x_list *= 3  # extend space between points

    circles = [plt.Circle((x, 0.5), radius=r) for x, r in zip(x_list, radius_list)]
    col = PatchCollection(circles, color="gray", edgecolor='gray')
    ax.add_collection(col)

    ax.set_xticks(x_list)
    labels = ["{:.1f}".format(v) for v in value_list]  # todo: make this more flexible with regards to integers
    ax.set_xticklabels(labels, fontsize=8)

    # remove y ticks and labels
    ax.tick_params(axis='y', left=False, labelleft=False, labelright=False)

    # remove surrounding lines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)

    pad = 0.15
    ax.set_ylim(0 - 2 * pad, 1 + pad)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 - radius_min - pad, x1 + radius_max + pad)
    ax.set_xlabel(title, fontsize=8)
    ax.xaxis.set_label_position('top')
    ax.set_aspect('equal')


def clustermap_dotplot(table, x, y, color, size, save=None, fillna=0, cmap="bwr", **kwargs) -> sns.clustermap:
    """Plot a heatmap with dots instead of cells which can contain the dimension of "size".

    Parameters
    ----------
    table : pandas.DataFrame
        Dataframe containing the data to plot.
    x : str
        Column in table to plot on the x-axis.
    y : str
        Column in table to plot on the y-axis.
    color : str
        Column in table to use for the color of the dots.
    size : str
        Column in table to use for the size of the dots.
    save : str, default None
        If given, the figure will be saved to this path.
    fillna : float, default 0
        Replace NaN with given value.
    cmap : str, default bwr
        Colormap of the plot.
    **kwargs : arguments
        Additional arguments to pass to seaborn.clustermap.

    Returns
    -------
    sns.clustermap
        Clustermap containing the plot.

    Examples
    --------
    .. plot::
        :context: close-figs

        import sctoolbox.plotting as pl
        import scanpy as sc

        table = sc.datasets.pbmc68k_reduced().obs.reset_index()[:10]

    .. plot::
        :context: close-figs

        pl.clustermap_dotplot(
            table=table,
            x="bulk_labels",
            y="index",
            color="n_genes",
            size="n_counts",
            cmap="viridis"
        )
    """

    # This code is very hacky
    # Major todo is to get better control of location of legends
    # automatic scaling of figsize
    # and to make the code more flexible, potentially using a class

    # Create pivots with colors/size
    # .fillna fixes NaN in table which caused cluster error
    color_pivot = pd.pivot(table, index=y, columns=x, values=color).fillna(fillna)
    size_pivot = pd.pivot(table, index=y, columns=x, values=size).fillna(fillna)

    # Plot clustermap of values
    g = sns.clustermap(color_pivot, yticklabels=True, cmap=cmap,
                       # vmin=color_min, vmax=color_max, #should be given as kwargs
                       figsize=(5, 12),
                       cbar_kws={'label': color, "orientation": "horizontal"},
                       **kwargs)

    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=7, rotation=45, ha="right")

    # Turn off existing heatmap
    g.ax_heatmap._children[0]._visible = False

    # Add dots on top of cells
    data_ordered = g.__dict__["data2d"]
    nrows, ncols = data_ordered.shape
    x, y = np.meshgrid(np.arange(0.5, ncols + 0.5, 1), np.arange(0.5, nrows + 0.5, 1))

    # todo: get control of the min/max values of the color scale
    color_mat = data_ordered.values
    if "vmin" in kwargs:
        color_min = kwargs["vmin"]
        color_mat[color_mat < color_min] = color_min

    if "vmax" in kwargs:
        color_max = kwargs["vmax"]
        color_mat[color_mat > color_max] = color_max

    size_ordered = size_pivot.loc[data_ordered.index, data_ordered.columns]
    size_mat = size_ordered.values
    radius_mat = _scale_values(size_mat, 0.05, 0.5)

    circles = [plt.Circle((j, i), radius=r) for r, j, i in zip(radius_mat.flat, x.flat, y.flat)]
    col = PatchCollection(circles, array=color_mat.flatten(), cmap=cmap)
    g.ax_heatmap.add_collection(col)

    # Adjust size of individual cells and dendrograms
    g.ax_heatmap.set_aspect('equal')

    f = plt.gcf()
    f.canvas.draw()

    # trans_data = g.ax_heatmap.transData
    # trans_data_inv = trans_data.inverted()

    dend_row_pos = g.ax_row_dendrogram.get_position()
    # dend_col_pos = g.ax_col_dendrogram.get_position()
    heatmap_pos = g.ax_heatmap.get_position()

    cell_width = heatmap_pos.height / data_ordered.shape[0]
    den_size = cell_width * 5
    pad = cell_width * 0.1

    # Resize dendrograms
    g.ax_row_dendrogram.set_position([heatmap_pos.x0 - den_size - pad, dend_row_pos.y0,
                                      den_size, dend_row_pos.height])
    g.ax_col_dendrogram.set_position([heatmap_pos.x0, heatmap_pos.y0 + heatmap_pos.height + pad,
                                      heatmap_pos.width, den_size])

    # TODO: get right bounds for y-axis labels to correctly place legends
    # texts = g.ax_heatmap.get_yticklabels()
    # bboxes = [t.get_window_extent(renderer=f.canvas.renderer) for t in texts]
    # right_bound = max([bbox.x1 for bbox in bboxes])

    # Move colorbar
    max_txt_width = cell_width * 25
    hm_pos = g.ax_heatmap.get_position()
    g.ax_cbar.set_position([hm_pos.x1 + max_txt_width, hm_pos.y0, cell_width * 20, cell_width])
    g.ax_cbar.set_xticklabels(g.ax_cbar.get_xticklabels(), fontsize=7)

    g.ax_cbar.set_xlabel(g.ax_cbar.get_xlabel(), fontsize=8)
    g.ax_cbar.xaxis.set_label_position('top')

    # Add size legend manually
    cbar_pos = g.ax_cbar.get_position()
    ax_size = f.add_axes([cbar_pos.x0, cbar_pos.y1 + cbar_pos.height * 5, cbar_pos.width, cell_width])
    _plot_size_legend(ax_size, np.min(size_mat), np.max(size_mat), np.min(radius_mat), np.max(radius_mat), title=size)

    # Add border to heatmap
    g.ax_heatmap.add_patch(Rectangle((0, 0), ncols, nrows, fill=False, edgecolor='grey', lw=1))

    # Save figure
    _save_figure(save)

    return g


########################################################################################
#                                       Barplot                                        #
########################################################################################

def bidirectional_barplot(df,
                          title=None,
                          colors=None,
                          figsize=None,
                          save=None) -> matplotlib.axes.Axes:
    """Plot a bidirectional barplot.

    The input is a dataframe with the following columns:
    - left_label
    - right_label
    - left_value
    - right_value

    Parameters
    ----------
    df : `pandas.DataFrame`
        Dataframe with the following columns: left_label, right_label, left_value, right_value.
    title : `str`, optional (default: `None`)
        Title of the plot.
    colors : `dict`, optional (default: `None`)
        Dictionary with label names as keys and colors as values.
    figsize : `tuple`, optional (default: `None`)
        Figure size.
    save : `str`, optional (default: `None`)
        If given, the figure will be saved to this path.

    Returns
    -------
    matplotlib.axes.Axes
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

def boxplot(dt, show_median=True, ax=None) -> matplotlib.axes.Axes:
    """Generate one plot containing one box per column. The median value is shown.

    Parameters
    ----------
    dt : pandas.DataFrame
        pandas datafame containing numerical values in every column.
    show_median : boolean, default True
        If True show median value as small box inside the boxplot.
    ax : matplotlib.axes.Axes, default None
        Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        containing boxplot for every column.

    Examples
    --------
    .. plot::
        :context: close-figs

        import sctoolbox.plotting as pl
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

    .. plot::
        :context: close-figs

        dt = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

    .. plot::
        :context: close-figs

        pl.boxplot(dt, show_median=True, ax=None)
        plt.show()
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        # TODO: check if ax is an ax object
        pass

    sns.boxplot(data=dt, ax=ax)

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


def violinplot(table, y, color_by=None, hlines=None, colors=None, ax=None, title=None, ylabel=True) -> matplotlib.axes.Axes:
    """Plot a violinplot with optional horizontal lines for each violin.

    Parameters
    ----------
    table : pandas.DataFrame
        Values to create the violins from.
    y : str
        Column name of table. Values that will be shown on y-axis.
    color_by : str, default None
        Column name of table. Used to color group violins.
    hlines : float/ list or dict of float/ list with color_by categories as keys, default None
        Define horizontal lines for each violin.
    colors : list of str, default None
        List of colors to use for violins.
    ax : matplotlib.axes.Axes, default None
        Axes object to draw the plot on. Otherwise use current axes.
    title : str, default None
        Title of the plot.
    ylabel : bool or str, default True
        Boolean if ylabel should be shown. Or str for custom ylabel.

    Returns
    -------
    matplotlib.axes.Axes :
        Object containing the violinplot.

    Raises
    ------
    ValueError
        If y or color_by is not a column name of table. Or if hlines is not a number or list of numbers for color_by=None.

    Examples
    --------
    .. plot::
        :context: close-figs

        import sctoolbox.plotting as pl
        import matplotlib.pyplot as plt
        import seaborn as sns

    .. plot::
        :context: close-figs

        table = sns.load_dataset("titanic")

    .. plot::
        :context: close-figs

        pl.violinplot(table, "age", color_by="class", hlines=None, colors=None, ax=None, title=None, ylabel=True)
        plt.show()
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
    plot = sns.violinplot(data=table, y=y, x=color_by, order=color_group_order, color=colors, ax=ax)

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

def plot_venn(groups_dict, title=None, save=None):
    """Plot a Venn diagram from a dictionary of 2-3 groups of lists.

    Parameters
    ----------
    groups_dict : `dict`
        A dictionary where the keys are group names (strings) and the values
        are lists of items belonging to that group (e.g. {'Group A': ['A', 'B', 'C'], ...}).
    title : `str`, optional (default: `None`)
        Title of the plot.
    save : `str`, optional (default: `None`)
        Filename to save the plot to.

    Raises
    ------
    ValueError
        If groups_dict is not a dictionary or number of groups is not 2 or 3.
    """
    # Check if input is dict
    if not isinstance(groups_dict, dict):
        s = "The 'groups_dict' variable must be a dictionary. "
        s += "Please ensure that you are passing a valid dictionary as input."
        raise ValueError(s)

    # Extract the lists of items from the dictionary and convert them to sets
    group_sets = [set(groups_dict[group]) for group in groups_dict]

    plt.figure()

    # Plot the Venn diagram using matplotlib_venn
    if len(group_sets) == 2:
        venn2(group_sets, set_labels=list(groups_dict.keys()))
    elif len(group_sets) == 3:
        venn3(group_sets, set_labels=list(groups_dict.keys()))
    else:
        raise ValueError("Only 2 or 3 groups are supported.")

    # Add a title to the plot
    if title is not None:
        plt.title(title)

    # Show the plot
    _save_figure(save)
