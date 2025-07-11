"""Funtions of different single cell embeddings e.g. UMAP, PCA, tSNE."""

import multiprocessing as mp
import warnings
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.stats
from scipy.sparse import issparse
import itertools

import seaborn as sns
from matplotlib import __version__ as mpl_version, rcParams
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.collections import PathCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly as po
import plotly.graph_objects as go

from numba import errors as numba_errors

from beartype import beartype
from beartype.typing import Literal, Tuple, Optional, Union, Any, List, Annotated, Callable
from beartype.vale import Is
from numpy.typing import NDArray

import sctoolbox.utils as utils
import sctoolbox.tools as tools
from sctoolbox.plotting.general import _save_figure, _make_square, boxplot
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


#############################################################################
#                                  Utilities                                #
#############################################################################

@beartype
def sc_colormap() -> ListedColormap:
    """Get a colormap with 0-count cells colored grey (to use for embeddings).

    Returns
    -------
    cmap : ListedColormap
        Colormap with 0-count cells colored grey.
    """

    # Custom colormap for single cells
    color_cmap = cm.get_cmap('Reds', 200)
    newcolors = color_cmap(np.linspace(0.2, 0.9, 200))
    newcolors[0, :] = colors.to_rgba("lightgrey")  # count 0 = grey
    sc_cmap = ListedColormap(newcolors)

    return sc_cmap


def grey_colormap() -> ListedColormap:
    """Get a colormap with grey-scale colors, but without white to still show cells.

    Returns
    -------
    cmap : ListedColormap
        Grey-scale colormap.
    """
    color_cmap = cm.get_cmap('Greys', 200)
    newcolors = color_cmap(np.linspace(0.2, 1, 200))
    cmap = ListedColormap(newcolors)

    return cmap


@deco.log_anndata
@beartype
def flip_embedding(adata: sc.AnnData, key: str = "X_umap", how: Literal["vertical", "horizontal"] = "vertical"):
    """Flip the embedding in adata.obsm[key] along the given axis.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    key : str, default "X_umap"
        Key in adata.obsm to flip.
    how : Literal["vertical", "horizontal"], default "vertical"
        Axis to flip along. Can be "vertical" (flips up/down) or "horizontal" (flips left/right).

    Raises
    ------
    KeyError
        If the given key is not found in adata.obsm.
    ValueError
        If the given 'how' is not supported.
    """

    if key not in adata.obsm:
        raise KeyError(f"The given key '{key}' cannot be found in adata.obsm. Please check the key value")

    if how == "vertical":
        adata.obsm[key][:, 1] = -adata.obsm[key][:, 1]
    elif how == "horizontal":
        adata.obsm[key][:, 0] = -adata.obsm[key][:, 0]
    else:
        raise ValueError("The given axis '{0}' is not supported. Please use 'vertical' or 'horizontal'.".format(how))


#####################################################################
# -------------------- UMAP / tSNE embeddings ----------------------#
#####################################################################

@beartype
def _add_contour(x: NDArray,
                 y: NDArray,
                 ax: Axes):
    """Add contour plot to a scatter plot.

    Parameters
    ----------
    x : NDArray
        x-coordinates of the scatter plot.
    y : NDArray
        y-coordinates of the scatter plot.
    ax : Axes
        Axis object to add the contour plot to.
    """

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Peform the kernel density estimate
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = scipy.stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, X.shape)

    # Contour plot
    ax.contour(X, Y, f, colors="black", linewidths=0.5)


@beartype
def _add_legend_ax(ax_obj: Axes, ax_label: str = "<legend>") -> Optional[Axes]:
    """
    Create a dedicated ax-object and move the legend of the given ax to it.

    Similar to how colorbars are handled.
    Template: https://joseph-long.com/writing/colorbars/

    Parameters
    ----------
    ax_obj : Axes
        The ax-object with the legend to move.
    ax_label : str, default '<legend>'
        The label of the legend-ax.

    Returns
    -------
    Optional[Axes]
        Either the newly created legend-ax or None if there is no legend within the provided ax.
    """
    handles, labels = ax_obj.get_legend_handles_labels()

    # exit if there is no suitable legend within the ax
    if not handles or not labels:
        return

    # get current ax
    last_axes = plt.gca()

    # add a new ax taking 10% of ax_obj space
    divider = make_axes_locatable(ax_obj)
    lax = divider.append_axes("right", size="10%", pad=0)

    lax.legend(
        handles=handles,
        labels=labels,
        frameon=False,  # same parameters as scanpy.pl.embedding
        loc="center left",
        bbox_to_anchor=(-0.9, 0.5),
        ncol=(1 if len(labels) <= 14 else 2 if len(labels) <= 30 else 3),
        handletextpad=0
    )

    # add label for identification and disable axis-lines
    lax.set_label(ax_label)
    lax.set_axis_off()

    # return to former ax
    plt.sca(last_axes)

    # remove former legend
    ax_obj.get_legend().remove()

    return lax


@beartype
def _binarize_expression(adata: sc.AnnData,
                         features: list[str],
                         threshold: Optional[float] = 0,
                         percentile_threshold: Optional[float] = None):
    """
    Binarize the expression of a list of features based on a threshold and store the results in adata.obs.

    The function updates adata.obs in-place with binary expression data for each feature.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    features : list[str]
        A list of feature names to be binarized.
    threshold : Optional[float], default 0
        The expression threshold for binarization. Only one of the threshold parameters may be given.
    percentile_threshold : Optional[float]
        The expression threshold as a percentile of the features expression. Only one of the threshold parameters may be given.

    Raises
    ------
    ValueError
        If the feature names cannot be found in adata.var_names.
        If "threshold" and "percentile_threshold" are both set

    """
    if threshold is not None and percentile_threshold is not None:
        raise ValueError("The usage of 'threshold' excludes the usage of 'percentile_threshold' and vice versa. Set one or both of the parameters to None.")

    # Check if all features are present in the adata object
    missing_features = [feature for feature in features if feature not in adata.var_names]
    if missing_features:
        raise ValueError(f"Features not found in adata.var_names: {', '.join(missing_features)}")

    for feature in features:
        feature_expr = adata[:, feature].X

        if not isinstance(feature_expr, np.ndarray):
            feature_expr = feature_expr.toarray()

        feature_expr = feature_expr.flatten()

        if percentile_threshold is not None:
            threshold = np.percentile(feature_expr, percentile_threshold)

        # Binarize the expression data
        binary_expr = np.where(feature_expr > threshold, 'expressed', 'not expressed')
        adata.obs[feature] = pd.Categorical(binary_expr)

    # add "_" postfix to adata.var.index to clarify that adata.obs should be used
    adata.var.index += "_"


@deco.log_anndata
@beartype
def plot_embedding(adata: sc.AnnData,
                   method: str = "umap",
                   color: Optional[list[str | None] | str] = None,
                   style: Literal["dots", "hexbin", "density"] = "dots",
                   show_borders: bool = False,
                   show_contour: bool = False,
                   show_count: bool = True,
                   show_title: bool = True,
                   hexbin_gridsize: int = 30,
                   shrink_colorbar: float | int = 0.3,
                   square: bool = True,
                   suptitle: Optional[str] = None,
                   save: Optional[str] = None,
                   report: Optional[str] = None,
                   **kwargs) -> NDArray[Axes]:
    """Plot a dimensionality reduction embedding e.g. UMAP or tSNE with different style options. This is a wrapper around scanpy.pl.embedding.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    method : str, default "umap"
        Dimensionality reduction method to use. Must be a key in adata.obsm, or a method available as "X_<method>" such as "umap", "tsne" or "pca".
    color : Optional[str | list[str]], default None
        Key for annotation of observations/cells or variables/genes.
    style : Literal["dots", "hexbin", "density"], default "dots"
        Style of the plot. Must be one of "dots", "hexbin" or "density".
    show_borders : bool, default False
        Whether to show borders around embedding plot. If False, the borders are removed and a small legend is added to the plot.
    show_contour : bool, default False
        Whether to show a contour plot on top of the plot.
    show_count : bool, default True
        Whether to show the number of cells in the plot.
    show_title : bool, default True
        Whether to show the titles of the plots. If False, the titles are removed and the names are added to the colorbar/legend instead.
    hexbin_gridsize : int, default 30
        Number of hexbins across plot - higher values give smaller bins. Only used if style="hexbin".
    shrink_colorbar : float | int, default 0.3
        Shrink the height of the colorbar by this factor.
    square : bool, default True
        Whether to make the plot square.
    suptitle : Optional[str], default None
        The title for the whole figure.
    save : Optional[str], default None
        Filename to save the figure.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    **kwargs : arguments
        Additional keyword arguments are passed to :func:`scanpy.pl.plot_embedding`.

    Returns
    -------
    axes : NDArray[Axes]
        Array of axis objects

    Raises
    ------
    KeyError
        If the given method is not found in adata.obsm.
    ValueError
        If the 'components' given is larger than the number of components in the embedding.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.embedding.plot_embedding(adata, color="louvain", legend_loc="on data")

    .. plot::
        :context: close-figs

        _ = pl.embedding.plot_embedding(adata, method="pca", color="n_genes", show_contour=True, show_title=False)

    .. plot::
        :context: close-figs

        _ = pl.embedding.plot_embedding(adata, color=['n_genes', 'HES4'], style="hexbin")

    .. plot::
        :context: close-figs

        _ = pl.embedding.plot_embedding(adata, method="pca", color=['n_genes', 'HES4'],
                              style="hexbin", components=["1,2", "2,3"], ncols=2)

    .. plot::
        :context: close-figs

        ax = pl.embedding.plot_embedding(adata, color=['n_genes', 'louvain'], style="density")
    """

    # Get key in obsm from method
    if method in adata.obsm:  # method is directly available in obsm
        obsm_key = method
    elif "X_" + method in adata.obsm:  # method is available as "X_<method>"
        obsm_key = "X_" + method
    else:
        raise KeyError(f"The given method '{method}' or 'X_{method}' cannot be found in adata.obsm. The available keys are: {list(adata.obsm.keys())}.")

    # ---- Plot embedding for chosen colors ---- #

    # get embedding dimensions if passed as a kwarg
    # otherwise use defalut dimensions 1 and 2
    n_components = adata.obsm[obsm_key].shape[1]
    args = locals()  # get all arguments passed to function
    kwargs = args.pop("kwargs")  # split args from kwargs dict
    if "components" in kwargs:
        dims = kwargs["components"]
        if type(dims) is str:
            if dims == "all":
                dims = ["{0},{1}".format(c[0], c[1]) for c in itertools.combinations(range(1, n_components + 1), 2)]  # "1,2", "1,3", "2,3" etc.
            else:
                dims = [dims]

        # Check that dims are valid
        for dim in dims:
            dim1, dim2 = [int(d.strip()) for d in dim.split(",")]
            if dim1 > n_components or dim2 > n_components:
                raise ValueError(f"The given component '{dim}' is larger than the number of components in '{obsm_key}' ({n_components}). Please adjust 'components'.")
    else:
        dims = ["1,2"]
    kwargs["components"] = dims  # overwrite components kwarg

    kwargs["color_map"] = kwargs.get("color_map", sc_colormap())  # set cmap to sc_colormap if not given
    parameters = {"color": color,
                  "basis": method,  # sc.pl.embedding can take either "umap" or "X_umap"
                  "show": False}
    if style != "dots":
        parameters["alpha"] = 0  # make dots transparent
    kwargs.update(parameters)

    axarr = sc.pl.embedding(adata, **kwargs)

    # add figure title
    if suptitle:
        axarr[0].get_figure().suptitle(suptitle, fontsize="x-large")

    # add dedicated legend ax to make it uniform with colorbar
    leg_ax = None
    if "ax" in kwargs:
        leg_ax = _add_legend_ax(axarr)

    # if only one axis is returned, convert to list
    if not isinstance(axarr, list):
        axarr = [axarr]
    if not isinstance(color, list):
        color = [color]

    # Duplicate colors/dimensions if needed
    if len(kwargs["components"]) > 1 or len(color) > 1:
        color_list = [color[i // len(kwargs["components"])] for i in range(len(axarr))]  # color1, color1, color2, color2, etc.
        components_list = kwargs["components"] * len(color)
    else:
        color_list = color
        components_list = kwargs["components"]

    # ---- Adjust style of individual plots ---- #
    for i, ax in enumerate(axarr):

        # Establish which color/dimensions are used for current plot
        ax_color = color_list[i]
        dim1, dim2 = [int(dim.strip()) for dim in components_list[i].split(",")]  # (1, 2)
        coordinates = adata.obsm[obsm_key][:, [dim1 - 1, dim2 - 1]]

        # Remove title
        if not show_title:
            ax.set_title("")

        # Set titles of legend / colorbar / plot
        legend = (leg_ax if leg_ax else ax).get_legend()
        local_axes = (kwargs["ax"] if "ax" in kwargs else ax).figure.axes  # list of all plot and colorbar axes in figure
        has_colorbar = False
        if legend is not None:  # legend of categorical variables
            if not show_title:
                legend.set_title(ax_color)
        else:  # legend of continuous variables (colorbar)
            if "ax" in kwargs:
                # assume that the colorbar is added at the end of the axis list
                cbar_ax = local_axes[-1]
            else:
                # assume it is added directly after the plot axis
                cbar_ax_idx = local_axes.index(ax) + 1  # colorbar is always right after plot
                cbar_ax_idx = min(cbar_ax_idx, len(local_axes) - 1)  # ensure that idx is within bounds
                cbar_ax = local_axes[cbar_ax_idx]

            if cbar_ax.get_label() == "<colorbar>":
                has_colorbar = True  # this ax has a colorbar

                if not show_title:
                    cbar_ax.set_title(ax_color)

        # Add additional style to plots
        if style != "dots":

            # Prepare color values
            if ax_color is None:
                color_values = None
            else:
                color_values = utils.adata.get_cell_values(adata, ax_color)

            # Determine colors to use
            cmap = kwargs["color_map"]
            cmap = rcParams["image.cmap"] if cmap is None else cmap  # if cmap is None, scanpy uses default cmap for matplotlib
            if color_values is None:
                cmap = grey_colormap()  # if no color values are given, use greyscale to show density

            # Plot hexbin/density style if chosen
            if style == "hexbin":

                # Ensure that color is continuous
                if ax_color is not None and has_colorbar is False:
                    raise ValueError(f"Hexbin style is only supported for continuous variables, and is not possible for the values found in '{ax_color}'. Please set 'style' to 'dots', 'density' or use a continuous variable.")

                # Plot hexbin
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                hb = ax.hexbin(coordinates[:, 0], y=coordinates[:, 1], C=color_values,
                               mincnt=1, gridsize=hexbin_gridsize, cmap=cmap)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                # Replace colorbar with hexbin values
                if has_colorbar:
                    ax.figure.colorbar(hb, ax=ax, cax=cbar_ax)

                # Set colorbar for number of cells if color is None
                if color_values is None:
                    ax.figure.colorbar(hb, ax=ax, label="Number of cells")
                    cbar_ax = ax.figure.axes[-1]
                    has_colorbar = True

                    # Move colorbar to the correct position in _localaxes
                    index = local_axes.index(ax)
                    local_axes.insert(index + 1, local_axes.pop())  # insert colorbar after plot

            elif style == "density":

                # remove NaN values
                if color_values is not None:
                    is_nan = pd.isna(color_values)  # numpy's isnan throws error for string array
                    color_values = color_values[~is_nan]
                    coordinates = coordinates[~is_nan]

                if ax_color is None:
                    has_colorbar = True  # even non-colored plots have colorbar with density of cells

                # Values are continous
                if has_colorbar:
                    if color_values is None:
                        sns.kdeplot(x=coordinates[:, 0], y=coordinates[:, 1], fill=True, ax=ax, cmap=cmap, thresh=0.01, cbar=True,
                                    cbar_kws={"label": "Cell density"})
                        cbar_ax = ax.figure.axes[-1]  # colorbar was added to last axis

                    else:
                        color_values_scaled = (color_values - color_values.min()) / (color_values.max() - color_values.min())  # scale to 0-1
                        sns.kdeplot(x=coordinates[:, 0], y=coordinates[:, 1], fill=True, weights=color_values_scaled,
                                    ax=ax, cmap=cmap, thresh=0.01, cbar=True, cbar_ax=cbar_ax, cbar_kws={"label": f"Cell density\n(weighted by {ax_color})"})

                else:  # values are categorical
                    cat2color = dict(zip(adata.obs[ax_color].cat.categories, adata.uns[ax_color + "_colors"]))

                    adata_subsets = utils.adata.get_adata_subsets(adata, groupby=ax_color)
                    for group, adata_sub in adata_subsets.items():
                        coordinates_sub = adata_sub.obsm[obsm_key][:, [dim1 - 1, dim2 - 1]]

                        # Plot kde in color from original plot
                        kde_color = cat2color[group]
                        collection_len_before = len(ax.collections)
                        custom_cmap = LinearSegmentedColormap.from_list(f'{group}_cmap', ['lightgrey', kde_color], N=256)
                        sns.kdeplot(x=coordinates_sub[:, 0], y=coordinates_sub[:, 1], fill=True, ax=ax, cmap=custom_cmap, thresh=0.01)

                        # Set alpha for each level (enables seeing overlapping groups; lowest level are most see-through)
                        n_obj_added = len(ax.collections) - collection_len_before
                        objects = ax.collections[-n_obj_added:]
                        alpha_list = np.linspace(0.2, 1, len(objects))
                        for i, obj in enumerate(objects):
                            obj.set_alpha(alpha_list[i])

        # Add contour to plot
        if show_contour:
            _add_contour(coordinates[:, 0], coordinates[:, 1], ax)

        # Remove borders and add small UMAP1/UMAP2 legend
        if show_borders is False:

            # Remove all spines (axes lines)
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Move x and y-labels to the start of axes
            label = ax.xaxis.get_label()
            label.set_horizontalalignment('left')
            x_lab_pos, y_lab_pos = label.get_position()
            label.set_position([0, y_lab_pos])

            label = ax.yaxis.get_label()
            label.set_horizontalalignment('left')
            x_lab_pos, y_lab_pos = label.get_position()
            label.set_position([x_lab_pos, 0])

            # Draw UMAP coordinate arrows
            ymin, ymax = ax.get_ylim()
            xmin, xmax = ax.get_xlim()
            yrange = ymax - ymin
            xrange = xmax - xmin
            arrow_len_y = yrange * 0.2
            arrow_len_x = xrange * 0.2

            ax.annotate("", xy=(xmin, ymin), xytext=(xmin, ymin + arrow_len_y), arrowprops=dict(arrowstyle="<-", shrinkB=0))  # UMAP2 / y-axis
            ax.annotate("", xy=(xmin, ymin), xytext=(xmin + arrow_len_x, ymin), arrowprops=dict(arrowstyle="<-", shrinkB=0))  # UMAP1 / x-axis

        # Add number of cells to plot
        if show_count:
            ax.text(0.02, 0.02, f"{adata.n_obs:,} cells",
                    transform=ax.transAxes,
                    horizontalalignment='left',
                    verticalalignment='bottom')

        # Adjust aspect ratio
        if square:
            _make_square(ax)

        # Final formatting of colorbar incl. shrink
        if has_colorbar:

            cbar = cbar_ax._colorbar

            # fix colorbar to same height as plot
            # https://joseph-long.com/writing/colorbars/
            if "ax" in kwargs:
                last_axes = plt.gca()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                # TODO shrink not working
                plt.colorbar(cbar.mappable, cax=cax, pad=0.01, aspect=30 * shrink_colorbar, shrink=shrink_colorbar, fraction=0.08, anchor=(0.0, 0.0))  # need to plot again to gain control of aspect ratio

                plt.sca(last_axes)  # return to correct ax
            else:
                plt.colorbar(cbar.mappable, ax=ax, pad=0.01, aspect=30 * shrink_colorbar, shrink=shrink_colorbar, fraction=0.08, anchor=(0.0, 0.0))  # need to plot again to gain control of aspect ratio

            new_cbar_ax = ax.figure.axes[-1]

            # Carry over title and ylabel
            new_cbar_ax.set_title(cbar_ax.get_title(), fontsize=10)
            new_cbar_ax.set_ylabel(cbar_ax.get_ylabel(), fontsize=10)

            # Set specific cbar style for density plots
            if style == "density":

                # Adjust colorbar to remove density values
                yticks = new_cbar_ax.get_yticks()
                new_cbar_ax.set_yticks([yticks[0], yticks[-1]])
                new_cbar_ax.set_yticklabels(["low", "high"])

            # Move colorbar to the correct position in local_axes
            local_axes = ax.figure.axes  # update list
            cbar_idx = local_axes.index(cbar_ax)
            local_axes[cbar_idx] = new_cbar_ax
            cbar_ax.remove()

    # Save figure
    _save_figure(save)

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True)

    return np.array(axarr)


@deco.log_anndata
@beartype
def feature_per_group(adata: sc.AnnData,
                      y: str,
                      x: Optional[Union[str, list[str]]] = None,
                      top_n: Optional[int] = None,
                      style: Literal["dots", "hexbin", "density"] = "hexbin",
                      marker_key: Optional[str] = "rank_genes_groups",
                      binarize_threshold: Optional[float] = None,
                      binarize_percentile_threshold: Optional[float] = None,
                      figsize: Optional[Tuple[int | float, int | float]] = None,
                      save: Optional[str] = None,
                      report: Optional[str] = None,
                      **kwargs) -> NDArray[Axes]:
    """
    Plot a grid of embeddings with rows/columns corresponding to adata.obs column(s).

    The first column shows the groups (one per row) given through (x -> adata.obs) and the following plots show the expression of the selected features (y or top_n).

    Parameters
    ----------
    adata : scanpy.AnnData
        AnnData used for plotting.
    y : str
        Column name of adata.obs. Column should contain categorical values (i.e. not numeric). If not will give a warning attempt to convert to categorical.
    x : Optional[Union[str, list[str]]]
        A list of features which will be displayed. Valid names are found in adata.var.index. "x" prohibits the usage of the "top_n" parameter.
    top_n : Optional[int]
        Use to display the top markers per group. "top_n" prohibits the usage of the "x" parameter.
        Expects a precomputed feature ranking e.g. using :func:`scanpy.tl.rank_genes_groups`.
    style : Literal["dots", "hexbin", "density"], default "dots"
        The plotting style of the embedding. This selects the style of columns two onward (first column is always "dots").
        If a binarize_* parameter is given, the style is fixed to "dots".
        "dots": Plot each given cell.
        "hexbin": Aggregate the cells into local hexagonal-shapes
        "density": Aggregate the cells using a kernel density estimation.
    marker_key : Optional[str], default 'rank_genes_groups'
        In case of "top_n" use this key to access the ranking information.
    binarize_threshold : Optional[float], default None
        Binarize the expression values, given the threshold. Only one of the binarize_* parameters may be given.
    binarize_percentile_threshold : Optional[float], default None
        Binarize the expression values, given a percentile. The percentile of a features expression is used as a threshold. Only one of the binarize_* parameters may be given.
    figsize : Optional[Tuple[int | float, int | float]], default None
        Figure size. Default is (4.8 * number of columns, 3.8 * number of rows).
    save : Optional[str], default None
        Filename to save the figure.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    **kwargs : arguments
        Additional keyword arguments are passed to :func:`sctoolbox.plotting.embedding.plot_embedding`.

    Raises
    ------
    ValueError
        "top_n" and "x" are both set or neither is set
        groups of "y" do not match with the "marker_key"

    Returns
    -------
    axes : NDArray[Axes]
        Array of axis objects
    """
    if x and top_n:
        raise ValueError("The usage of 'top_n' excludes the usage of 'x' and vice versa. Set one of the parameters to None.")
    elif not x and not top_n:
        raise ValueError("Set either 'top_n' or 'x'.")

    binarize = binarize_threshold is not None or binarize_percentile_threshold is not None

    # check column type
    if adata.obs[y].dtype.name != 'category':
        warnings.warn(f"Expected adata.obs['{y}'] to be of type 'category' got '{adata.obs[y].dtype.name}'. This may lead to unexpected behavior convert the coulmn-type to 'category' to ensure proper results.")

    # fetch the groups
    grps = set(adata.obs[y])

    if top_n:
        # get the top n markers for each group
        x = tools.marker_genes.get_rank_genes_tables(adata,
                                                     key=marker_key,
                                                     n_genes=top_n)

        # prepare dict
        x = {key: t["names"].tolist() for key, t in x.items()}

        if grps != set(x.keys()):
            raise ValueError(f"Groups found in adata.obs['{y}'] does not match to precomputed groups in adata.uns['{marker_key}']")

        ncol = top_n + 1
    else:
        # prepare custom features for plotting
        if not isinstance(x, list):
            x = [x]

        ncol = len(x) + 1

        x = {g: x for g in grps}

    if binarize:
        adata = adata.copy()  # _binarize_expression will change the adata, we want to keep the original, but plot the changed.
        features = list()
        # collect all feature names and extend all names in x
        for grp in grps:
            features += x[grp]
        _binarize_expression(adata, features, binarize_threshold, binarize_percentile_threshold)

    # create plot
    fig, axs = plt.subplots(nrows=len(grps),
                            ncols=ncol,
                            figsize=figsize if figsize else (4.8 * ncol, 3.8 * len(grps)))

    for i, grp in enumerate(grps):
        color_names = [y] + x[grp]
        for j in range(ncol):
            if j == 0:
                group_restriction = grp
            elif binarize:
                group_restriction = "expressed"
            else:
                group_restriction = None

            if j < len(color_names):
                plot_embedding(
                    adata=adata,
                    color=color_names[j],
                    style="dots" if j == 0 or binarize else style,
                    groups=group_restriction,
                    ax=axs[i][j],
                    **kwargs
                )
            else:
                # remove empty axis
                fig.delaxes(axs[i][j])

    # save figure
    _save_figure(save)

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True)

    return axs


@deco.log_anndata
@beartype
def agg_feature_embedding(adata: sc.AnnData, features: List, fname: str, keep_score: bool = False, fun: Callable = np.mean, fun_kwargs: dict = {"axis": 1}, report: Optional[str] = None, layer: Optional[str] = None, **kwargs) -> NDArray[Axes]:
    """
    Plot the embedding colored by an aggregated score based on the given set of features. E.g. a UMAP colored by the mean expression several provided genes.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object.
    features : List
        A list of features to aggregate. Uses the names in adata.var.index.
    fname : str
        Name of the selected feature group. Will be added as column to adata.obs (see keep_score) and used as plot title.
    keep_score : bool, default False
        Set to keep the aggregated feature score stored in adata.obs[fname].
    fun : Callable, default np.mean
        The aggregation function. Expects a numpy array with values to aggregate as first parameter. E.g.:
        numpy.sum, numpy.mean (re-creates the cellxgene gene set), numpy.median, etc.
    fun_kwargs : dict, default {"axis": 1}
        Additional arguments for the aggregation function.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    layer : Optional[str], default None
        Name of the adata layer used for the calculation. Defaults to `adata.X`.
    **kwargs : arguments
        Additional keyword arguments are passed to :func:`sctoolbox.plotting.embedding.plot_embedding`.

    Raises
    ------
    ValueError
        For features not found in adata.var.index or if fname already exists in adata.obs.columns.

    Returns
    -------
    axes : NDArray[Axes]
        Array of axis objects

    Examples
    --------
    .. plot::
        :context: close-figs

        # select the first three genes
        features = list(adata.var.index[:3])

        pl.embedding.agg_feature_embedding(adata=adata, features=features, fname=f"Mean expression of {features}")
    """
    try:
        # check for missing features
        missing = set(features) - set(adata.var.index)
        if missing:
            raise ValueError(f"Features {missing} are not found in adata.var.index!")

        # create subset of features
        subset = adata[:, features]

        # select layer
        if layer:
            matrix = subset.layers[layer].toarray()
        else:
            matrix = subset.X.toarray()

        # TODO https://github.com/scverse/scanpy/issues/532 support sc.tl.score_genes?

        # make sure to not overwrite an existing obs column
        if fname in adata.obs.columns:
            raise ValueError(f"{fname} already exists in adata.obs.columns. Select a different name or remove the column before running this function.")

        # calculate score and add as obs column
        adata.obs[fname] = np.array(fun(matrix, **fun_kwargs)).flatten()

        # plot
        return plot_embedding(adata, color=fname, report=report, **kwargs)
    finally:
        if not keep_score:
            adata.obs.drop(columns=[fname], errors="ignore", inplace=True)


@deco.log_anndata
@beartype
def search_umap_parameters(adata: sc.AnnData,
                           min_dist_range: Tuple[float | int, float | int, float | int] = (0.2, 0.9, 0.2),  # 0.2, 0.4, 0.6, 0.8
                           spread_range: Tuple[float | int, float | int, float | int] = (0.5, 2.0, 0.5),    # 0.5, 1.0, 1.5
                           color: Optional[str] = None,
                           n_components: int = 2,
                           threads: int = 4,
                           save: Optional[str] = None,
                           **kwargs: Any) -> NDArray:
    """Plot a grid of different combinations of min_dist and spread variables for UMAP plots.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    min_dist_range : Tuple[float | int, float | int, float | int], default: (0.2, 0.9, 0.2)
        Range of 'min_dist' parameter values to test. Must be a tuple in the form (min, max, step).
    spread_range : Tuple[float | int, float | int, float | int], default (0.5, 2.0, 0.5)
        Range of 'spread' parameter values to test. Must be a tuple in the form (min, max, step).
    color : Optional[str], default None
        Name of the column in adata.obs to color plots by. If None, plots are not colored.
    n_components : int, default 2
        Number of components in UMAP calculation.
    threads : int, default 4
        Number of threads to use for UMAP calculation.
    save : Optional[str], default None
        Path to save the figure to. If None, the figure is not saved.
    **kwargs : Any
        Additional keyword arguments are passed to :func:`scanpy.tl.umap`.

    Returns
    -------
    NDArray
        2D numpy array of axis objects

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.embedding.search_umap_parameters(adata, min_dist_range=(0.2, 0.9, 0.2),
                                            spread_range=(2.0, 3.0, 0.5),
                                            color="bulk_labels")
    """

    args = locals()  # get all arguments passed to function
    args["method"] = "umap"
    kwargs = args.pop("kwargs")  # split args from kwargs dict

    return _search_dim_red_parameters(**args, **kwargs)


@deco.log_anndata
@beartype
def search_tsne_parameters(adata: sc.AnnData,
                           perplexity_range: Tuple[int, int, int] = (30, 60, 10),
                           learning_rate_range: Tuple[int, int, int] = (600, 1000, 200),
                           color: Optional[str] = None,
                           threads: int = 4,
                           save: Optional[str] = None,
                           **kwargs: Any) -> NDArray:
    """Plot a grid of different combinations of perplexity and learning_rate variables for tSNE plots.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    perplexity_range : Tuple[int, int, int], default (30, 60, 10)
        tSNE parameter: Range of 'perplexity' parameter values to test. Must be a tuple in the form (min, max, step).
    learning_rate_range : Tuple[int, int, int], default (600, 1000, 200)
        tSNE parameter: Range of 'learning_rate' parameter values to test. Must be a tuple in the form (min, max, step).
    color : Optional[str], default None
        Name of the column in adata.obs to color plots by. If None, plots are not colored.
    threads : int, default 1
        The threads paramerter is currently not supported. Please leave at 1.
        This may be fixed in the future.
    save : Optional[str], default None (not saved)
        Path to save the figure to.
    **kwargs : Any
        Additional keyword arguments are passed to :func:`scanpy.tl.tsne`.

    Returns
    -------
    NDArray
        2D numpy array of axis objects

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.embedding.search_tsne_parameters(adata, perplexity_range=(30, 60, 10),
                                            learning_rate_range=(600, 1000, 200),
                                            color="bulk_labels")
    """

    args = locals()  # get all arguments passed to function
    args["method"] = "tsne"
    kwargs = args.pop("kwargs")

    return _search_dim_red_parameters(**args, **kwargs)


@beartype
def _search_dim_red_parameters(adata: sc.AnnData,
                               method: Literal["umap", "tsne"],
                               min_dist_range: Optional[Tuple[int | float, int | float, int | float]] = None,  # for UMAP
                               spread_range: Optional[Tuple[int | float, int | float, int | float]] = None,  # for UMAP
                               perplexity_range: Optional[Tuple[int, int, int]] = None,  # for tSNE
                               learning_rate_range: Optional[Tuple[int, int, int]] = None,  # for tSNE
                               color: Optional[str] = None,
                               threads: int = 4,
                               save: Optional[str] = None,
                               **kwargs: Any) -> NDArray:
    """Search different combinations of parameters for UMAP or tSNE and plot a grid of the embeddings.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    method : Literal["umap", "tsne"]
        Dimensionality reduction method to use. Must be either 'umap' or 'tsne'.
    min_dist_range : Optional[Tuple[int | float, int | float, int | float]], default None
        UMAP parameter: Range of 'min_dist' parameter values to test. Must be a tuple in the form (min, max, step).
    spread_range : Optional[Tuple[int | float, int | float, int | float]], default None
        UMAP parameter: Range of 'spread' parameter values to test. Must be a tuple in the form (min, max, step).
    perplexity_range : Optional[Tuple[int, int, int]], default None
        tSNE parameter: Range of 'perplexity' parameter values to test. Must be a tuple in the form (min, max, step).
    learning_rate_range : Optional[Tuple[int, int, int]], default None
        tSNE parameter: Range of 'learning_rate' parameter values to test. Must be a tuple in the form (min, max, step).
    color : Optional[str], default None
        Name of the column in adata.obs to color plots by. If None, plots are not colored.
    threads : int, default 4
        Number of threads to use for calculating embeddings. In case of UMAP, the embeddings will be calculated in parallel with each job using 1 thread.
        For tSNE, the embeddings are calculated serially, but each calculation uses 'threads' as 'n_jobs' within sc.tl.tsne.
    save : Optional[str], default None
        Path to save the figure to.
    **kwargs : Any
        Additional keyword arguments are passed to :func:`scanpy.tl.umap` or :func:`scanpy.tl.tsne`.

    Returns
    -------
    NDArray
        2D numpy array of axis objects
    """

    def get_loop_params(r):
        """Get parameters to loop over."""
        # Check validity of range parameters
        if len(r) != 4:
            raise ValueError(f"The parameter '{r[0]}' must be a tuple in the form (min, max, step)")
        if r[3] > r[2] - r[1]:
            raise ValueError(f"'step' of '{r[0]}' is larger than 'max' - 'min'. Please adjust.")

        return np.around(np.arange(r[1], r[2], r[3]), 2)

    # remove data to save memory
    adata = utils.adata.get_minimal_adata(adata)
    # Allows for all case variants of method parameter
    method = method.lower()

    if method == "umap":
        range_1 = ["min_dist_range"] + list(min_dist_range)
        range_2 = ["spread_range"] + list(spread_range)
    elif method == "tsne":
        range_1 = ["perplexity_range"] + list(perplexity_range)
        range_2 = ["learning_rate_range"] + list(learning_rate_range)

    # Get tool and plotting function
    tool_func = getattr(sc.tl, method)

    # Setup loop parameter
    loop_params = list()
    for r in [range_1, range_2]:
        loop_params.append(get_loop_params(r))

    # Should the functions be run in parallel?
    run_parallel = False
    if threads > 1 and method == "umap":
        run_parallel = True

    # Calculate umap/tsne for each combination of spread/dist
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=numba_errors.NumbaDeprecationWarning)  # numba warning for 0.59.0 (only for UMAP)
        warnings.filterwarnings("ignore", category=UserWarning, message="In previous versions of scanpy, calling tsne with n_jobs > 1 would use MulticoreTSNE.")

        if run_parallel:
            pool = mp.Pool(threads)
        else:
            pbar = utils.multiprocessing.get_pbar(len(loop_params[0]) * len(loop_params[1]), f"Computing {method.upper()}s")

        # Setup jobs
        jobs = {}
        for i, r2_param in enumerate(loop_params[1]):  # rows
            for j, r1_param in enumerate(loop_params[0]):  # columns
                kwds = {range_1[0].rsplit('_', 1)[0]: r1_param,
                        range_2[0].rsplit('_', 1)[0]: r2_param,
                        "copy": True}
                if method == "tsne":
                    kwds["n_jobs"] = threads
                kwds |= kwargs  # gives the option to overwrite e.g. n_jobs if given in kwargs

                logger.debug(f"Running '{method}' with kwds: {kwds}")

                if run_parallel:
                    job = pool.apply_async(tool_func, args=(adata, ), kwds=kwds)
                else:
                    job = tool_func(adata, **kwds)  # run the tool function one by one; returns an anndata object
                    pbar.update(1)
                jobs[(i, j)] = job

        if run_parallel:
            pool.close()
            utils.multiprocessing.monitor_jobs(jobs, f"Computing {method.upper()}s")
            pool.join()

    # Figure with rows=spread, cols=dist
    fig, axes = plt.subplots(len(loop_params[1]), len(loop_params[0]),
                             figsize=(4 * len(loop_params[0]), 4 * len(loop_params[1])))
    axes = np.array(axes).reshape((-1, 1)) if len(loop_params[0]) == 1 else axes  # reshape 1-column array
    axes = np.array(axes).reshape((1, -1)) if len(loop_params[1]) == 1 else axes  # reshape 1-row array

    # Fill in UMAPs
    for i, r2_param in enumerate(loop_params[1]):  # rows
        for j, r1_param in enumerate(loop_params[0]):  # columns

            if run_parallel:
                jobs[(i, j)] = jobs[(i, j)].get()

            # Add precalculated UMAP to adata
            adata.obsm[f"X_{method}"] = jobs[(i, j)].obsm[f"X_{method}"]

            logger.debug(f"Plotting {method} for row={r2_param} and col={r1_param} ({i*len(loop_params[0])+j+1}/{len(loop_params[0])*len(loop_params[1])})")

            # Set legend loc for last column
            if i == 0 and j == (len(loop_params[0]) - 1):
                legend_loc = "center left"
            else:
                legend_loc = "none"

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="No data for colormapping provided via 'c'*")
                sc.pl.embedding(adata, basis="X_" + method, color=color, title='', legend_loc=legend_loc, show=False, ax=axes[i, j])

            if j == 0:
                axes[i, j].set_ylabel(f"{range_2[0].rsplit('_', 1)[0]}: {r2_param}", fontsize=14)
            else:
                axes[i, j].set_ylabel("")

            if i == 0:
                axes[i, j].set_title(f"{range_1[0].rsplit('_', 1)[0]}: {r1_param}", fontsize=14)

            axes[i, j].set_xlabel("")

    plt.tight_layout()
    _save_figure(save)

    return axes


#######################################################################################
# -------------------------- Different group embeddings ------------------------------#
#######################################################################################

@deco.log_anndata
@beartype
def plot_group_embeddings(adata: sc.AnnData,
                          groupby: str,
                          col: Optional[str] = None,
                          embedding: Literal["umap", "tsne", "pca"] = "umap",
                          ncols: int = 4,
                          save: Optional[str] = None,
                          report: Optional[str] = None,
                          **kwargs: Any) -> NDArray:
    """
    Plot a grid of embeddings (UMAP/tSNE/PCA) per group of cells within 'groupby'.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    groupby : str
        Name of the column in adata.obs to group by.
    col : Optional[str], default None
        Set color for grouped plots. Using groupby if set to None.
        Set if numerical values should be shown, e.g. density. Set to None for categorical values.
    embedding : Literal["umap", "tsne", "pca"], default "umap"
        Embedding to plot. Must be one of "umap", "tsne", "pca".
    ncols : int, default 4
        Number of columns in the figure.
    save : Optional[str], default None
        Path to save the figure.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    **kwargs : Any
        Additional keyword arguments are passed to :func:`scanpy.pl.umap` or :func:`scanpy.pl.tsne` or :func:`scanpy.pl.pca`.

    Returns
    -------
    NDArray
        Flat numpy array of axis objects

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.embedding.plot_group_embeddings(adata, 'phase', embedding='umap', ncols=4)
    """

    adata = adata.copy()

    # Get categories
    groups = adata.obs[groupby].astype("category").cat.categories
    n_groups = len(groups)

    # Find out how many rows are needed
    ncols = min(ncols, n_groups)  # Make sure ncols is not larger than n_groups
    nrows = int(np.ceil(len(groups) / ncols))

    # Setup subplots
    fig, axarr = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axarr = np.array(axarr).reshape((-1, 1)) if ncols == 1 else axarr
    axarr = np.array(axarr).reshape((1, -1)) if nrows == 1 else axarr
    axes_list = axarr.flatten()
    n_plots = len(axes_list)

    # Plot UMAP/tSNE/pca per group
    for i, group in enumerate(groups):

        ax = axes_list[i]

        if col:
            adata.obs["color"] = np.where(adata.obs[groupby] == group, adata.obs[col], np.nan)
            color = "color"
        else:
            color = groupby

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message="Categorical.replace is deprecated")
            warnings.filterwarnings("ignore", category=FutureWarning, message="In a future version of pandas")
            warnings.filterwarnings("ignore", category=UserWarning, message="No data for colormapping provided via 'c'*")

            # Plot individual embedding
            if embedding == "umap":
                sc.pl.umap(adata, color=color, groups=group, ax=ax, show=False, legend_loc=None, **kwargs)
            elif embedding == "tsne":
                sc.pl.tsne(adata, color=color, groups=group, ax=ax, show=False, legend_loc=None, **kwargs)
            elif embedding == "pca":
                sc.pl.pca(adata, color=color, groups=group, ax=ax, show=False, legend_loc=None, **kwargs)

        ax.set_title(group)

    # Hide last empty plots
    n_empty = n_plots - n_groups
    if n_empty > 0:
        for ax in axes_list[-n_empty:]:
            ax.set_visible(False)

    # Save figure
    _save_figure(save)

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True)

    return axarr


@beartype
def compare_embeddings(adata_list: list[sc.AnnData],
                       var_list: list[str] | str,
                       embedding: Literal["umap", "tsne", "pca"] = "umap",
                       adata_names: Optional[list[str]] = None,
                       **kwargs: Any) -> NDArray:
    """Compare embeddings across different adata objects.

    Plots a grid of embeddings with the different adatas on the x-axis, and colored variables on the y-axis.

    Parameters
    ----------
    adata_list : list[sc.AnnData]
        List of AnnData objects to compare.
    var_list : list[str] | str
        List of variables to color in plot.
    embedding : Literal["umap", "tsne", "pca"], default "umap"
        Embedding to plot. Must be one of "umap", "tsne" or "pca".
    adata_names : Optional[list[str]], default None (adatas will be named adata_1, adata_2, etc.)
        List of names for the adata objects. Must be the same length as adata_list or None
    **kwargs : Any
        Additional arguments to pass to sc.pl.umap/sc.pl.tsne/sc.pl.pca.

    Returns
    -------
    NDArray
        2D numpy array of axis objects

    Raises
    ------
    ValueError
        If none of the variables in var_list are found in any of the adata objects.

    Examples
    --------
    .. plot::
        :context: close-figs

        import scanpy as sc

    .. plot::
        :context: close-figs

        adata1 = sc.datasets.pbmc68k_reduced()
        adata2 = sc.datasets.pbmc3k_processed()
        adata_list = [adata1, adata2]
        var_list = ['n_counts', 'n_cells']

    .. plot::
        :context: close-figs

        pl.embedding.compare_embeddings(adata_list, var_list)
    """

    embedding = embedding.lower()

    # Check the availability of vars in the adata objects
    all_vars = set()
    for adata in adata_list:
        all_vars.update(set(adata.var.index))
        all_vars.update(set(adata.obs.columns))

    # Subset var list to those available in any of the adata objects
    if isinstance(var_list, str):
        var_list = [var_list]

    not_found = set(var_list) - all_vars
    if len(not_found) == len(var_list):
        raise ValueError("None of the variables from var_list were found in the adata objects.")
    elif len(not_found) > 0:
        logger.warning(f"The following variables from var_list were not found in any of the adata objects: {list(not_found)}. These will be excluded.")

    var_list = [var for var in var_list if var in all_vars]

    # Setup plot grid
    n_adata = len(adata_list)
    n_var = len(var_list)
    fig, axes = plt.subplots(n_var, n_adata, figsize=(4 * n_adata, 4 * n_var))

    # Fix indexing
    n_cols = n_adata
    n_rows = n_var
    axes = np.array(axes).reshape((-1, 1)) if n_cols == 1 else axes  # Fix indexing for one column figures
    axes = np.array(axes).reshape((1, -1)) if n_rows == 1 else axes  # Fix indexing for one row figures

    if adata_names is None:
        adata_names = [f"adata_{n+1}" for n in range(len(adata_list))]

    # code for coloring single cell expressions?
    # import matplotlib.colors as clr
    # cmap = clr.LinearSegmentedColormap.from_list('custom umap', ['#f2f2f2', '#ff4500'], N=256)

    for i, adata in enumerate(adata_list):

        # Available vars for this adata
        available = set(adata.var.index)
        available.update(set(adata.obs.columns))

        for j, var in enumerate(var_list):

            # Check if var is available for this specific adata
            if var not in available:
                print(f"Variable '{var}' was not found in adata object '{adata_names[i]}'. Skipping coloring.")
                var = None

            if embedding == "umap":
                sc.pl.umap(adata, color=var, show=False, ax=axes[j, i], **kwargs)
            elif embedding == "tsne":
                sc.pl.tsne(adata, color=var, show=False, ax=axes[j, i], **kwargs)
            elif embedding == "pca":
                sc.pl.pca(adata, color=var, show=False, ax=axes[j, i], **kwargs)

            # Set y-axis label
            if i == 0:
                axes[j, i].set_ylabel(var)
            else:
                axes[j, i].set_ylabel("")

            # Set title
            if j == 0:
                axes[j, i].set_title(list(adata_names)[i])
            else:
                axes[j, i].set_title("")

            axes[j, i].set_xlabel("")

            _make_square(axes[j, i])

    fig.tight_layout()

    for ax in fig.get_axes():
        if hasattr(ax, 'collections'):
            for collection in ax.collections:
                if isinstance(collection, PathCollection):
                    if collection.colorbar is not None:
                        colorbar = collection.colorbar
                        bbox = ax.get_position()
                        colorbar.ax.set_position([bbox.x1 + 0.01, bbox.y0, 0.02, bbox.height])

    return axes


#######################################################################################
# ---------------------------------- 3D UMAP -----------------------------------------#
#######################################################################################

@beartype
def _get_3d_dotsize(n: int) -> int:
    """Get the optimal plotting dotsize for a given number of points."""
    if n < 1000:
        return 12
    elif n < 10000:
        return 8
    else:
        return 3


@deco.log_anndata
@beartype
def plot_3D_UMAP(adata: sc.AnnData,
                 color: str,
                 save: str,
                 **kwargs: Any) -> None:
    """Save 3D UMAP plot to a html file.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    color : str
        Variable to color in plot. Must be a column in adata.obs or an index in adata.var.
    save : str
        Save prefix. Plot will be saved to <save>.html.
    **kwargs : Any
        Additional keyword arguments are passed to :func:`plotly.graph_objects.Scatter3d`.

    Raises
    ------
    KeyError
        If the given 'color' attribute was not found in adata.obs columns or adata.var index.

    Examples
    --------
    .. plot::
        :context: close-figs

        min_dist = 0.3
        spread = 2.5
        sc.tl.umap(adata, min_dist=min_dist, spread=spread, n_components=3)

    .. plot::
        :context: close-figs

        pl.embedding.plot_3D_UMAP(adata, color="louvain", save="my3d_umap")

    This will create an .html-file with the interactive 3D UMAP: :download:`my3d_umap.html <my3d_umap.html>`
    """

    n_cells = len(adata.obs)
    size = _get_3d_dotsize(n_cells)

    # Get coordinates
    coordinates = adata.obsm['X_umap'][:, :3]
    df = pd.DataFrame(coordinates)
    df.columns = ["x", "y", "z"]

    # Create plot
    po.offline.init_notebook_mode(connected=True)  # prints a dict when not run in notebook
    fig = go.Figure()

    # Plot per group in obs
    if color in adata.obs.columns and isinstance(adata.obs[color].iloc[0], str):

        df["category"] = adata.obs[color].values  # color should be interpreted as a categorical variable
        categories = df["category"].unique()
        n_groups = len(categories)
        color_list = sns.color_palette("Set1", n_groups)
        color_list = list(map(colors.to_hex, color_list))  # convert to hex

        for i, name in enumerate(categories):
            df_sub = df[df['category'] == name]

            go_plot = go.Scatter3d(x=df_sub['x'],
                                   y=df_sub['y'],
                                   z=df_sub['z'],
                                   name=name,
                                   hovertemplate=name + '<br>(' + str(len(df_sub)) + ' cells)<extra></extra>',
                                   showlegend=True,
                                   mode='markers',
                                   marker=dict(size=size,
                                               color=[color_list[i] for _ in range(len(df_sub))],
                                               opacity=0.8),
                                   **kwargs)
            fig.add_trace(go_plot)

    # Plot a gene expression
    else:

        # Color is a value column in obs
        if color in adata.obs.columns:
            color_values = adata.obs[color]

        # color is a gene
        elif color in adata.var.index:
            color_idx = list(adata.var.index).index(color)
            color_values = adata.X[:, color_idx]
            color_values = color_values.todense().A1 if issparse(color_values) else color_values

        # color was not found
        else:
            raise KeyError("The given 'color' attribute was not found in adata.obs columns or adata.var index.")

        # Plot 3d with colorbar
        go_plot = go.Scatter3d(x=df['x'],
                               y=df['y'],
                               z=df['z'],
                               name='Expression of ' + color,
                               hovertemplate='Expression of ' + color + '<br>(' + str(len(df)) + ' cells)<extra></extra>',
                               showlegend=True,
                               mode='markers',
                               marker=dict(size=size,
                                           color=color_values,
                                           colorscale='Viridis',
                                           colorbar=dict(thickness=20, lenmode='fraction', len=0.75),
                                           opacity=0.8))
        fig.add_trace(go_plot)

    # Finalize plot
    fig.update_layout(legend={'itemsizing': 'constant'}, legend_title_text='<br><br>' + color)
    fig.update_scenes(xaxis=dict(showspikes=False),
                      yaxis=dict(showspikes=False),
                      zaxis=dict(showspikes=False))
    fig.update_layout(scene=dict(xaxis_title='UMAP1',
                                 yaxis_title='UMAP2',
                                 zaxis_title='UMAP3'))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # Save to file
    if isinstance(save, str):
        path = settings.full_figure_prefix + save + ".html"
        fig.write_html(path)
        logger.info(f"Plot written to '{path}'")

    else:
        logger.error("Please specify save parameter for html export")


@deco.log_anndata
@beartype
def umap_marker_overview(adata: sc.AnnData,
                         markers: list[str] | str,
                         ncols: int = 3,
                         figsize: Optional[Tuple[int, int]] = None,
                         save: Optional[str] = None,
                         cbar_label: str = "Relative expr.",
                         **kwargs: Any) -> list:
    """Plot a pretty grid of UMAPs with marker gene expression.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    markers : list[str] | str
        List of markers or singel marker
    ncols : int, default 3
        Number of columns in grid.
    figsize : Optional[Tuple[int, int]], default None
        Tuple of figure size.
    save : Optional[str], default None
        If not None save plot under given name.
    cbar_label : str, default "Relative expr."
        Colorbar label
    **kwargs : Any
        Additional parameter for scanpy.pl.umap()

    Returns
    -------
    list
        List of axis objects
    """

    if isinstance(markers, str):
        markers = [markers]

    # Find out how many rows we need
    n_markers = len(markers)
    nrows = int(np.ceil(n_markers / ncols))

    if figsize is None:
        figsize = (ncols * 3, nrows * 3)
    fig, axarr = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    params = {"cmap": sc_colormap(),
              "ncols": ncols,
              "frameon": False}
    params.update(**kwargs)

    axes_list = axarr.flatten()

    for i, marker in enumerate(markers):
        ax = axes_list[i]

        _ = sc.pl.umap(adata,
                       color=marker,
                       show=False,
                       colorbar_loc=None,
                       ax=ax,
                       **params)

        # Add title to upper left corner
        # ax.text(0, 1, marker, transform=ax.transAxes,
        #                      horizontalalignment='left',
        #                      verticalalignment='top')

    # Hide axes not used
    for ax in axes_list[len(markers):]:
        ax.set_visible(False)

    axes_list = axes_list[:len(markers)]

    # Add colorbar next to the last plot
    cax = fig.add_axes([0, 0, 1, 1])  # dummy size, will be resized
    lastax_pos = axes_list[len(markers) - 1].get_position()  # get the position of the last axis
    newpos = [lastax_pos.x1 * 1.1, lastax_pos.y0, lastax_pos.width * 0.1, lastax_pos.height * 0.5]
    cax.set_position(newpos)  # set a new position

    cbar = plt.colorbar(cm.ScalarMappable(cmap=params["cmap"]), cax=cax, label=cbar_label)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)  # remove border of colorbar

    # Make plots square
    for ax in axes_list:
        _make_square(ax)

    # Save figure if chosen
    _save_figure(save)

    return list(axes_list)


# See https://github.com/beartype/beartype/issues/347
_VALID_PLOTS = frozenset(("UMAP", "tSNE", "PCA", "PCA-var", "LISI"))

ListOfValidPlots = Annotated[List[Literal["UMAP", "tSNE", "PCA", "PCA-var", "LISI"]], Is[
    lambda lst: all(item in _VALID_PLOTS for item in lst)]]


@beartype
def anndata_overview(adatas: dict[str, sc.AnnData],
                     color_by: str | list[str],
                     plots: Union[ListOfValidPlots,
                                  Literal["UMAP", "tSNE", "PCA", "PCA-var", "LISI"]] = ["PCA", "PCA-var", "UMAP", "LISI"],
                     figsize: Optional[Tuple[int, int]] = None,
                     max_clusters: int = 20,
                     output: Optional[str] = None,
                     dpi: int = 300,
                     report: Optional[str] = None,
                     **kwargs: Any) -> NDArray[Axes]:
    """Create a multipanel plot comparing PCA/UMAP/tSNE/(...) plots for different adata objects.

    Parameters
    ----------
    adatas : dict[str, sc.AnnData]
        Dict containing an anndata object for each batch correction method as values. Keys are the name of the respective method.
        E.g.: {"bbknn": anndata}
    color_by : str | list[str]
        Name of the .obs column to use for coloring in applicable plots (e.g. for UMAP or PCA).
    plots : Union[list[Literal["UMAP", "tSNE", "PCA", "PCA-var", "LISI"]],
            Literal["UMAP", "tSNE", "PCA", "PCA-var", "LISI"]], default ["PCA", "PCA-var", "UMAP", "LISI"]
        Decide which plots should be created. Options are ["UMAP", "tSNE", "PCA", "PCA-var", "LISI"]
        Note: List order is forwarded to plot.
        - UMAP: Plots the UMAP embedding of the data.
        - tSNE: Plots the tSNE embedding of the data.
        - PCA: Plots the PCA embedding of the data.
        - PCA-var: Plots the variance explained by each PCA component.
        - LISI: Plots the distribution of any "LISI_score*" scores available in adata.obs
    figsize : Optional[Tuple[int, int]], default None
        Size of the plot in inch. Defaults to automatic size based on number of columns/rows.
    max_clusters : int, default 20
        Maximum number of clusters to show in legend.
    output : Optional[str], default None
        Path to plot output file.
    dpi : int, default 300
        Dots per inch for output
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    **kwargs : Any
        Additional keyword arguments are passed to :func:`scanpy.pl.umap`, :func:`scanpy.pl.tsne` or :func:`scanpy.pl.pca`.

    Returns
    -------
    axes : NDArray[Axes]
        Array of Axes objects created by matplotlib.

    Raises
    ------
    ValueError
        If any of the adatas is not of type anndata.AnnData.

    Examples
    --------
    .. plot::
        :context: close-figs

        adatas = {}  # dictionary of adata objects
        adatas["standard"] = adata
        adatas["parameter1"] = sc.tl.umap(adata, min_dist=1, copy=True)
        adatas["parameter2"] = sc.tl.umap(adata, min_dist=2, copy=True)

        pl.embedding.anndata_overview(adatas, color_by="louvain", plots=["PCA", "PCA-var", "UMAP"])
    """

    if not isinstance(color_by, list):
        color_by = [color_by]

    if not isinstance(plots, list):
        plots = [plots]

    # ---- helper functions ---- #
    def annotate_row(ax, plot_type):
        """Annotate row in figure."""
        # https://stackoverflow.com/a/25814386
        ax.annotate(plot_type,
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label,
                    textcoords='offset points',
                    size=ax.title._fontproperties._size * 1.2,  # increase title fontsize
                    horizontalalignment='right',
                    verticalalignment='center',
                    fontweight='bold')

    # ---- checks ---- #
    # dict contains only anndata
    wrong_type = {k: type(v) for k, v in adatas.items() if not isinstance(v, sc.AnnData)}
    if wrong_type:
        raise ValueError(f"All items in 'adatas' parameter have to be of type AnnData. Found: {wrong_type}")

    # check if color_by exists in anndata.obs
    for color_group in color_by:
        for name, adata in adatas.items():
            if color_group not in adata.obs.columns and color_group not in adata.var.index:
                raise ValueError(f"Couldn't find column '{color_group}' in the adata.obs or adata.var for '{name}'")

    # ---- plotting ---- #
    # setup subplot structure
    row_count = {"PCA-var": 1, "LISI": 1}  # all other plots count for len(color_by)
    rows = sum([row_count.get(plot, len(color_by)) for plot in plots])  # the number of rows in output plot
    cols = len(adatas)
    figsize = figsize if figsize is not None else (2 + cols * 4, rows * 4)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)  # , constrained_layout=True)
    axs = axs.flatten() if rows > 1 or cols > 1 else np.array([axs])  # flatten to 1d array per row

    # Fill in plots for every adata across plot type and color_by
    ax_idx = 0
    LISI_axes = []
    for plot_type in plots:
        for color in color_by:

            # Iterate over adatas to find all possible categories for 'color'
            categories = []
            for adata in adatas.values():
                if color in adata.obs.columns:  # color can also be an index in var
                    categories += list(adata.obs[color].unique())
            categories = sorted(list(set(categories)))

            # Create color palette equal for all columns
            if len(categories) > 0:
                colors = sns.color_palette("tab10", len(categories))
                color_dict = dict(zip(categories, colors))
            else:
                color_dict = None  # use default color palette

            # Plot for each adata (one row)
            for i, (name, adata) in enumerate(adatas.items()):

                ax = axs[ax_idx]

                # Only show legend for the last column
                if i == len(adatas) - 1:
                    legend_loc = "right margin"
                    # Disable colorbar for continuous values (will be re-added later)
                    colorbar_loc = "right" if color in adata.obs.select_dtypes(exclude="number").columns else None
                else:
                    legend_loc = "none"
                    colorbar_loc = None

                # add row label to first plot
                if i == 0:
                    annotate_row(ax, plot_type)

                # Collect options for plotting
                embedding_kwargs = {"color": color,
                                    "palette": color_dict,  # only used for categorical color
                                    "title": "",
                                    "legend_loc": legend_loc,
                                    "colorbar_loc": colorbar_loc,
                                    "show": False}
                embedding_kwargs.update(**kwargs)  # overwrite with kwargs from user

                # Plot depending on type
                if plot_type == "PCA-var":
                    plot_pca_variance(adata, ax=ax, show_cumulative=False)  # this plot takes no color

                elif plot_type == "LISI":

                    # Find any LISI scores in adata.obs
                    lisi_columns = [col for col in adata.obs.columns if col.startswith("LISI_score")]

                    if len(lisi_columns) == 0:
                        e = f"No LISI scores found in adata.obs for '{name}'"
                        e += "Please run 'sctoolbox.tools.norm_correct.wrap_batch_evaluation()' or remove LISI from the plots list"
                        raise ValueError(e)

                    # Plot LISI scores
                    boxplot(adata.obs[lisi_columns], ax=ax)
                    LISI_axes.append(ax)

                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, message="No data for colormapping provided via 'c'*")

                        if plot_type == "UMAP":
                            sc.pl.umap(adata, ax=ax, **embedding_kwargs)

                        elif plot_type == "tSNE":
                            sc.pl.tsne(adata, ax=ax, **embedding_kwargs)

                        elif plot_type == "PCA":
                            sc.pl.pca(adata, ax=ax, **embedding_kwargs)

                # Set title for the legend (for categorical color)
                if hasattr(ax, "legend_") and ax.legend_ is not None:

                    # Get current legend and remove
                    lines, labels = ax.get_legend_handles_labels()
                    ax.get_legend().remove()

                    # Replot legend with limited number of clusters
                    per_column = 10
                    n_clusters = min(max_clusters, len(lines))
                    n_cols = int(np.ceil(n_clusters / per_column))

                    if mpl_version > '3.6.0':
                        ax.legend(lines[:max_clusters], labels[:max_clusters],
                                  title=color, ncols=n_cols, frameon=False,
                                  bbox_to_anchor=(1.05, 0.5),
                                  loc=6)
                    else:
                        ax.legend(lines[:max_clusters], labels[:max_clusters],
                                  title=color, ncol=n_cols, frameon=False,
                                  bbox_to_anchor=(1.05, 0.5),
                                  loc=6)

                # Adjust colorbars (for continuous color)
                elif i == len(adatas) - 1 and (color in adata.obs.select_dtypes(include="number").columns or color in adata.var.index):
                    # Replace native scanpy colorbar with self-made one to gain the abililty to set a label
                    # Size parameter values are taken from scanpy: https://github.com/scverse/scanpy/blob/383a61b2db0c45ba622f231f01d0e7546d99566b/scanpy/plotting/_tools/scatterplots.py#L456
                    if len(ax.collections) > 0:
                        plt.colorbar(ax.collections[0], pad=0.01, fraction=0.08, aspect=30, ax=ax, orientation='vertical', label=color)

                _make_square(ax)
                ax_idx += 1  # increment index for next plot

            if plot_type in row_count:
                break  # If not dependent on color; break off early from color_by loop

    # Set common y-axis limit for LISI plots
    if len(LISI_axes) > 0:
        min_y, max_y = np.inf, -np.inf
        for ax in LISI_axes:
            ylim = ax.get_ylim()
            min_y = min(min_y, ylim[0])
            max_y = max(max_y, ylim[1])

        for ax in LISI_axes:
            ax.set_ylim(min_y, max_y)  # scale all plots to same y-limits

        LISI_axes[0].set_ylabel("Unique batch labels in cell neighborhood")

    # Finalize axes titles and labels
    for i, name in enumerate(adatas):
        fontsize = axs[i].title._fontproperties._size * 1.2  # increase title fontsize
        axs[i].set_title(name, size=fontsize, fontweight='bold')  # first rows should have the adata names

    # save
    _save_figure(output, dpi=dpi)

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True)

    return axs


@deco.log_anndata
@beartype
def plot_pca_variance(adata: sc.AnnData,
                      method: str = "pca",
                      n_pcs: int = 20,
                      selected: Optional[List[int]] = None,
                      show_cumulative: bool = True,
                      n_thresh: Optional[int] = None,
                      corr_plot: Optional[Literal["spearmanr", "pearsonr"]] = None,
                      corr_on: Literal["obs", "var"] = "obs",
                      corr_thresh: Optional[float] = None,
                      ignore: Optional[list[str]] = None,
                      ax: Optional[Axes] = None,
                      save: Optional[str] = None,
                      sel_col: str = "grey",
                      om_col: str = "lightgrey",
                      suptitle: Optional[str] = "PCA Component Selection",
                      log_var_exp: bool = False,
                      report: Optional[str] = None,
                      ) -> Axes:
    """Plot the pca variance explained by each component as a barplot.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    method : str, default "pca"
        Method used for calculating variation. Is used to look for the coordinates in adata.uns[<method>].
    n_pcs : int, default 20
        Number of components to plot.
    selected : Optional[List[int]], default None
        Number of components to highlight in the plot.
    show_cumulative : bool, default True
        Whether to show the cumulative variance explained in a second y-axis.
    n_thresh : Optional[int], default None
        Enables a vertical threshold line.
    corr_plot : Optional[str], default None
        Enable correlation plot. Shows highest absolute correlation for each bar.
    corr_on : Literal["obs", "var"], default "obs"
        Calculate correlation on either observations (adata.obs) or variables (adata.var).
    corr_thresh : Optional[float], default None
        Enables a red threshold line in the lower plot.
    ignore : Optional[list[str]], default None
        List of column names to ignore for correlation. By default (None) all numeric columns are used.
        All non numeric columns are ignored by default and cannot be used for correlation.
    ax : Optional[Axes], default None
        Axes object to plot on. If None, a new figure is created.
    save : Optional[str], default None (not saved)
        Filename to save the figure. If None, the figure is not saved.
    sel_col : str, default "grey"
        Bar color of selected bars.
    om_col : str, default "lightgrey"
        Bar color of omitted bars.
    suptitle : Optional[str], default "PCA Component Selection"
        The title of the figure.
    log_var_exp : bool, default False
        Wether to apply log-scale to the "variance explained" (left) y-axis.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.

    Returns
    -------
    Axes
        Axes object containing the plot.

    Raises
    ------
    KeyError
        If the given method is not found in adata.uns.
    ValueError
        If the 'ax' parameter is not an Axes object.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.embedding.plot_pca_variance(adata, method="pca",
                                       n_pcs=20,
                                       selected=[2, 3, 4, 5, 7, 8, 9],
                                       corr_plot="spearmanr")
    """

    if ax is None:
        _, ax = plt.subplots()
    else:
        if not type(ax).__name__.startswith("Axes"):
            raise ValueError("'ax' parameter needs to be an Axes object. Please check your input.")

    if method not in adata.uns:
        raise KeyError("The given method '{0}' is not found in adata.uns. Please make sure to run the method before plotting variance.")

    # Get variance from object
    var_explained = adata.uns[method]["variance_ratio"][:n_pcs]
    var_explained = var_explained * 100  # to percent

    # Cumulative variance
    var_cumulative = np.cumsum(var_explained)
    # cumulative variance for the selected PCs
    if selected:
        sel_var_cumulative = [0]
        for i, var in enumerate(var_explained, start=1):
            if i in selected:
                sel_var_cumulative.append(var + sel_var_cumulative[i-1])
            else:
                sel_var_cumulative.append(sel_var_cumulative[i-1])
        sel_var_cumulative = sel_var_cumulative[1:]

    if corr_plot:
        # compute correlation coefficients
        corrcoefs, _ = tools.embedding.correlation_matrix(adata,
                                                          which=corr_on,
                                                          basis=method,
                                                          n_components=n_pcs,
                                                          ignore=ignore,
                                                          method=corr_plot)

        abs_corrcoefs = list(corrcoefs.abs().max(axis=0))

    # prepare bar coloring by threshold
    if selected:
        palette = [sel_col if i in selected else om_col for i in range(1, n_pcs + 1)]
    else:
        # no threshold
        palette = [sel_col] * n_pcs

    # hide the initial ax object
    ax.set_axis_off()

    # get the figure where the plots will be drawn on
    fig = ax.get_figure()

    if suptitle:
        fig.suptitle(suptitle, fontsize="x-large")

    # create a gridspec (a manual subplot grid) and position it at the location of the ax object
    upper_left, bottom_right = ax.get_position().get_points()
    gridspec = fig.add_gridspec(ncols=1,
                                nrows=2 if corr_plot else 1,
                                left=upper_left[0],
                                right=bottom_right[0],
                                top=bottom_right[1],
                                bottom=upper_left[1],
                                hspace=0.1)  # set the horizontal space between the plots

    axs = [fig.add_subplot(gridspec[0, 0])]

    if corr_plot:
        axs.append(fig.add_subplot(gridspec[1, 0]))

        # share x axis between plots
        axs[0].sharex(axs[1])

    # Plot barplot of variance
    x = list(range(1, len(var_explained) + 1))
    sns.barplot(x=x,
                hue=x,
                y=var_explained,
                color="grey",
                palette=palette,
                legend=False,
                ax=axs[0])

    axs[0].set_ylabel("Variance explained (%)", fontsize=12)

    # apply log-scale
    if log_var_exp:
        axs[0].set_yscale("log")

    # Plot cumulative variance
    if show_cumulative:
        ax2 = axs[0].twinx()

        # add cumulative variance line
        var_lines = [ax2.plot(range(len(var_cumulative)), var_cumulative, color="blue", marker="o", linewidth=1, markersize=3)[0]]
        # add line showing selected cumulative variance
        if selected:
            var_lines.append(ax2.plot(range(len(sel_var_cumulative)), sel_var_cumulative, color="blue", marker="x", linewidth=1, markersize=4)[0])

        ax2.set_ylabel("Cumulative\nvariance explained (%)", color="blue", fontsize=12)
        ax2.spines['right'].set_color('blue')
        ax2.yaxis.label.set_color('blue')
        ax2.tick_params(axis='y', colors='blue')
        # add more than needed to the top to make space for the textbox
        ax2.set_ylim(bottom=0,
                     top=int(var_cumulative[-1] + (var_cumulative[-1] / 3)))

        # add variance legend
        var_labels = [f"Total var.: {var_cumulative[-1]:.2f}%"]
        if selected:
            var_labels.append(f"\nSelected var.: {sel_var_cumulative[-1]:.2f}%")

        ax2.legend(
            handles=var_lines,
            labels=var_labels,
            loc="upper right",
            fontsize=12,
            framealpha=0.5,
            facecolor="#c8c8ff",
            edgecolor="blue",
            bbox_transform=ax2.transAxes,
            labelspacing=-1,
            handlelength=1,
            handletextpad=0.2,
            markerscale=1.3,
            borderpad=0.2
        )

    # Add number of selected as line
    if n_thresh:
        if show_cumulative:
            ylim = ax2.get_ylim()
            yrange = ylim[1] - ylim[0]
            ax2.set_ylim(ylim[0], ylim[1] + yrange * 0.1)  # add 10% to make room for legend of n_seleced line
        axs[0].axvline(n_thresh - 0.5, color="red")  # , label=f"n components included: {n_selected}")
        # axs[0].legend()

    # Plot absolute correlation bar plot
    if corr_plot:
        if corr_thresh:
            # add threshold line
            axs[1].axhline(corr_thresh, color="red")

        sns.barplot(x=x,
                    hue=x,
                    y=abs_corrcoefs,
                    color="grey",
                    palette=palette,
                    legend=False,
                    ax=axs[1])

        # add basis text box
        axs[1].text(
            x=0.95,
            y=0.05,
            s=f"Based on .{corr_on} columns",
            fontsize=12,
            bbox={"boxstyle": "Round", "facecolor": "white", "edgecolor": "black", "alpha": 0.5},
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=axs[1].transAxes
        )

        # Finalize plot
        axs[1].set_xlabel('Principal components', fontsize=12, labelpad=10)
        axs[1].set_ylabel(f"max( |{corr_plot}| )", fontsize=12)
        axs[1].set_ylim([0, 1])
        axs[1].set_xticks(axs[1].get_xticks())  # https://stackoverflow.com/a/68794383/19870975
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, size=7)
        axs[1].set_axisbelow(True)
        axs[1].invert_yaxis()
        axs[1].margins(x=0.01)  # space before first and after last bar

        axs[0].tick_params(bottom=False, labelbottom=False)
        axs[0].margins(x=0.01)  # space before first and after last bar
    else:
        # Finalize plot
        axs[0].set_xlabel('Principal components', fontsize=12, labelpad=10)
        axs[0].set_xticks(axs[0].get_xticks())  # https://stackoverflow.com/a/68794383/19870975
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90, size=7)
        axs[0].set_axisbelow(True)
        axs[0].margins(x=0.01)  # space before first and after last bar

    # Save figure
    _save_figure(save)

    # report
    if settings.report_dir and report:
        _save_figure(report, report=True)

    return ax


@deco.log_anndata
@beartype
def plot_pca_correlation(adata: sc.AnnData,
                         which: Literal["obs", "var"] = "obs",
                         basis: str = "pca",
                         n_components: int = 10,
                         ignore: Optional[list[str]] = None,
                         pvalue_threshold: float = 0.01,
                         method: Literal["spearmanr", "pearsonr"] = "spearmanr",
                         plot_values: Literal["corrcoefs", "pvalues"] = "corrcoefs",
                         figsize: Optional[Tuple[int, int]] = None,
                         title: Optional[str] = None,
                         save: Optional[str] = None,
                         **kwargs: Any) -> Axes:
    """
    Plot a heatmap of the correlation between dimensionality reduction coordinates (e.g. umap or pca) and the given columns.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    which : Literal["obs", "var"], default "obs"
        Whether to use the observations ("obs") or variables ("var") for the correlation.
    basis : str, default "pca"
        Dimensionality reduction to calculate correlation with. Must be a key in adata.obsm, or a basis available as "X_<basis>" such as "umap", "tsne" or "pca".
    n_components : int, default 10
        Number of components to use for the correlation.
    ignore : Optional[list[str]], default None
        List of column names to ignore for correlation. By default (None) all numeric columns are used.
        All non numeric columns are ignored by default and cannot be used for correlation.
    pvalue_threshold : float, default 0.01
        Threshold for significance of correlation. If the p-value is below this threshold, a star is added to the heatmap.
    method : Literal["spearmanr", "pearson"], default "spearmanr"
        Method to use for correlation. Must be either "pearsonr" or "spearmanr".
    plot_values: Literal["corrcoefs", "pvalues"], default "corrcoefs"
        Values which will be used to plot the heatmap, either "corrcoefs" (correlation coefficients) or "pvalues". P-values will be shown as
        `np.sign(corrcoefs)*np.log10(p-value)`, the logged p-value with the sign of the corresponding correlation coefficient.
    figsize : Optional[Tuple[int, int]], default None
        Size of the figure in inches. If None, the size is automatically determined.
    title : Optional[str], default None
        Title of the plot. If None, no title is added.
    save : Optional[str], default None
        Filename to save the figure.
    **kwargs : Any
        Additional keyword arguments are passed to :func:`seaborn.heatmap`.

    Returns
    -------
    ax : Axes
        Axes object containing the heatmap.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.embedding.plot_pca_correlation(adata, which="obs")

    .. plot::
        :context: close-figs

        pl.embedding.plot_pca_correlation(adata, basis="umap")
    """

    # compute correlation matrix
    corrcoefs, pvalues = tools.embedding.correlation_matrix(adata=adata,
                                                            which=which,
                                                            basis=basis,
                                                            n_components=n_components,
                                                            ignore=ignore,
                                                            method=method)

    # decide which values should be shown
    if plot_values == "corrcoefs":
        table = corrcoefs
    elif plot_values == "pvalues":
        # log pvalues
        table = np.sign(corrcoefs) * np.log10(pvalues)

    # prepare annotation shown on the heatmap
    annot = table.copy()

    annot = annot.map(lambda x: str(np.round(x, 2)))
    # add stars to significant values
    stars = pvalues.map(lambda p: "*" if p < pvalue_threshold else "")
    annot += stars

    # Plot heatmap
    figsize = figsize if figsize is not None else (len(corrcoefs.columns) / 1.5, len(corrcoefs) / 1.5)
    fig, ax = plt.subplots(figsize=figsize)

    if plot_values == "corrcoefs":
        # center of cbar is 0
        vmin = -1
        vmax = 1
    elif plot_values == "pvalues":
        # infer min and max for cbar from data
        vmin = None
        vmax = None

    ax = sns.heatmap(corrcoefs,
                     annot=annot,
                     fmt='',
                     annot_kws={"fontsize": 9},
                     cbar_kws={"label": f"{method} ({plot_values})"},
                     cmap="seismic",
                     vmin=vmin,
                     vmax=vmax,
                     ax=ax,
                     **kwargs)
    ax.set_aspect(0.8)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Set size of cbar to the same height as the heatmap
    cbar_ax = fig.get_axes()[-1]
    ax_pos = ax.get_position()
    cbar_pos = cbar_ax.get_position()

    cbar_ax.set_position([ax_pos.x1 + 2 * cbar_pos.width, ax_pos.y0,
                          cbar_pos.width, ax_pos.height])

    # Add black borders to axes
    for ax_obj in [ax, cbar_ax]:
        for _, spine in ax_obj.spines.items():
            spine.set_visible(True)

    # Add title
    if title is not None:
        ax.set_title(str(title))

    # Save figure
    _save_figure(save)

    return ax
