"""Funtions of different single cell embeddings e.g. UMAP, PCA, tSNE."""

import multiprocessing as mp
import warnings
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.stats
from scipy.sparse import issparse
import itertools
import re

import seaborn as sns
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import plotly as po
import plotly.graph_objects as go

from numba import errors as numba_errors

import deprecation
from sctoolbox import __version__
from beartype import beartype
from beartype.typing import Literal, Tuple, Optional, Union, Any
import numpy.typing as npt

import sctoolbox.utils as utils
from sctoolbox.plotting.general import _save_figure, _make_square, boxplot
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


#############################################################################
#                                  Utilities                                #
#############################################################################

@beartype
def sc_colormap() -> matplotlib.colors.ListedColormap:
    """Get a colormap with 0-count cells colored grey (to use for embeddings).

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap with 0-count cells colored grey.
    """

    # Custom colormap for single cells
    color_cmap = cm.get_cmap('Reds', 200)
    newcolors = color_cmap(np.linspace(0.2, 0.9, 200))
    newcolors[0, :] = colors.to_rgba("lightgrey")  # count 0 = grey
    sc_cmap = ListedColormap(newcolors)

    return sc_cmap


def grey_colormap() -> matplotlib.colors.ListedColormap:
    """Get a colormap with grey-scale colors, but without white to still show cells.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
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
def _add_contour(x: np.ndarray,
                 y: np.ndarray,
                 ax: matplotlib.axes.Axes):
    """Add contour plot to a scatter plot.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates of the scatter plot.
    y : np.ndarray
        y-coordinates of the scatter plot.
    ax : matplotlib.axes.Axes
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
                   save: Optional[str] = None,
                   **kwargs) -> npt.ArrayLike:
    """Plot a dimensionality reduction embedding e.g. UMAP or tSNE with different style options. This is a wrapper around scanpy.pl.embedding.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    method : str, default "umap"
        Dimensionality reduction method to use. Must be a key in adata.obsm, or a method available as "X_<method>" such as "umap", "tsne" or "pca".
    color : Optional[str | list[str]], default None
        Key for annotation of observations/cells or variables/genes.
    style : Literal["dots", "hexbin", "density".], default "dots"
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
    save : Optional[str], default None
        Filename to save the figure.
    **kwargs : arguments
        Additional keyword arguments are passed to :func:`scanpy.pl.plot_embedding`.

    Returns
    -------
    axes : npt.ArrayLike
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

        pl.plot_embedding(adata, color="louvain", legend_loc="on data")

    .. plot::
        :context: close-figs

        _ = pl.plot_embedding(adata, method="pca", color="n_genes", show_contour=True, show_title=False)

    .. plot::
        :context: close-figs

        _ = pl.plot_embedding(adata, color=['n_genes', 'HES4'], style="hexbin")

    .. plot::
        :context: close-figs

        _ = pl.plot_embedding(adata, method="pca", color=['n_genes', 'HES4'],
                              style="hexbin", components=["1,2", "2,3"], ncols=2)

    .. plot::
        :context: close-figs

        ax = pl.plot_embedding(adata, color=['n_genes', 'louvain'], style="density")
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
        legend = ax.get_legend()
        local_axes = ax.figure._localaxes  # list of all plot and colorbar axes in figure
        has_colorbar = False
        if legend is not None:  # legend of categorical variables
            if not show_title:
                legend.set_title(ax_color)
        else:                   # legend of continuous variables
            cbar_ax_idx = local_axes.index(ax) + 1  # colorbar is always right after plot
            cbar_ax_idx = min(cbar_ax_idx, len(local_axes) - 1)  # ensure that idx is within bounds
            cbar_ax = local_axes[cbar_ax_idx]
            if cbar_ax._label == "<colorbar>":
                has_colorbar = True  # this ax has colorbar

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
            cmap = mpl.rcParams["image.cmap"] if cmap is None else cmap  # if cmap is None, scanpy uses default cmap for matplotlib
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

                    adata_subsets = utils.get_adata_subsets(adata, groupby=ax_color)
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

            # Move colorbar to the correct position in _localaxes
            cbar_idx = local_axes.index(cbar_ax)
            new_cbar_idx = local_axes.index(new_cbar_ax)
            local_axes[cbar_idx] = new_cbar_ax
            local_axes.pop(new_cbar_idx)  # remove original idx of new_cbar_ax

    # Save figure
    _save_figure(save)

    return axarr


@deco.log_anndata
@beartype
def search_umap_parameters(adata: sc.AnnData,
                           min_dist_range: Tuple[float | int, float | int, float | int] = (0.2, 0.9, 0.2),  # 0.2, 0.4, 0.6, 0.8
                           spread_range: Tuple[float | int, float | int, float | int] = (0.5, 2.0, 0.5),    # 0.5, 1.0, 1.5
                           color: Optional[str] = None,
                           n_components: int = 2,
                           threads: int = 4,
                           save: Optional[str] = None,
                           **kwargs: Any) -> np.ndarray:
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
    np.ndarray
        2D numpy array of axis objects

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.search_umap_parameters(adata, min_dist_range=(0.2, 0.9, 0.2),
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
                           **kwargs: Any) -> np.ndarray:
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
    np.ndarray
        2D numpy array of axis objects

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.search_tsne_parameters(adata, perplexity_range=(30, 60, 10),
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
                               **kwargs: Any) -> np.ndarray:
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
    np.ndarray
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
    adata = utils.get_minimal_adata(adata)
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
            pbar = utils.get_pbar(len(loop_params[0]) * len(loop_params[1]), f"Computing {method.upper()}s")

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
            utils.monitor_jobs(jobs, f"Computing {method.upper()}s")
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
                legend_loc = "left"
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
                          embedding: Literal["umap", "tsne", "pca"] = "umap",
                          ncols: int = 4,
                          save: Optional[str] = None) -> np.ndarray:
    """
    Plot a grid of embeddings (UMAP/tSNE/PCA) per group of cells within 'groupby'.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    groupby : str
        Name of the column in adata.obs to group by.
    embedding : Literal["umap", "tsne", "pca"], default "umap"
        Embedding to plot. Must be one of "umap", "tsne", "pca".
    ncols : int, default 4
        Number of columns in the figure.
    save : Optional[str], default None
        Path to save the figure.

    Returns
    -------
    np.ndarray
        Flat numpy array of axis objects

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.plot_group_embeddings(adata, 'phase', embedding='umap', ncols=4)
    """

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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message="Categorical.replace is deprecated")
            warnings.filterwarnings("ignore", category=FutureWarning, message="In a future version of pandas")
            warnings.filterwarnings("ignore", category=UserWarning, message="No data for colormapping provided via 'c'*")

            # Plot individual embedding
            if embedding == "umap":
                sc.pl.umap(adata, color=groupby, groups=group, ax=ax, show=False, legend_loc=None)
            elif embedding == "tsne":
                sc.pl.tsne(adata, color=groupby, groups=group, ax=ax, show=False, legend_loc=None)
            elif embedding == "pca":
                sc.pl.pca(adata, color=groupby, groups=group, ax=ax, show=False, legend_loc=None)

        ax.set_title(group)

    # Hide last empty plots
    n_empty = n_plots - n_groups
    if n_empty > 0:
        for ax in axes_list[-n_empty:]:
            ax.set_visible(False)

    # Save figure
    _save_figure(save)

    return axarr


@beartype
def compare_embeddings(adata_list: list[sc.AnnData],
                       var_list: list[str] | str,
                       embedding: Literal["umap", "tsne", "pca"] = "umap",
                       adata_names: Optional[list[str]] = None,
                       **kwargs: Any) -> np.ndarray:
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
    np.ndarray
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

        pl.compare_embeddings(adata_list, var_list)
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

    # fig.tight_layout()
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
                 save: str) -> None:
    """Save 3D UMAP plot to a html file.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    color : str
        Variable to color in plot. Must be a column in adata.obs or an index in adata.var.
    save : str
        Save prefix. Plot will be saved to <save>.html.

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

        pl.plot_3D_UMAP(adata, color="louvain", save="my3d_umap")

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
    if color in adata.obs.columns and isinstance(adata.obs[color][0], str):

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
                                               opacity=0.8))
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


@deprecation.deprecated(deprecated_in="0.3b", removed_in="0.5",
                        current_version=__version__,
                        details="Use the 'sctoolbox.pl.plot_embedding' function instead.")
@deco.log_anndata
@beartype
def umap_pub(adata: sc.AnnData,
             color: Optional[str | list[str]] = None,
             title: Optional[str | list[str]] = None,
             save: Optional[str] = None,
             **kwargs: Any) -> list:
    """Plot a publication ready UMAP without spines, but with a small UMAP1/UMAP2 legend.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    color : Optional[str | list[str]], default None
        Key for annotation of observations/cells or variables/genes.
    title : Optional[str | list[str]], default None
        Title of the plot. Default is no title.
    save : Optional[str], default None
        Filename to save the figure.
    **kwargs : Any
        Additional arguments passed to `sc.pl.umap`.

    Returns
    -------
    axarr : list
        list of matplotlib axis objects

    Raises
    ------
    ValueError
        If color and title have different lengths.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.umap_pub(adata, color="louvain", title="Louvain clusters")
    """

    axarr = sc.pl.umap(adata, color=color, show=False, **kwargs)

    if title is not None and not isinstance(title, list):
        title = [title]

    if not isinstance(axarr, list):
        axarr = [axarr]
        color = [color]

    if title and len(title) != len(color):
        raise ValueError("Color and Title must have the same length.")

    colorbar_count = 0
    for i, ax in enumerate(axarr):

        # Set legend
        legend = ax.get_legend()
        if legend is not None:  # legend of categorical variables
            legend.set_title(color[i])
        else:                   # legend of continuous variables
            colorbar_idx = i + colorbar_count + 1
            local_axes = ax.figure._localaxes
            if colorbar_idx < len(local_axes) and local_axes[colorbar_idx]._label == '<colorbar>':
                local_axes[colorbar_idx].set_title(color[i])
                colorbar_count += 1

        # Remove automatic title
        ax.set_title("")
        if title is not None:
            ax.set_title(title[i])

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
        ax.text(0.02, 0.02, f"{adata.n_obs:,} cells",
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='bottom')

        # Adjust aspect ratio
        _make_square(ax)

    # Save figure
    _save_figure(save)

    return axarr


@beartype
def anndata_overview(adatas: dict[str, sc.AnnData],
                     color_by: str | list[str],
                     plots: Union[list[Literal["UMAP", "tSNE", "PCA", "PCA-var", "LISI"]],
                                  Literal["UMAP", "tSNE", "PCA", "PCA-var", "LISI"]] = ["PCA", "PCA-var", "UMAP", "LISI"],
                     figsize: Optional[Tuple[int, int]] = None,
                     max_clusters: int = 20,
                     output: Optional[str] = None,
                     dpi: int = 300) -> npt.ArrayLike:
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

    Returns
    -------
    axes : npt.ArrayLike
        Array of matplotlib.axes.Axes objects created by matplotlib.

    Raises
    ------
    ValueError
        If any of the adatas is not of type anndata.AnnData or an invalid plot is specified.

    Examples
    --------
    .. plot::
        :context: close-figs

        adatas = {}  # dictionary of adata objects
        adatas["standard"] = adata
        adatas["parameter1"] = sc.tl.umap(adata, min_dist=1, copy=True)
        adatas["parameter2"] = sc.tl.umap(adata, min_dist=2, copy=True)

        pl.anndata_overview(adatas, color_by="louvain", plots=["PCA", "PCA-var", "UMAP"])
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

    # check if plots are valid
    valid_plots = ["UMAP", "tSNE", "PCA", "PCA-var", "LISI"]
    invalid_plots = set(plots) - set(valid_plots)
    if invalid_plots:
        raise ValueError(f"Invalid plot specified: {invalid_plots}")

    # ---- plotting ---- #
    # setup subplot structure
    row_count = {"PCA-var": 1, "LISI": 1}  # all other plots count for len(color_by)
    rows = sum([row_count.get(plot, len(color_by)) for plot in plots])  # the number of rows in output plot
    cols = len(adatas)
    figsize = figsize if figsize is not None else (2 + cols * 4, rows * 4)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)  # , constrained_layout=True)
    axs = axs.flatten() if rows > 1 or cols > 1 else [axs]  # flatten to 1d array per row

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
                    colorbar_loc = "right"
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
                                    "legend_loc": legend_loc, "colorbar_loc": colorbar_loc,
                                    "show": False}

                # Plot depending on type
                if plot_type == "PCA-var":
                    plot_pca_variance(adata, ax=ax, show_cumulative=False)  # this plot takes no color

                elif plot_type == "LISI":

                    # Find any LISI scores in adata.obs
                    lisi_columns = [col for col in adata.obs.columns if col.startswith("LISI_score")]

                    if len(lisi_columns) == 0:
                        e = f"No LISI scores found in adata.obs for '{name}'"
                        e += "Please run 'sctoolbox.tools.wrap_batch_evaluation()' or remove LISI from the plots list"
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

                    # Get current legend and rmove
                    lines, labels = ax.get_legend_handles_labels()
                    ax.get_legend().remove()

                    # Replot legend with limited number of clusters
                    per_column = 10
                    n_clusters = min(max_clusters, len(lines))
                    n_cols = int(np.ceil(n_clusters / per_column))

                    if mpl.__version__ > '3.6.0':
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
                elif hasattr(ax, "_colorbars") and len(ax._colorbars) > 0:
                    ax._colorbars[0].set_title(color, ha="left")
                    ax._colorbars[0]._colorbar_info["shrink"] = 0.8
                    ax._colorbars[0]._colorbar_info["pad"] = -0.15  # move colorbar closer to plot

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

    return axs


@deco.log_anndata
@beartype
def plot_pca_variance(adata: sc.AnnData,
                      method: str = "pca",
                      n_pcs: int = 20,
                      n_selected: Optional[int] = None,
                      show_cumulative: bool = True,
                      ax: Optional[matplotlib.axes.Axes] = None,
                      save: Optional[str] = None) -> matplotlib.axes.Axes:
    """Plot the pca variance explained by each component as a barplot.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object.
    method : str, default "pca"
        Method used for calculating variation. Is used to look for the coordinates in adata.uns[<method>].
    n_pcs : int, default 20
        Number of components to plot.
    n_selected : Optional[int], default None
        Number of components to highlight in the plot with a red line.
    show_cumulative : bool, default True
        Whether to show the cumulative variance explained in a second y-axis.
    ax : Optional[matplotlib.axes.Axes], default None
        Axes object to plot on. If None, a new figure is created.
    save : Optional[str], default None (not saved)
        Filename to save the figure. If None, the figure is not saved.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object containing the plot.

    Raises
    ------
    KeyError
        If the given method is not found in adata.uns.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.plot_pca_variance(adata, method="pca",
                      n_pcs=20,
                      n_selected=7)
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

    # Plot barplot of variance
    x = list(range(1, len(var_explained) + 1))
    sns.barplot(x=x,
                y=var_explained,
                color="grey",
                ax=ax)

    # Plot cumulative variance
    if show_cumulative:
        ax2 = ax.twinx()
        ax2.plot(range(len(var_cumulative)), var_cumulative, color="blue", marker="o", linewidth=1, markersize=3)
        ax2.set_ylabel("Cumulative variance explained (%)", color="blue", fontsize=12)
        ax2.spines['right'].set_color('blue')
        ax2.yaxis.label.set_color('blue')
        ax2.tick_params(axis='y', colors='blue')

    # Add number of selected as line
    if n_selected is not None:
        if show_cumulative:
            ylim = ax2.get_ylim()
            yrange = ylim[1] - ylim[0]
            ax2.set_ylim(ylim[0], ylim[1] + yrange * 0.1)  # add 10% to make room for legend of n_seleced line
        ax.axvline(n_selected - 0.5, color="red", label=f"n components included: {n_selected}")
        ax.legend()

    # Finalize plot
    ax.set_xlabel('Principal components', fontsize=12, labelpad=10)
    ax.set_ylabel("Variance explained (%)", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=7)
    ax.set_axisbelow(True)

    # Save figure
    _save_figure(save)

    return ax


@deco.log_anndata
@beartype
def plot_pca_correlation(adata: sc.AnnData,
                         which: Literal["obs", "var"] = "obs",
                         basis: str = "pca",
                         n_components: int = 10,
                         columns: Optional[list[str]] = None,
                         pvalue_threshold: float = 0.01,
                         method: Literal["spearmanr", "pearsonr"] = "spearmanr",
                         plot_values: Literal["corrcoefs", "pvalues"] = "corrcoefs",
                         figsize: Optional[Tuple[int, int]] = None,
                         title: Optional[str] = None,
                         save: Optional[str] = None) -> matplotlib.axes.Axes:
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
    columns : Optional[list[str]], default None
        List of columns to use for the correlation. If None, all numeric columns are used.
    pvalue_threshold : float, default 0.01
        Threshold for significance of correlation. If the p-value is below this threshold, a star is added to the heatmap.
    method : Literal["spearmanr", "pearson"], default "spearmanr"
        Method to use for correlation. Must be either "pearsonr" or "spearmanr".
    plot_values: Literal["corrcoefs", "pvalues"], default "corrcoefs"
        Values which will be used to plot the heatmap, either "corrcoefs" (correlation coefficients) or "pvalues".
    figsize : Optional[Tuple[int, int]], default None
        Size of the figure in inches. If None, the size is automatically determined.
    title : Optional[str], default None
        Title of the plot. If None, no title is added.
    save : Optional[str], default None
        Filename to save the figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object containing the heatmap.

    Raises
    ------
    ValueError
        If "basis" is not found in data, if "which" is not "obs" or "var", if "method" is not "pearsonr" or "spearmanr", or if "which" is "var" and "basis" not "pca".
    KeyError
        If any of the given columns is not found in the respective table.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.plot_pca_correlation(adata, which="obs")

    .. plot::
        :context: close-figs

        pl.plot_pca_correlation(adata, basis="umap")
    """

    # Check that basis is in adata.obsm
    if basis not in adata.obsm:
        basis = "X_" + basis if not basis.startswith("X_") else basis  # check if basis is available as "X_<basis>"
        if basis not in adata.obsm:
            raise KeyError(f"The given basis '{basis}' cannot be found in adata.obsm. The available keys are: {list(adata.obsm.keys())}.")

    # Establish which table to use
    if which == "obs":
        table = adata.obs.copy()
        mat = adata.obsm[basis]
    elif which == "var":
        if "pca" not in basis.lower():
            raise ValueError("Correlation with 'var' can only be calculated with PCA components!")
        table = adata.var.copy()
        mat = adata.varm["PCs"]

    # Check that method is available
    try:
        corr_method = getattr(scipy.stats, method)
    except AttributeError:
        s = f"'{method}' is not a valid method within scipy.stats. Please choose one of pearsonr/spearmanr."
        raise ValueError(s)

    # Get columns
    if columns is None:
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric_columns = table.select_dtypes(include=numerics).columns.tolist()
    else:
        utils.check_columns(table, columns)

    # Get table of pcs and columns
    n_components = min(n_components, mat.shape[1])  # make sure we don't exceed the number of pcs available
    if "pca" in basis.lower():
        comp_columns = [f"PC{i+1}" for i in range(n_components)]  # e.g. PC1, PC2, ...
    else:
        comp_columns = [f"{re.sub('^X_', '', basis.upper())}{i+1}" for i in range(n_components)]  # e.g. UMAP1, UMAP2, ...
    comp_table = pd.DataFrame(mat[:, :n_components], columns=comp_columns)
    comp_table[numeric_columns] = table[numeric_columns].reset_index(drop=True)

    # Calculate correlation of columns
    combinations = list(itertools.product(numeric_columns, comp_columns))

    corr_table = pd.DataFrame(index=numeric_columns, columns=comp_columns, dtype=float)
    corr_table_annot = corr_table.copy()
    for row, col in combinations:
        # remove NaN values and the corresponding values from both lists
        x = np.vstack([comp_table[row], comp_table[col]])  # stack values of row and column
        x = x[:, ~np.any(np.isnan(x), axis=0)]  # remove columns with NaN values

        res = corr_method(x[0], x[1])

        if plot_values == "corrcoefs":
            value = res.statistic
            # center of cbar is 0
            vmin = -1
            vmax = 1
        elif plot_values == "pvalues":
            value = -np.sign(res.statistic) * np.log10(res.pvalue)
            # infer min and max for cbar from data
            vmin = None
            vmax = None

        corr_table.loc[row, col] = value
        corr_table_annot.loc[row, col] = str(np.round(value, 2))
        corr_table_annot.loc[row, col] += "*" if res.pvalue < pvalue_threshold else ""

    # Plot heatmap
    figsize = figsize if figsize is not None else (len(comp_columns) / 1.5, len(numeric_columns) / 1.5)
    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.heatmap(corr_table,
                     annot=corr_table_annot,
                     fmt='',
                     annot_kws={"fontsize": 9},
                     cbar_kws={"label": f"{method} ({plot_values})"},
                     cmap="seismic",
                     vmin=vmin, vmax=vmax,
                     ax=ax)
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
