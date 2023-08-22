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
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap
import plotly as po
import plotly.graph_objects as go

from numba import errors as numba_errors

import sctoolbox.utils as utils
from sctoolbox.plotting.general import _save_figure, _make_square, boxplot
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


#############################################################################
#                                  Utilities                                #
#############################################################################

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


@deco.log_anndata
def flip_embedding(adata, key="X_umap", how="vertical"):
    """Flip the embedding in adata.obsm[key] along the given axis.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    key : str, default "X_umap"
        Key in adata.obsm to flip.
    how : str, default "vertical"
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

@deco.log_anndata
def search_umap_parameters(adata,
                           min_dist_range=(0.2, 0.9, 0.2),  # 0.2, 0.4, 0.6, 0.8
                           spread_range=(0.5, 2.0, 0.5),    # 0.5, 1.0, 1.5
                           color=None, n_components=2, threads=4, save=None, **kwargs) -> np.ndarray:
    """Plot a grid of different combinations of min_dist and spread variables for UMAP plots.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    min_dist_range : tuple, default: (0.2, 0.9, 0.2)
        Range of 'min_dist' parameter values to test. Must be a tuple in the form (min, max, step).
    spread_range : tuple, default (0.5, 2.0, 0.5)
        Range of 'spread' parameter values to test. Must be a tuple in the form (min, max, step).
    color : str, default None
        Name of the column in adata.obs to color plots by. If None, plots are not colored.
    n_components : int, default 2
        Number of components in UMAP calculation.
    threads : int, default 4
        Number of threads to use for UMAP calculation.
    save : str, default None
        Path to save the figure to. If None, the figure is not saved.
    **kwargs : arguments
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
def search_tsne_parameters(adata,
                           perplexity_range=(30, 60, 10), learning_rate_range=(600, 1000, 200),
                           color=None, threads=4, save=None, **kwargs) -> np.ndarray:
    """Plot a grid of different combinations of perplexity and learning_rate variables for tSNE plots.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    perplexity_range : tuple, default (30, 60, 10)
        tSNE parameter: Range of 'perplexity' parameter values to test. Must be a tuple in the form (min, max, step).
    learning_rate_range : tuple, default (600, 1000, 200)
        tSNE parameter: Range of 'learning_rate' parameter values to test. Must be a tuple in the form (min, max, step).
    color : str, default None
        Name of the column in adata.obs to color plots by. If None, plots are not colored.
    threads : int, default 1
        The threads paramerter is currently not supported. Please leave at 1.
        This may be fixed in the future.
    save : str, default None (not saved)
        Path to save the figure to.
    **kwargs : arguments
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


def _search_dim_red_parameters(adata, method,
                               min_dist_range=None, spread_range=None,  # for UMAP
                               perplexity_range=None, learning_rate_range=None,  # for tSNE
                               color=None, threads=4, save=None, **kwargs) -> np.ndarray:
    """Search different combinations of parameters for UMAP or tSNE and plot a grid of the embeddings.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    method : str
        Dimensionality reduction method to use. Must be either 'umap' or 'tsne'.
    min_dist_range : tuple, default None
        UMAP parameter: Range of 'min_dist' parameter values to test. Must be a tuple in the form (min, max, step).
    spread_range : tuple, default None
        UMAP parameter: Range of 'spread' parameter values to test. Must be a tuple in the form (min, max, step).
    perplexity_range : tuple, default None
        tSNE parameter: Range of 'perplexity' parameter values to test. Must be a tuple in the form (min, max, step).
    learning_rate_range : tuple, default None
        tSNE parameter: Range of 'learning_rate' parameter values to test. Must be a tuple in the form (min, max, step).
    color : str, default None
        Name of the column in adata.obs to color plots by. If None, plots are not colored.
    threads : int, default 4
        Number of threads to use for calculating embeddings. In case of UMAP, the embeddings will be calculated in parallel with each job using 1 thread.
        For tSNE, the embeddings are calculated serially, but each calculation uses 'threads' as 'n_jobs' within sc.tl.tsne.
    save : str, default None
        Path to save the figure to.
    **kwargs : arguments
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
    else:
        raise ValueError("Invalid method. Please choose from ['tsne', 'umap']")

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
def plot_group_embeddings(adata, groupby, embedding="umap", ncols=4, save=None) -> np.ndarray:
    """
    Plot a grid of embeddings (UMAP/tSNE/PCA) per group of cells within 'groupby'.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    groupby : str
        Name of the column in adata.obs to group by.
    embedding : str, default "umap"
        Embedding to plot. Must be one of "umap", "tsne", "pca".
    ncols : int, default 4
        Number of columns in the figure.
    save : str, default None
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


def compare_embeddings(adata_list, var_list, embedding="umap", adata_names=None, **kwargs) -> np.ndarray:
    """Compare embeddings across different adata objects.

    Plots a grid of embeddings with the different adatas on the x-axis, and colored variables on the y-axis.

    Parameters
    ----------
    adata_list : list[sc.AnnData]
        List of AnnData objects to compare.
    var_list : list of str
        List of variables to color in plot.
    embedding : str, default "umap"
        Embedding to plot. Must be one of "umap", "tsne" or "pca".
    adata_names : list of str, default None (adatas will be named adata_1, adata_2, etc.)
        List of names for the adata objects. Must be the same length as adata_list or None
    **kwargs : arguments
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

def _get_3d_dotsize(n):
    """Get the optimal plotting dotsize for a given number of points."""
    if n < 1000:
        return 12
    elif n < 10000:
        return 8
    else:
        return 3


@deco.log_anndata
def plot_3D_UMAP(adata, color, save):
    """Save 3D UMAP plot to a html file.

    Parameters
    ----------
    adata : anndata.AnnData
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

        import scanpy as sc
        import sctoolbox.plotting
        min_dist = 0.3
        spread = 2.5
        sc.tl.umap(adata, min_dist=min_dist, spread=spread, n_components=3)

    .. plot::
        :context: close-figs

        sctoolbox.plotting.plot_3D_UMAP(adata, color="louvain", save="my3d_umap")

    .. plot::
        :context: close-figs

        RESULT = "File was written to my3d_umap.html"
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
def umap_marker_overview(adata, markers, ncols=3, figsize=None,
                         save=None,
                         cbar_label="Relative expr.",
                         **kwargs):
    """Plot a pretty grid of UMAPs with marker gene expression."""

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


@deco.log_anndata
def umap_pub(adata, color=None, title=None, save=None, **kwargs) -> list:
    """Plot a publication ready UMAP without spines, but with a small UMAP1/UMAP2 legend.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    color : str or lst of str, default None
        Key for annotation of observations/cells or variables/genes.
    title : str, default None
        Title of the plot. Default is no title.
    save : str, default None
        Filename to save the figure.
    **kwargs : dict
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


def anndata_overview(adatas,
                     color_by,
                     plots=["PCA", "PCA-var", "UMAP", "LISI"],
                     figsize=None,
                     max_clusters=20,
                     output=None,
                     dpi=300) -> list:
    """Create a multipanel plot comparing PCA/UMAP/tSNE/(...) plots for different adata objects.

    Parameters
    ----------
    adatas : dict[str, sc.AnnData]
        Dict containing an anndata object for each batch correction method as values. Keys are the name of the respective method.
        E.g.: {"bbknn": anndata}
    color_by : str or list of str
        Name of the .obs column to use for coloring in applicable plots (e.g. for UMAP or PCA).
    plots : str or list of str, default ["PCA", "PCA-var", "UMAP", "LISI"]
        Decide what plots should be created. Options are ["UMAP", "tSNE", "PCA", "PCA-var", "LISI"]
        Note: List order is forwarded to plot.
        - UMAP: Plots the UMAP embedding of the data.
        - tSNE: Plots the tSNE embedding of the data.
        - PCA: Plots the PCA embedding of the data.
        - PCA-var: Plots the variance explained by each PCA component.
        - LISI: Plots the distribution of any "LISI_score*" scores available in adata.obs
    figsize : number tuple, default None (automatic based on number of columns/rows)
        Size of the plot in inch.
    max_clusters : int, default 20
        Maximum number of clusters to show in legend.
    output : str, default None (not saved)
        Path to plot output file.
    dpi : number, default 300
        Dots per inch for output

    Returns
    -------
    axes : list
        List of matplotlib.axes.Axes objects created by matplotlib.

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
        LISI_axes[0].get_shared_y_axes().join(LISI_axes[0], *LISI_axes[1:])
        LISI_axes[0].autoscale()  # scale all plots to same y-limits

        LISI_axes[0].set_ylabel("Unique batch labels in cell neighborhood")

    # Finalize axes titles and labels
    for i, name in enumerate(adatas):
        fontsize = axs[i].title._fontproperties._size * 1.2  # increase title fontsize
        axs[i].set_title(name, size=fontsize, fontweight='bold')  # first rows should have the adata names

    # save
    _save_figure(output, dpi=dpi)

    return axs


@deco.log_anndata
def plot_pca_variance(adata, method="pca",
                      n_pcs=20,
                      n_selected=None,
                      show_cumulative=True,
                      ax=None,
                      save=None) -> matplotlib.axes.Axes:
    """Plot the pca variance explained by each component as a barplot.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    method : str, default "pca"
        Method used for calculating variation. Is used to look for the coordinates in adata.uns[<method>].
    n_pcs : int, default 20
        Number of components to plot.
    n_selected : int, default None
        Number of components to highlight in the plot with a red line.
    show_cumulative : bool, default True
        Whether to show the cumulative variance explained in a second y-axis.
    ax : matplotlib.axes.Axes, default None
        Axes object to plot on. If None, a new figure is created.
    save : str, default None (not saved)
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

        #init data
        import sctoolbox.plotting as pl
        import scanpy as sc

        adata = sc.datasets.pbmc68k_reduced()

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
def plot_pca_correlation(adata, which="obs",
                         n_pcs=10,
                         columns=None,
                         pvalue_threshold=0.01,
                         method="spearmanr",
                         figsize=None,
                         title=None,
                         save=None) -> matplotlib.axes.Axes:
    """
    Plot a heatmap of the correlation between the first n_pcs and the given columns.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    which : str, default "obs"
        Whether to use the observations ("obs") or variables ("var") for the correlation.
    n_pcs : int, default 10
        Number of principal components to use for the correlation.
    columns : list of str, default None
        List of columns to use for the correlation. If None, all numeric columns are used.
    pvalue_threshold : float, default 0.01
        Threshold for significance of correlation. If the p-value is below this threshold, a star is added to the heatmap.
    method : str, default "spearmanr"
        Method to use for correlation. Must be either "pearsonr" or "spearmanr".
    figsize : tuple of int, default None
        Size of the figure in inches. If None, the size is automatically determined.
    title : str, default None
        Title of the plot. If None, no title is added.
    save : str, default None
        Filename to save the figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object containing the heatmap.

    Raises
    ------
    ValueError
        If "which" is not "obs" or "var", or if "method" is not "pearsonr" or "spearmanr".
    KeyError
        If any of the given columns is not found in the respective table.

    Examples
    --------
    .. plot::
        :context: close-figs

        pl.plot_pca_correlation(adata, which="obs")
    """

    # Establish which table to use
    if which == "obs":
        table = adata.obs.copy()
        mat = adata.obsm["X_pca"]
    elif which == "var":
        table = adata.var.copy()
        mat = adata.varm["PCs"]
    else:
        raise ValueError(f"'which' must be either 'var'/'obs', but '{which}' was given.")

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
    n_pcs = min(n_pcs, mat.shape[1])  # make sure we don't exceed the number of pcs available
    pc_columns = [f"PC{i+1}" for i in range(n_pcs)]
    pc_table = pd.DataFrame(mat[:, :n_pcs], columns=pc_columns)
    pc_table[numeric_columns] = table[numeric_columns].reset_index(drop=True)

    # Calculate correlation of columns
    combinations = list(itertools.product(numeric_columns, pc_columns))

    corr_table = pd.DataFrame(index=numeric_columns, columns=pc_columns, dtype=float)
    corr_table_annot = corr_table.copy()
    for row, col in combinations:

        res = corr_method(pc_table[row], pc_table[col])
        corr_table.loc[row, col] = res.statistic

        corr_table_annot.loc[row, col] = str(np.round(res.statistic, 2))
        corr_table_annot.loc[row, col] += "*" if res.pvalue < pvalue_threshold else ""

    # Plot heatmap
    figsize = figsize if figsize is not None else (len(pc_columns) / 1.5, len(numeric_columns) / 1.5)
    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.heatmap(corr_table,
                     annot=corr_table_annot,
                     fmt='',
                     annot_kws={"fontsize": 9},
                     cbar_kws={"label": method},
                     cmap="seismic",
                     vmin=-1, vmax=1,  # center is 0
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
