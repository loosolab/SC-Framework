"""
Modules for plotting single cell data

.. rubric:: Loading the module

.. plot ::
    :context: close-figs

    import sctoolbox.plotting as pl


.. rubric:: Loading example data

.. plot ::
    :context: close-figs

    import numpy as np
    import scanpy as sc

    adata = sc.datasets.pbmc68k_reduced()
    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])

.. rubric:: Functions

"""

from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scanpy as sc
import qnorm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import multiprocessing as mp
import warnings

from matplotlib import cm, colors
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.sparse import issparse

import sctoolbox.utilities
import sctoolbox.analyser
import sctoolbox.utilities as utils
from sctoolbox.utilities import save_figure
import plotly as po
import plotly.graph_objects as go


def _make_square(ax):
    """ Utility function to set the aspect ratio of a plot to be a square """

    xrange = np.diff(ax.get_xlim())[0]
    yrange = np.diff(ax.get_ylim())[0]

    aspect = xrange / yrange
    ax.set_aspect(aspect)


#############################################################################
#                     PCA/tSNE/UMAP plotting functions                      #
#############################################################################

def sc_colormap():
    """
    Get a colormap with 0-count cells colored grey (to use for embeddings).

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


def flip_embedding(adata, key="X_umap", how="vertical"):
    """
    Flip the embedding in adata.obsm[key] along the given axis.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    key : str, default "X_umap"
        Key in adata.obsm to flip.
    how : str, default "vertical"
        Axis to flip along. Can be "vertical" (flips up/down) or "horizontal" (flips left/right).
    """

    if how == "vertical":
        adata.obsm[key][:, 1] = -adata.obsm[key][:, 1]
    elif how == "horizontal":
        adata.obsm[key][:, 0] = -adata.obsm[key][:, 0]
    else:
        raise ValueError("The given axis '{0}' is not supported. Please use 'vertical' or 'horizontal'.".format(how))


def plot_pca_variance(adata, method="pca", n_pcs=20, ax=None):
    """
    Plot the pca variance explained by each component as a barplot.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    method : str
        Method used for calculating variation. Is used to look for the coordinates in adata.uns[<method>]. Default: "pca".
    n_pcs : int, optional
        Number of components to plot. Default: 20.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure is created. Default: None.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        # TODO: check if ax is an ax object
        pass

    if method not in adata.uns:
        raise KeyError("The given method '{0}' is not found in adata.uns. Please make sure to run the method before plotting variance.")

    # Get variance from object
    var_explained = adata.uns["pca"]["variance_ratio"][:n_pcs]
    var_explained = var_explained * 100  # to percent

    # Plot barplot of variance
    sns.barplot(x=list(range(1, len(var_explained) + 1)),
                y=var_explained,
                color="limegreen",
                ax=ax)

    # Finalize plot
    ax.set_xlabel('PCs', fontsize=12)
    ax.set_ylabel("Variance explained (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, size=7)
    ax.set_axisbelow(True)

    return ax


def search_umap_parameters(adata,
                           dist_range=(0.1, 0.4, 0.1),
                           spread_range=(2.0, 3.0, 0.5),
                           metacol="Sample", n_components=2, verbose=True, threads=4, save=None):
    """
    Plot a grid of different combinations of min_dist and spread variables for UMAP plots.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    dist_range : tuple
        Range of 'min_dist' parameter values to test. Must be a tuple in the form (min, max, step).  Default: (0.1, 0.4, 0.1)
    spread_range : tuple
        Range of 'spread' parameter values to test. Must be a tuple in the form (min, max, step).  Default: (2.0, 3.0, 0.5)
    metacol : str
        Name of the column in adata.obs to color by. Default: "Sample".
    n_components : int
        Number of components in UMAP calculation. Default: 2.
    verbose : bool
        Print progress to console. Default: True.
    threads : int
        Number of threads to use for UMAP calculation. Default: 4.
    save : str
        Path to save the figure to. Default: None.

    Example
    --------
    .. plot::
        :context: close-figs

        pl.search_umap_parameters(adata, dist_range=(0.1, 0.4, 0.1),
                                         spread_range=(2.0, 3.0, 0.5),
                                         metacol="bulk_labels")
    """

    adata = sctoolbox.analyser.get_minimal_adata(adata)  # remove data to save memory

    if len(dist_range) != 3:
        raise ValueError("The parameter 'dist_range' must be a tuple in the form (min, max, step)")
    if len(spread_range) != 3:
        raise ValueError("The parameter 'spread_range' must be a tuple in the form (min, max, step)")

    dist_min, dist_max, dist_step = dist_range
    spread_min, spread_max, spread_step = spread_range

    # Check validity of parameters
    if dist_step > dist_max - dist_min:
        raise ValueError("'step' of dist_range is larger than 'max' - 'min'. Please adjust.")
    if spread_step > spread_max - spread_min:
        raise ValueError("'step' of spread_range is larger than 'max' - 'min'. Please adjust.")

    # Setup parameters to loop over
    dists = np.arange(dist_min, dist_max, dist_step)
    dists = np.around(dists, 2)
    spreads = np.arange(spread_min, spread_max, spread_step)
    spreads = np.around(spreads, 2)

    # Calculate umap for each combination of spread/dist
    pool = mp.Pool(threads)
    jobs = {}
    for i, spread in enumerate(spreads):  # rows
        for j, dist in enumerate(dists):  # columns
            job = pool.apply_async(sc.tl.umap, args=(adata, ), kwds={"min_dist": dist,
                                                                     "spread": spread,
                                                                     "n_components": n_components,
                                                                     "copy": True})
            jobs[(i, j)] = job
    pool.close()

    utils.monitor_jobs(jobs, "Computing UMAPs")
    pool.join()

    # Figure with rows=spread, cols=dist
    fig, axes = plt.subplots(len(spreads), len(dists), figsize=(4 * len(dists), 4 * len(spreads)))
    axes = np.array(axes).reshape((-1, 1)) if len(dists) == 1 else axes    # reshape 1-column array
    axes = np.array(axes).reshape((1, -1)) if len(spreads) == 1 else axes  # reshape 1-row array

    # Fill in UMAPs
    for i, spread in enumerate(spreads):  # rows
        for j, dist in enumerate(dists):  # columns

            # Add precalculated UMAP to adata
            adata.obsm["X_umap"] = jobs[(i, j)].get().obsm["X_umap"]

            if verbose is True:
                print(f"Plotting umap for spread={spread} and dist={dist} ({i*len(dists)+j+1}/{len(dists)*len(spreads)})")

            # Set legend loc for last column
            if i == 0 and j == (len(dists) - 1):
                legend_loc = "left"
            else:
                legend_loc = "none"

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="No data for colormapping provided via 'c'*")
                sc.pl.umap(adata, color=metacol, title='', legend_loc=legend_loc, show=False, ax=axes[i, j])

            if j == 0:
                axes[i, j].set_ylabel(f"spread: {spread}")
            else:
                axes[i, j].set_ylabel("")

            if i == 0:
                axes[i, j].set_title(f"min_dist: {dist}")

            axes[i, j].set_xlabel("")

    plt.tight_layout()
    save_figure(save)

    return axes


def search_clustering_parameters(adata,
                                 method="leiden",
                                 resolution_range=(0.1, 1, 0.1),
                                 embedding="X_umap",
                                 ncols=3,
                                 verbose=True,
                                 save=None):
    """
    Plot a grid of different resolution parameters for clustering.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    method : str, default: "leiden"
        Clustering method to use. Can be one of 'leiden' or 'louvain'.
    resolution_range : tuple, default: (0.1, 1, 0.1)
        Range of 'resolution' parameter values to test. Must be a tuple in the form (min, max, step).
    embedding : str, default: "X_umap".
        Embedding method to use. Must be a key in adata.obsm.
    ncols : int, default: 3
        Number of columns in the grid.
    verbose : bool, default: True
        Print progress to console.
    save : str, default None
        Path to save figure.

    Returns
    -------
    axarr : array of matplotlib.axes.Axes
        Array of axes objects containing the plot(s).
    """

    # Check input
    if len(resolution_range) != 3:
        raise ValueError("The parameter 'dist_range' must be a tuple in the form (min, max, step)")

    # Check validity of parameters
    res_min, res_max, res_step = resolution_range
    if res_step > res_max - res_min:
        raise ValueError("'step' of resolution_range is larger than 'max' - 'min'. Please adjust.")

    # Check that coordinates for embedding is available in .obsm
    if embedding not in adata.obsm:
        raise KeyError(f"The embedding '{embedding}' was not found in adata.obsm. Please adjust this parameter.")

    # Check that method is valid
    if method == "leiden":
        cl_function = sc.tl.leiden
    elif method == "louvain":
        cl_function = sc.tl.louvain
    else:
        raise ValueError(f"Method '{method} is not valid. Method must be one of: leiden, louvain")

    # Setup parameters to loop over
    res_min, res_max, res_step = resolution_range
    resolutions = np.arange(res_min, res_max, res_step)
    resolutions = np.around(resolutions, 2)

    # Figure with given number of cols
    ncols = min(ncols, len(resolutions))  # number of resolutions caps number of columns
    nrows = int(np.ceil(len(resolutions) / ncols))
    fig, axarr = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axarr = np.array(axarr).reshape((-1, 1)) if ncols == 1 else axarr    # reshape 1-column array
    axarr = np.array(axarr).reshape((1, -1)) if nrows == 1 else axarr  # reshape 1-row array

    axes = axarr.flatten()

    for i, res in enumerate(resolutions):

        if verbose is True:
            print(f"Plotting umap for resolution={res} ({i+1} / {len(resolutions)})")

        # Run clustering
        key_added = method + "_" + str(round(res, 2))
        cl_function(adata, resolution=res, key_added=key_added)
        adata.obs[key_added] = sctoolbox.analyser.rename_categories(adata.obs[key_added])  # rename to start at 1
        n_clusters = len(adata.obs[key_added].cat.categories)

        # Plot embedding
        title = f"Resolution: {res} (clusters: {n_clusters})\ncolumn name: {key_added}"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="No data for colormapping provided via 'c'*")
            sc.pl.embedding(adata, embedding, color=key_added, ax=axes[i], legend_loc="on data", title=title, show=False)

    # Hide plots not filled in
    for ax in axes[len(resolutions):]:
        ax.axis('off')

    plt.tight_layout()
    save_figure(save)

    return axarr


def plot_group_embeddings(adata, groupby, embedding="umap", ncols=4, save=None):
    """ Plot a grid of embeddings (UMAP/tSNE/PCA) per group of cells within 'groupby'.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix object.
    groupby : str
        Name of the column in adata.obs to group by.
    embedding : str
        Embedding to plot. Must be one of "umap", "tsne", "pca". Default: "umap".
    ncols : int
        Number of columns in the figure. Default: 4.
    save : str
        Path to save the figure. Default: None.
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
    save_figure(save)

    return axarr


def compare_embeddings(adata_list, var_list, embedding="umap", adata_names=None, **kwargs):
    """ Compare embeddings across different adata objects. Plots a grid of embeddings with the different adatas on the
    x-axis, and colored variables on the y-axis.

    Parameters
    ----------
    adata_list : list of anndata.AnnData
        List of AnnData objects to compare.
    var_list : list of str
        List of variables to color in plot.
    embedding : str
        Embedding to plot. Must be one of "umap", "tsne" or "pca". Default: "umap".
    adata_names : list of str
        List of names for the adata objects. Default: None (adatas will be named adata_1, adata_2, etc.).
    kwargs : arguments
        Additional arguments to pass to sc.pl.umap/sc.pl.tsne/sc.pl.pca.
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
        print(f"The following variables from var_list were not found in any of the adata objects: {list(not_found)}. These will be excluded.")

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
                axes[j, i].set_title(adata_names[i])
            else:
                axes[j, i].set_title("")

            axes[j, i].set_xlabel("")

            _make_square(axes[j, i])

    # fig.tight_layout()
    return axes


def _get_3d_dotsize(n):
    """ Utility to get the dotsize for a given number of points. """
    if n < 1000:
        size = 12
    if n < 10000:
        size = 8
    else:
        size = 3

    return size


def plot_3D_UMAP(adata, color, save):
    """ Save 3D UMAP plot to a html file.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    color : str
        Variable to color in plot. Must be a column in adata.obs or an index in adata.var.
    save : str
        Save prefix. Plot will be saved to <save>.html.
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
        categories = df["category"].astype("category").cat.categories
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
    if save is not None:
        path = save + ".html"
        fig.write_html(path)

    print(f"Plot written to '{path}'")


#############################################################################
#                   Other overview plots for expression                     #
#############################################################################


def n_cells_pieplot(adata, groupby,
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
    **kwargs
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

    save_figure(save)

    return axarr


def group_expression_boxplot(adata, gene_list, groupby, figsize=None):
    """
    Plot a boxplot showing gene expression of genes in `gene_list` across the groups in `groupby`. The total gene expression is quantile normalized
    per group, and are subsequently normalized to 0-1 per gene across groups.

    Parameters
    ------------
    adata : anndata.AnnData object
        An annotated data matrix object containing counts in .X.
    gene_list : list
        A list of genes to show expression for.
    groupby : str
        A column in .obs for grouping cells into groups on the x-axis
    figsize : tuple, optional
        Control the size of the output figure, e.g. (6,10). Default: None (matplotlib default).
    """

    # Obtain pseudobulk
    gene_table = sctoolbox.utilities.pseudobulk_table(adata, groupby)

    # Normalize across clusters
    gene_table = qnorm.quantile_normalize(gene_table, axis=1)

    # Normalize to 0-1 across groups
    scaler = MinMaxScaler()
    df = gene_table.T
    df[df.columns] = scaler.fit_transform(df[df.columns])
    gene_table = df

    # Melt to long format
    gene_table_melted = gene_table.reset_index().melt(id_vars="index", var_name="gene")
    gene_table_melted.rename(columns={"index": groupby}, inplace=True)

    # Subset to input gene list
    gene_table_melted = gene_table_melted[gene_table_melted["gene"].isin(gene_list)]

    # Sort by median
    medians = gene_table_melted.groupby(groupby).median()
    medians.columns = ["medians"]
    gene_table_melted_sorted = gene_table_melted.merge(medians, left_on=groupby, right_index=True).sort_values("medians", ascending=False)

    # Joined figure with all
    fig, ax = plt.subplots(figsize=figsize)
    g = sns.boxplot(data=gene_table_melted_sorted, x=groupby, y="value", ax=ax, color="darkgrey")
    ax.set_ylabel("Normalized expression")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    return g


def group_heatmap(adata, groupby, gene_list=None, save=None, figsize=None):
    """ Plot a heatmap of gene expression across groups in `groupby`. The rows are z-scored per gene

    Parameters
    ----------
    adata : anndata.AnnData object
        An annotated data matrix object containing counts in .X.
    groupby : str
        A column in .obs for grouping cells into groups on the x-axis
    gene_list : list, optional
        A list of genes to show expression for. Default: None (all genes)
    save : str, optional
        Save the figure to a file. Default: None (do not save)
    figsize : tuple, optional
        Control the size of the output figure, e.g. (6,10). Default: None (matplotlib default).

    Returns
    -------
    g : seaborn.clustermap
        The seaborn clustermap object
    """

    # Obtain pseudobulk
    gene_table = sctoolbox.utilities.pseudobulk_table(adata, groupby)

    # Subset to input gene list
    if gene_list is not None:
        gene_table = gene_table[gene_list]

    # Z-score
    gene_table = utils.table_zscore(gene_table)

    # Plot heatmap
    g = sns.heatmap(gene_table, figsize=figsize, xticklabels=True, yticklabels=True, cmap="RdBu_r", center=0)  # center=0, vmin=-2, vmax=2)

    utils.save_figure(save)

    return g


#############################################################################
#                          Quality control plotting                         #
#############################################################################

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

    utils.save_figure(save)

    return g


def violinplot(table, y, color_by=None, hlines=None, colors=None, ax=None, title=None, ylabel=True):
    """
    Creates a violinplot. With optional horizontal lines for each violin.

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


def qc_violins(anndata, thresholds, colors=None, filename=None, ncols=3, figsize=None, dpi=300):
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
            index (two columns)
                - Name of anndata.var or anndata.obs column.
                - Name of origin. Either "obs" or "var".
            1st column - Threshold number(s) defining violinplot lines. Either None, single number or list of numbers.
            2nd column - Name of anndata.var or anndata.obs column used for color grouping or None to disable.
    colors : list of str, default None
        List of colors for the violins.
    filename : str, default None
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
    if filename:
        save_figure(filename)


def anndata_overview(adatas,
                     color_by,
                     plots=["PCA", "PCA-var", "UMAP", "LISI"],
                     figsize=None,
                     max_clusters=20,
                     output=None,
                     dpi=300):
    """
    Create a multipanel plot comparing PCA/UMAP/tSNE/(...) plots for different adata objects.

    Parameters
    ------------
    adatas : dict of anndata.AnnData
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

    Example
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
        """ Annotate row in figure. """
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
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, constrained_layout=True)
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
                    if adata.obs[color].dtype.name == "category":
                        categories += list(adata.obs[color].cat.categories)
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
                    plot_pca_variance(adata, ax=ax)  # this plot takes no color

                elif plot_type == "LISI":

                    # Find any LISI scores in adata.obs
                    lisi_columns = [col for col in adata.obs.columns if col.startswith("LISI_score")]

                    if len(lisi_columns) == 0:
                        e = f"No LISI scores found in adata.obs for '{name}'"
                        e += "Please run 'sctoolbox.analyser.wrap_batch_evaluation()' or remove LISI from the plots list"
                        raise ValueError(e)

                    # Plot LISI scores
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning, message="iteritems is deprecated*")
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

                    ax.legend(lines[:max_clusters], labels[:max_clusters],
                              title=color, ncols=n_cols, frameon=False,
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
    save_figure(output, dpi=dpi)

    return axs


def boxplot(dt, show_median=True, ax=None):
    """
    Generate one plot containing one box per column. The median value is shown.

    Parameter
    ---------
    dt : pandas.DataFrame
        pandas datafame containing numerical values in every column.
    show_median: boolean, default True
        If True show median value as small box inside the boxplot.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure is created. Default: None.

    Returns
    -------
    AxesSubplot
        containing boxplot for every column.
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


def grouped_violin(adata, x, y=None, groupby=None, figsize=None, title=None, style="violin", save=None, **kwargs):
    """
    Create violinplot of values across cells in an adata object grouped by x and 'groupby'.
    Can for example show the expression of one gene across groups (x = obs_group, y = gene),
    expression of multiple genes grouped by cell type (x = gene_list, groupby = obs_cell_type),
    or values from adata.obs across cells (x = obs_group, y = obs_column).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    x : str or list
        Column name in adata.obs or gene name(s) in adata.var.index to group by on the x-axis. Multiple gene names can be given in a list.
    y : str, default None
        A column name in adata.obs or a gene in adata.var.index to plot values for. Only needed if x is a column in adata.obs.
    groupby : str, default None
        Column name in adata.obs to create grouped violins. If None, a single violin is plotted per group in 'x'.
    figsize : tuple, default None
        Figure size.
    title : str, default None
        Title of the plot. If None, no title is shown.
    style : str, default "violin"
        Plot style. Either "violin" or "boxplot".
    save : str, default None
        Path to save the figure to. If None, the figure is not saved.
    kwargs : arguments, optional
        Additional arguments passed to seaborn.violinplot or seaborn.boxplot.

    Returns
    -------
    matplotlib.axes.Axes
    """

    if isinstance(x, str):
        x = [x]
    x = list(x)  # convert to list incase x was a numpy array or other iterable

    # Establish if x is a column in adata.obs or a gene in adata.var.index
    x_assignment = []
    for element in x:
        if element not in adata.obs.columns and element not in adata.var.index:
            raise ValueError(f"{element} is not a column in adata.obs or a gene in adata.var.index")
        else:
            if element in adata.obs.columns:
                x_assignment.append("obs")
            else:
                x_assignment.append("var")

    if len(set(x_assignment)) > 1:
        raise ValueError("x must be either a column in adata.obs or all genes in adata.var.index")
    else:
        x_assignment = x_assignment[0]

    # Establish if y is a column in adata.obs or a gene in adata.var.index
    if x_assignment == "obs" and y is None:
        raise ValueError("Because 'x' is a column in obs, 'y' must be given as parameter")

    if y is not None:
        if y in adata.obs.columns:
            y_assignment = "obs"
        elif y in adata.var.index:
            y_assignment = "var"
        else:
            raise ValueError(f"y' ({y}) was not found in either adata.obs or adata.var.index")

    # Create obs table with column
    obs_cols = [col for col in [x[0], y, groupby] if col is not None and col in adata.obs.columns]
    obs_table = adata.obs.copy()[obs_cols]  # creates a copy

    for element in x + [y]:
        if element in adata.var.index:
            gene_idx = np.argwhere(adata.var.index == element)[0][0]
            vals = adata.X[:, gene_idx].todense().A1
            obs_table[element] = vals

    # Convert table to long format if the x-axis contains gene expressions
    if x_assignment == "var":
        index_name = obs_table.index.name
        id_vars = [index_name, groupby]
        id_vars = id_vars + x if x_assignment == "obs" else id_vars
        id_vars = [v for v in id_vars if v is not None]

        obs_table.reset_index(inplace=True)
        obs_table = obs_table.melt(id_vars=id_vars, value_vars=x,
                                   var_name="Gene", value_name="Expression")
        x_var = "Gene"
        y_var = "Expression"

    else:
        x_var = x[0]
        y_var = y

    # Plot expression from obs table
    _, ax = plt.subplots(figsize=figsize)
    if style == "violin":
        kwargs["scale"] = "width" if "scale" not in kwargs else kwargs["scale"]  # set defaults
        kwargs["cut"] = 0 if "cut" not in kwargs else kwargs["cut"]
        sns.violinplot(data=obs_table, x=x_var, y=y_var, hue=groupby, ax=ax, **kwargs)
    elif style == "boxplot":
        sns.boxplot(data=obs_table, x=x_var, y=y_var, hue=groupby, ax=ax, **kwargs)
    else:
        raise ValueError(f"Style '{style}' is not valid for this function. Style must be one of 'violin' or 'boxplot'")

    if groupby is not None:
        ax.legend(title=groupby, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)  # Set location of legend

    # Final adjustments of labels
    if x_assignment == "obs" and y_assignment == "var":
        ax.set_ylabel(ax.get_ylabel() + " expression")

    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # save figure if output is given
    save_figure(save)

    return ax


#############################################################################
# ------------------------------ Dotplot ---------------------------------- #
#############################################################################

def _scale_values(array, mini, maxi):
    """
    Small utility to scale values in array to a given range.

    Parameters
    ----------
    array : np.ndarray
        Array to scale.
    mini : float
        Minimum value of the scale.
    maxi : float
        Maximum value of the scale.
    """
    val_range = array.max() - array.min()
    a = (array - array.min()) / val_range
    return a * (maxi - mini) + mini


def _plot_size_legend(ax, val_min, val_max, radius_min, radius_max, title):
    """ Fill in an axis with a legend for the dotplot size scale.

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


def clustermap_dotplot(table, x, y, color, size, save=None, **kwargs):
    """ Plot a heatmap with dots instead of cells which can contain the dimension of "size".

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
    kwargs : arguments
        Additional arguments to pass to seaborn.clustermap.
    """

    # This code is very hacky
    # Major todo is to get better control of location of legends
    # automatic scaling of figsize
    # and to make the code more flexible, potentially using a class

    # Create pivots with colors/size
    color_pivot = pd.pivot(table, index=y, columns=x, values=color)
    size_pivot = pd.pivot(table, index=y, columns=x, values=size)

    # Plot clustermap of values
    g = sns.clustermap(color_pivot, yticklabels=True, cmap="bwr",
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
    col = PatchCollection(circles, array=color_mat.flatten(), cmap="bwr")
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
    utils.save_figure(save)

    return g


def marker_gene_clustering(adata, groupby, marker_genes_dict, show_umap=True, save=None, figsize=None):
    """ Plot an overview of marker genes and clustering.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    groupby : `str`
        Key in `adata.obs` for which to plot the clustering.
    marker_genes_dict : `dict`
        Dictionary of marker genes to plot. Keys are the names of the groups and values are lists of marker genes.
    show_umap : `bool`, optional (default: `True`)
        Whether to show a UMAP plot on the left.
    save : `str`, optional (default: `None`)
        If given, save the figure to this path.
    figsize : `tuple`, optional (default: `None`)
        Size of the figure. If `None`, use default size.
    """

    i = 0
    if show_umap:
        figsize = (12, 6) if figsize is None else figsize
        fig, axarr = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 2]})

        # Plot UMAP colored by groupby on the left
        sc.pl.umap(adata, color=groupby, ax=axarr[0], legend_loc="on data", show=False)
        axarr[i].set_aspect('equal')
        i += 1

    else:
        figsize = (6, 6) if figsize is None else figsize
        fig, axarr = plt.subplots(1, 1, figsize=figsize)
        axarr = [axarr]  # Make sure axarr can be indexed

    # Make sure all genes are in the data
    marker_genes_dict = marker_genes_dict.copy()
    for group in list(marker_genes_dict.keys()):
        genes = marker_genes_dict[group]
        marker_genes_dict[group] = [gene for gene in genes if gene in adata.var_names]
        if len(marker_genes_dict[group]) == 0:
            del marker_genes_dict[group]  # Remove group if no genes are left in the data

    # Plot marker gene expression on the right
    ax = sc.pl.dotplot(adata, marker_genes_dict, groupby=groupby, show=False, dendrogram=True, ax=axarr[i])
    ax["mainplot_ax"].set_ylabel(groupby)
    ax["mainplot_ax"].set_xticklabels(ax["mainplot_ax"].get_xticklabels(), ha="right", rotation=45)

    for text in ax["gene_group_ax"]._children:
        text._rotation = 45
        text._horizontalalignment = "left"

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2)

    # Save figure
    utils.save_figure(save)

    return axarr


def umap_pub(adata, color=None, title=None, save=None, **kwargs):
    """
    Plot a publication ready UMAP without spines, but with a small UMAP1/UMAP2 legend.

    Parameters
    ----------
    adata :anndata.AnnData
        Annotated data matrix.
    color : str or lst of str, default None
        Key for annotation of observations/cells or variables/genes.
    title : str, default None
        Title of the plot. Default is no title.
    save : str, default None
        Filename to save the figure.
    kwargs : dict
        Additional arguments passed to `sc.pl.umap`.

    Example
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
        arrow_len_y = yrange * 0.15
        arrow_len_x = xrange * 0.15

        ax.annotate("", xy=(xmin, ymin), xytext=(xmin, ymin + arrow_len_y), arrowprops=dict(arrowstyle="<-", shrinkB=0))  # UMAP2 / y-axis
        ax.annotate("", xy=(xmin, ymin), xytext=(xmin + arrow_len_x, ymin), arrowprops=dict(arrowstyle="<-", shrinkB=0))  # UMAP1 / x-axis

        # Adjust aspect ratio
        _make_square(ax)

    # Save figure
    utils.save_figure(save)

    return axarr


def add_figure_title(axarr, title, y=1.3, fontsize=16):
    """
    Add a figure title to the top of a multi-axes figure.

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

    Returns
    -------
    None - adds a title to the figure directly.

    Example
    --------
    .. plot::
        :context: close-figs

        axes = sc.pl.umap(adata, color=["louvain", "condition"], show=False)
        pl.add_figure_title(axes, "UMAP plots")
    """

    # If only one axes is passed, convert to list
    if type(axarr).__name__.startswith("Axes"):
        axarr = [axarr]

    # Get figure
    fig = plt.gcf()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get bounding box of axes in relation to first axes
    trans_data_inv = axarr[0].transData.inverted()  # from display to data
    bbox_list = [ax.get_window_extent(renderer=renderer).transformed(trans_data_inv) for ax in axarr]

    # FInx y/x positions based on bboxes
    ty = np.max([bbox.y1 for bbox in bbox_list])
    ty *= y

    xmin = np.min([bbox.x0 for bbox in bbox_list])
    xmax = np.max([bbox.x1 for bbox in bbox_list])
    tx = np.mean([xmin, xmax])

    # Add text
    _ = axarr[0].text(tx, ty, title, va="bottom", ha="center", fontsize=fontsize)


def add_labels(x, y, labels, ax=None):
    """ Add labels to a scatter plot.

    Parameters
    ----------

    """

    if ax is None:
        ax = plt.gca()

    texts = []
    for i, label in enumerate(labels):

        text = ax.annotate(label, (x[i], y[i]))
        texts.append(text)

    # Adjust text positions
    # to-do
