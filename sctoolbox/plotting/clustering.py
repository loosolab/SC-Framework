import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import warnings

import sctoolbox.utils as utils
from sctoolbox.plotting.general import _save_figure
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


@deco.log_anndata
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
        Embedding method to use. Must be a key in adata.obsm. If not, will try to use f"X_{embedding}".
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

    Example
    --------
    .. plot::
        :context: close-figs

        import sctoolbox.plotting as pl

    .. plot::
        :context: close-figs

        pl.search_clustering_parameters(adata, method='louvain', resolution_range=(0.1, 2, 0.2), embedding='X_umap', ncols=3, verbose=True, save=None)
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
        embedding = f"X_{embedding}"
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
            logger.info(f"Plotting umap for resolution={res} ({i+1} / {len(resolutions)})")

        # Run clustering
        key_added = method + "_" + str(round(res, 2))
        cl_function(adata, resolution=res, key_added=key_added)
        adata.obs[key_added] = utils.rename_categories(adata.obs[key_added])  # rename to start at 1
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
    _save_figure(save)

    return axarr


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

    Example
    --------
    .. plot::
        :context: close-figs

        import scanpy as sc
        import sctoolbox.plotting as pl

    .. plot::
        :context: close-figs

        adata = sc.datasets.pbmc68k_reduced()
        marker_genes_dict = {"S": ["PCNA"], "G2M": ["HMGB2"]}

    .. plot::
        :context: close-figs

        pl.marker_gene_clustering(adata, "phase", marker_genes_dict, show_umap=True, save=None, figsize=None)
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
    _save_figure(save)

    return axarr
