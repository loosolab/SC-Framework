"""Module for cell clustering."""
import scanpy as sc
import numpy as np
import warnings
import matplotlib.pyplot as plt
import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco

from beartype.typing import Literal, Optional, Tuple
from beartype import beartype
import numpy.typing as npt


@deco.log_anndata
@beartype
def recluster(adata: sc.AnnData,
              column: str,
              clusters: str | list[str],
              task: Literal["join", "split"] = "join",
              method: Literal["leiden", "louvain"] = "leiden",
              resolution: float | int = 1,
              key_added: Optional[str] = None,
              plot: bool = True,
              embedding: str = "X_umap") -> None:
    """
    Recluster an anndata object based on an existing clustering column in .obs.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    column : str | list[str]
        Column in adata.obs to use for re-clustering.
    clusters : str | list[str]
        Clusters in `column` to re-cluster.
    task : Literal["join", "split"], default "join"
        Task to perform. Options are:
        - "join": Join clusters in `clusters` into one cluster.
        - "split": Split clusters in `clusters` are merged and then reclustered using `method` and `resolution`.
    method : Literal["leiden", "louvain"], default "leiden"
        Clustering method to use. Must be one of "leiden" or "louvain".
    resolution : float, default 1
        Resolution parameter for clustering.
    key_added : Optional[str], default None
        Name of the new column in adata.obs. If None, the column name is set to `<column>_recluster`.
    plot : bool, default True
        If a plot should be generated of the re-clustering.
    embedding : str, default 'X_umap'
        Select which embeding should be used.

    Raises
    ------
    KeyError
        If the given embeding is not in the data.
        1. If the given embedding is not in the data.
        2. If given column is not found in adata.obs
    ValueError
        If a given cluster is not found in the adata.

    Examples
    --------
    .. plot::
        :context: close-figs

        sctoolbox.tools.clustering.recluster(adata, column="louvain", clusters=["1", "5"], task="join", method="leiden", resolution=1.5, plot=True)

    .. plot::
        :context: close-figs

        sctoolbox.tools.clustering.recluster(adata, column="louvain", clusters=["2", "6"], task="split", method="leiden", resolution=1.5, plot=True)
    """

    adata_copy = adata.copy()

    # --- Get ready --- #
    # check if column is in adata.obs
    if column not in adata.obs.columns:
        raise KeyError(f"Column {column} not found in adata.obs")

    # Decide key_added
    if key_added is None:
        key_added = f"{column}_recluster"

    # Check that clusters is a list
    if isinstance(clusters, str):
        clusters = [clusters]

    # Check that method is valid
    if method == "leiden":
        # set future defaults to omit warning
        def cl_function(*args, **kwargs):
            sc.tl.leiden(*args, **kwargs, flavor="igraph", n_iterations=2)
    elif method == "louvain":
        cl_function = sc.tl.louvain

    # Check if clusters are found in column
    if not set(clusters).issubset(adata_copy.obs[column]):
        invalid_clusters = set(clusters) - set(adata_copy.obs[column])
        raise ValueError(f"Cluster(s) not found in adata.obs['{column}']: {invalid_clusters}")

    # --- Start reclustering --- #
    if task == "join":
        translate = {cluster: clusters[0] for cluster in clusters}
        # add rest of the clusters otherwise they will be NA
        translate.update({cluster: cluster for cluster in adata.obs[column] if cluster not in clusters})
        adata.obs[key_added] = adata.obs[column].map(translate).astype("category")
    elif task == "split":
        cl_function(adata, restrict_to=(column, clusters), resolution=resolution, key_added=key_added)

    adata.obs[key_added] = utils.tables.rename_categories(adata.obs[key_added])  # rename to start at 1

    # --- Plot reclustering before/after --- #
    if plot is True:

        # Check that coordinates for embedding is available in .obsm
        if embedding not in adata.obsm:
            embedding = f"X_{embedding}"
            if embedding not in adata.obsm:
                raise KeyError(f"The embedding '{embedding}' was not found in adata.obsm. Please adjust this parameter.")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="No data for colormapping provided via 'c'*")

            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            sc.pl.embedding(adata_copy, basis=embedding, color=column, ax=ax[0], show=False, legend_loc="on data")
            ax[0].set_title(f"Before re-clustering\n(column name: '{column}')")

            sc.pl.umap(adata, color=key_added, ax=ax[1], show=False, legend_loc="on data")
            ax[1].set_title(f"After re-clustering\n (column name: '{key_added}')")


def gini(x: npt.ArrayLike) -> float:
    """
    Calculate the Gini coefficient of a numpy array.

    Parameters
    ----------
    x : npt.ArrayLike
        Array to calculate Gini coefficient for.

    Returns
    -------
    float
        Gini coefficient.
    """
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))

    return total / (len(x)**2 * np.mean(x))


@deco.log_anndata
def calc_ragi(adata: sc.AnnData, condition_column: str = 'clustering', binary_layer: Optional[str] = None) -> Tuple[sc.AnnData, np.float64]:
    """
    Calculate the RAGI score over all clusters in the adata.

    The RAGI score is a measure of how well a cluster is defined by a set of genes.
    The score is the mean of the Gini coefficients of the gene enrichments across the clusters.
    The functions uses binary sparse matrices ONLY. If the data is not binary, use `sctoolbox.utils.binarize`.
    Binary layers can be selected using the `binary_layer` parameter.
    The adata.var table also needs the total counts for each gene.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    condition_column : str
        Column in `adata.obs` to use for clustering.
    binary_layer : Optional[str], default None
        Layer in `adata.layers` to use for calculating gene enrichment. If None, the raw layer is used.

    Returns
    -------
    Tuple[sc.AnnData, np.float64]
        Annotated data matrix with the Gini coefficients score in `adata.var` and RAGI score.
    """

    # copy adata
    adata_copy = adata.copy()

    # check if total_counts is in adata.var
    if 'total_counts' not in adata_copy.var:
        adata_copy.var["total_counts"] = adata_copy.X.sum(axis=1)

    # get conditions
    conditions = adata_copy.obs[condition_column].unique()
    enrichment_columns = []

    # loop over conditions
    for cond in conditions:

        # slice by condition
        adata_slice = adata_copy.obs_names[adata_copy.obs[condition_column] == cond]  # Check if string works
        subdata = adata_copy[adata_slice, :].copy()
        # select all peaks available in the cluster
        # check if binary layer is set
        if binary_layer:
            # check if binary layer is available
            if binary_layer in subdata.layers.keys():
                # count gene occurence
                genes = subdata.var[subdata.layers[binary_layer].sum(axis=0).A1 > 1].copy()

                # count total genes
                gene_counts = subdata.layers[binary_layer].sum(axis=0).A1[
                    subdata.layers[binary_layer].sum(axis=0).A1 > 1].copy()
            else:
                print('binary layer not available!')
        else:
            # count gene occurence
            genes = subdata.var[subdata.X.sum(axis=0).A1 > 1].copy()
            # count total genes
            gene_counts = subdata.X.sum(axis=0).A1[subdata.X.sum(axis=0).A1 > 1].copy()

        # add counts/cluster to genes
        genes.loc[:, 'cluster_counts_' + str(cond)] = gene_counts
        # calc enrichment
        genes.loc[:, 'enrichment_' + str(cond)] = genes['cluster_counts_' + str(cond)] / genes['total_counts']

        # remove old tables
        # join results
        adata_copy.var = adata_copy.var.join(genes['cluster_counts_' + str(cond)])
        adata_copy.var = adata_copy.var.join(genes['enrichment_' + str(cond)])
        # add column names to list
        enrichment_columns.append('enrichment_' + str(cond))

    # get enrichment values
    enrichments = adata_copy.var[enrichment_columns].values
    enrichments[np.isnan(enrichments)] = 0

    # calculate gini coefficients
    gini_coefficients = []
    for enrichment in enrichments:
        gini_coefficients.append(gini(enrichment))

    adata_copy.var[condition_column + '_' + 'gini'] = gini_coefficients

    # calculate ragi score
    ragi_score = np.mean(gini_coefficients)

    return adata_copy, ragi_score
