"""Module for cell clustering."""
import scanpy as sc
import warnings
import matplotlib.pyplot as plt
import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco

from typing import Literal, Optional
from beartype import beartype


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
    ValueError:
        1. If clustering method is not valid.
        2. If task is not valid.
    KeyError:
        If the given embeding is not in the data.
    """

    adata_copy = adata.copy()

    # --- Get ready --- #
    # check if column is in adata.obs
    if column not in adata.obs.columns:
        raise ValueError(f"Column {column} not found in adata.obs")

    # Decide key_added
    if key_added is None:
        key_added = f"{column}_recluster"

    # Check that clusters is a list
    if isinstance(clusters, str):
        clusters = [clusters]

    # Check that method is valid
    if method == "leiden":
        cl_function = sc.tl.leiden
    elif method == "louvain":
        cl_function = sc.tl.louvain
    else:
        # Will not be called due to beartype checks
        raise ValueError(f"Method '{method} is not valid. Method must be one of: leiden, louvain")

    # TODO: Check if clusters are found in column

    # --- Start reclustering --- #
    if task == "join":
        translate = {cluster: clusters[0] for cluster in clusters}
        adata.obs[key_added] = adata.obs[column].replace(translate)

    elif task == "split":
        cl_function(adata, restrict_to=(column, clusters), resolution=resolution, key_added=key_added)

    else:
        # Will not be called due to beartype checks
        raise ValueError(f"Task '{task}' is not valid. Task must be one of: 'join', 'split'")

    adata.obs[key_added] = utils.rename_categories(adata.obs[key_added])  # rename to start at 1

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
