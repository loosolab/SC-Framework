import os
import numpy as np
import scanpy as sc

from sctoolbox._settings import settings
import sctoolbox.utils as utils


def get_adata_subsets(adata, groupby):
    """
    Split an anndata object into a dict of sub-anndata objects based on a grouping column.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object to split.
    groupby : str
        Column name in adata.obs to split by.

    Returns
    -------
    dict :
        Dictionary of anndata objects in the format {<group1>: anndata, <group2>: anndata, (...)}.
    """

    group_names = adata.obs[groupby].astype("category").cat.categories.tolist()
    adata_subsets = {name: adata[adata.obs[groupby] == name] for name in group_names}

    return adata_subsets


def add_expr_to_obs(adata, gene):
    """
    Add expression of a gene from adata.X to adata.obs as a new column.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object to add expression to.
    gene : str
        Gene name to add expression of.
    """

    boolean = adata.var.index == gene
    if sum(boolean) == 0:
        raise Exception(f"Gene {gene} not found in adata.var.index")

    else:
        idx = np.argwhere(boolean)[0][0]
        adata.obs[gene] = adata.X[:, idx].todense().A1


def shuffle_cells(adata, seed=42):
    """
    Shuffle cells in an adata object to improve plotting.
    Otherwise, cells might be hidden due plotting samples in order e.g. sample1, sample2, etc.

    Parameters
    -----------
    adata : anndata.AnnData
        Anndata object to shuffle cells in.

    Returns
    -------
    anndata.AnnData :
        Anndata object with shuffled cells.
    seed : int, default 42
        Seed for random number generator.
    """

    import random
    state = random.getstate()

    random.seed(seed)
    shuffled_barcodes = random.sample(adata.obs.index.tolist(), len(adata))
    adata = adata[shuffled_barcodes]

    random.setstate(state)  # reset random state

    return adata


def get_minimal_adata(adata):
    """ Return a minimal copy of an anndata object e.g. for estimating UMAP in parallel.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.

    Returns
    -------
    anndata.AnnData
        Minimal copy of anndata object.
    """

    adata_minimal = adata.copy()
    adata_minimal.X = None
    adata_minimal.layers = None
    adata_minimal.raw = None

    return adata_minimal


def load_h5ad(path):
    """
    Load an anndata object from .h5ad file.

    Parameters
    ----------
    path : str
        Name of the file to load the anndata object. NOTE: Uses the internal 'sctoolbox.settings.adata_input_dir' + 'sctoolbox.settings.adata_input_prefix' as prefix.

    Returns
    -------
    anndata.AnnData :
        Loaded anndata object.
    """

    adata_input = settings.full_adata_input_prefix + path
    adata = sc.read_h5ad(filename=adata_input)

    print(f"The adata object was loaded from: {adata_input}")

    # Save information of source to adata
    utils.add_uns_info(adata, ["sctoolbox", "source"], os.path.abspath(adata_input))

    return adata


def save_h5ad(adata, path):
    """
    Save an anndata object to an .h5ad file.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object to save.
    path : str
        Name of the file to save the anndata object. NOTE: Uses the internal 'sctoolbox.settings.adata_output_dir' + 'sctoolbox.settings.adata_output_prefix' as prefix.
    """

    # Log user to adata.uns
    utils.initialize_uns(adata, "user")
    adata.uns["sctoolbox"]["user"].update({utils.get_user(): utils.get_datetime()})  # overwrites existing entry for each user

    # Save adata
    adata_output = settings.full_adata_output_prefix + path
    adata.write(filename=adata_output)

    print(f"The adata object was saved to: {adata_output}")
