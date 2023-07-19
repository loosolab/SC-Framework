import numpy as np
import scanpy as sc

import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


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

    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    group_names = adata.obs[groupby].astype("category").cat.categories.tolist()
    adata_subsets = {name: adata[adata.obs[groupby] == name] for name in group_names}

    logger.debug("Split adata into {} subsets based on column '{}'".format(len(adata_subsets), groupby))

    return adata_subsets


@deco.log_anndata
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


@deco.log_anndata
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

    logger.info(f"The adata object was loaded from: {adata_input}")

    return adata


@deco.log_anndata
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

    # Save adata
    adata_output = settings.full_adata_output_prefix + path
    adata.write(filename=adata_output)

    logger.info(f"The adata object was saved to: {adata_output}")


def add_uns_info(adata, key, value, how="overwrite"):
    """
    Add information to adata.uns['sctoolbox']. This is used for logging the parameters and options of different steps in the analysis.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object.
    key : str or list
        The key to add to adata.uns['sctoolbox']. If the key is a list, it represents a path within a nested dictionary.
    value : any
        The value to add to adata.uns['sctoolbox'].
    how : str, default overwrite
        When set to "overwrite" provided key will be overwriten. If "append" will add element to existing list or dict.
    """

    if "sctoolbox" not in adata.uns:
        adata.uns["sctoolbox"] = {}

    if isinstance(key, str):
        key = [key]

    d = adata.uns["sctoolbox"]
    for k in key[:-1]:  # iterate over all keys except the last one
        if k not in d:
            d[k] = d.get(k, {})
        d = d[k]

    # Add value to last key
    last_key = key[-1]
    if how == "overwrite":
        d[last_key] = value  # last key contains value

    elif how == "append":
        if key[-1] not in d:
            d[last_key] = value  # initialize with a value if key does not exist

        else:  # append to existing key

            current_value = d[last_key]

            if isinstance(value, dict) and not isinstance(current_value, dict):
                nested = "adata.uns['sctoolbox'][" + "][".join(key) + "]"
                raise ValueError(f"Cannot append {value} to {nested} because it is not a dict.")
            elif type(current_value).__name__ == "ndarray":  # convert numpy array to list in order to use "append"/extend"
                d[last_key] = list(current_value)
            else:
                d[last_key] = [current_value]

            # Append/extend/update value
            if isinstance(value, list):
                d[last_key].extend(value)
            elif isinstance(value, dict):
                d[last_key].update(value)  # update dict
            else:
                d[last_key].append(value)  # value is a single value

            # If list; remove duplicates and keep the last occurrence
            if isinstance(d[last_key], Sequence):
                d[last_key] = list(reversed(OrderedDict.fromkeys(reversed(d[last_key]))))  # reverse list to keep last occurrence instead of first
