
import numpy as np
import scanpy as sc
from collections.abc import Sequence  # check if object is iterable
from collections import OrderedDict
import scipy

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


def prepare_for_cellxgene(adata, keep_obs=None, keep_var=None, rename_obs=None, rename_var=None, inplace=False):
    """
    Prepares the given adata for cellxgene deployment.

    Parameters
    ----------
    adata : scanpy.Anndata
        Anndata object.
    keep_obs : list, default None
        adata.obs columns that should be kept. None to keep all.
    keep_var : list, default None
        adata.var columns that should be kept. None to keep all.
    rename_obs : dict or None, default None
        Dictionary of .obs columns to rename. Key is the old name, value the new one.
    rename_var : dict or None, default None
        Dictionary of .var columns to rename. Key is the old name, value the new one.
    inplace : bool, default False

    Returns
    -------
    scanpy.Anndata or None:
        Returns the deployment ready Anndata object.
    """
    out = adata if inplace else adata.copy()

    # TODO remove more adata internals not needed for cellxgene

    ##### .obs #####
    # remove obs columns
    if keep_obs:
        drop_obs = set(out.obs.columns) - set(keep_obs)

        out.obs.drop(columns=drop_obs, inplace=True)

    # rename obs columns
    if rename_obs:
        out.obs.rename(columns=rename_obs, inplace=True)

    for c in out.obs:
        if out.obs[c].dtype == 'Int32':
            out.obs[c] = out.obs[c].astype('float64')

    out.obs.index.names = ['index']

    ##### .var #####
    # remove var columns
    if keep_var:
        drop_var = set(out.var.columns) - set(keep_var)

        out.var.drop(columns=drop_var, inplace=True)

    # rename obs columns
    if rename_var:
        out.var.rename(columns=rename_var, inplace=True)

    for c in out.var:
        if out.var[c].dtype == 'Int32':
            out.var[c] = out.var[c].astype('float64')

    out.var.index.names = ['index']

    ###### .X ######
    # convert .X to sparse matrix if needed
    if not scipy.sparse.isspmatrix(out.X):
        out.X = scipy.sparse.csc_matrix(out.X)

    out.X = out.X.astype("float32")

    ##### .uns #####
    # apply color fix
    # https://github.com/chanzuckerberg/cellxgene/issues/2598
    for key in out.uns.keys():
        if key.endswith('colors'):
            out.uns[key] = np.array([(c if len(c) <= 7 else c[:-2]) for c in adata.uns[key]])

    if not inplace:
        return out
