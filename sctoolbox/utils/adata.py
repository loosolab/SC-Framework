"""anndata.AnnData related functions."""

import numpy as np
import scanpy as sc
from collections.abc import Sequence  # check if object is iterable
from collections import OrderedDict
import scipy
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import pandas as pd
from pathlib import Path

from beartype.typing import Optional, Any, Union, Collection, Mapping
from beartype import beartype

import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


@beartype
def get_adata_subsets(adata: sc.AnnData, groupby: str) -> dict[str, sc.AnnData]:
    """
    Split an anndata object into a dict of sub-anndata objects based on a grouping column.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to split.
    groupby : str
        Column name in adata.obs to split by.

    Returns
    -------
    dict[str, sc.AnnData]
        Dictionary of anndata objects in the format {<group1>: anndata, <group2>: anndata, (...)}.

    Raises
    ------
    ValueError
        If groupby is not found in `adata.obs.columns`.
    """

    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    group_names = adata.obs[groupby].astype("category").cat.categories.tolist()
    adata_subsets = {name: adata[adata.obs[groupby] == name] for name in group_names}

    logger.debug("Split adata into {} subsets based on column '{}'".format(len(adata_subsets), groupby))

    return adata_subsets


@deco.log_anndata
@beartype
def add_expr_to_obs(adata: sc.AnnData, gene: str) -> None:
    """
    Add expression of a gene from adata.X to adata.obs as a new column.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to add expression to.
    gene : str
        Gene name to add expression of.

    Raises
    ------
    Exception
        If the gene is not found in the adata object.
    """

    boolean = adata.var.index == gene
    if sum(boolean) == 0:
        raise Exception(f"Gene {gene} not found in adata.var.index")

    else:
        idx = np.argwhere(boolean)[0][0]
        adata.obs[gene] = adata.X[:, idx].todense().A1


@deco.log_anndata
@beartype
def shuffle_cells(adata: sc.AnnData, seed: int = 42) -> sc.AnnData:
    """
    Shuffle cells in an adata object to improve plotting.

    Otherwise, cells might be hidden due plotting samples in order e.g. sample1, sample2, etc.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to shuffle cells in.
    seed : int, default 42
        Seed for random number generator.

    Returns
    -------
    sc.AnnData
        Anndata object with shuffled cells.
    """

    import random
    state = random.getstate()

    random.seed(seed)
    shuffled_barcodes = random.sample(adata.obs.index.tolist(), len(adata))
    adata = adata[shuffled_barcodes]

    random.setstate(state)  # reset random state

    return adata


@beartype
def get_minimal_adata(adata: sc.AnnData) -> sc.AnnData:
    """
    Return a minimal copy of an anndata object e.g. for estimating UMAP in parallel.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.

    Returns
    -------
    sc.AnnData
        Minimal copy of anndata object.
    """

    adata_minimal = adata.copy()
    adata_minimal.X = None
    adata_minimal.layers = None
    adata_minimal.raw = None

    return adata_minimal


@beartype
def load_h5ad(path: str) -> sc.AnnData:
    """
    Load an anndata object from .h5ad file.

    Parameters
    ----------
    path : str
        Name of the file to load the anndata object. NOTE: Uses the internal 'sctoolbox.settings.adata_input_dir' + 'sctoolbox.settings.adata_input_prefix' as prefix.

    Returns
    -------
    sc.AnnData
        Loaded anndata object.
    """

    adata_input = settings.full_adata_input_prefix + path
    adata = sc.read_h5ad(filename=adata_input)

    logger.info(f"The adata object was loaded from: {adata_input}")

    if adata.raw:
        logger.warning("Found AnnData.raw! Be aware that Scanpy favors '.raw' unless explicitly told to do otherwise."
                       "Change this behavior by either setting 'AnnData.raw = None' or providing your preferred layer where neccessary.")

    return adata


@deco.log_anndata
@beartype
def save_h5ad(adata: sc.AnnData, path: str, report: Optional[str] = None) -> None:
    """
    Save an anndata object to an .h5ad file.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to save.
    path : str
        Name of the file to save the anndata object. NOTE: Uses the internal 'sctoolbox.settings.adata_output_dir' + 'sctoolbox.settings.adata_output_prefix' as prefix.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    """
    # fixes rank_genes nan in adata.uns[<rank_genes>]['names'] error
    # https://github.com/scverse/scanpy/issues/61
    rank_keys = set(['params', 'names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges'])  # the keys found with rank_genes_groups
    for unk in adata.uns.keys():
        if isinstance(adata.uns[unk], dict) and set(adata.uns[unk].keys()) == rank_keys:
            names = list()
            dnames = adata.uns[unk]["names"].dtype.names  # save dtype names
            for i in adata.uns[unk]["names"]:
                names.append([j if j == j else "" for j in i])
            tmp = pd.DataFrame(data=names).to_records(index=False)
            tmp.dtype.names = dnames  # set old names back in place
            adata.uns[unk]["names"] = tmp

    # Save adata
    adata_output = settings.full_adata_output_prefix + path
    adata.write(filename=adata_output)

    # generate report
    if settings.report_dir and report:
        with open(Path(settings.report_dir) / report, "w") as f:
            f.write("\n".join([
                "## Dataset",
                f"{adata.shape[0]} observations x {adata.shape[1]} variables",
                f"Observation information: {', '.join(adata.obs.columns)}",
                f"Variable information: {', '.join(adata.var.columns)}",
                f"Additional data layers: {', '.join(adata.layers.keys())}" if adata.layers.keys() else ""
            ]))

    logger.info(f"The adata object was saved to: {adata_output}")


@beartype
def add_uns_info(adata: sc.AnnData,
                 key: str | list[str],
                 value: Any,
                 how: str = "overwrite") -> None:
    """
    Add information to adata.uns['sctoolbox'].

    This is used for logging the parameters and options of different steps in the analysis.

    Parameters
    ----------
    adata : sc.AnnData
        An AnnData object.
    key : str | list[str]
        The key to add to adata.uns['sctoolbox']. If the key is a list, it represents a path within a nested dictionary.
    value : Any
        The value to add to adata.uns['sctoolbox'].
    how : str, default "overwrite"
        When set to "overwrite" provided key will be overwriten. If "append" will add element to existing list or dict.

    Raises
    ------
    ValueError
        If value can not be appended.
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


@beartype
def in_uns(adata: sc.AnnData,
           key: str | list[str]) -> bool:
    """
    Check whether a key or key path is in adata.uns.

    Parameters
    ----------
    adata: sc.AnnData
        The anndata object to check.
    key: str | list[str]
        The key(s) to check. A list is treated similar to a path which results in checking for nested lists. E.g.:
        a key ['a', 'b', 'c'] would return true if adata.uns = {'a': {'b': {'c': ...}}}.

    Returns
    -------
    bool
        True if key or key path is found otherwise False.
    """
    d = adata.uns
    for k in key:
        if k in d:
            d = d[k]
        else:
            return False
    return True


@beartype
def get_uns(adata: sc.AnnData,
            key: str | list[str]) -> Any:
    """
    Get value from adata.uns.

    Parameters
    ----------
    adata: sc.AnnData
        The anndata object.
    key: str | list[str]
        The key(s) of value. A list is treated similar to a path which results in checking for nested lists. E.g.:
        a key ['a', 'b', 'c'] would return true if adata.uns = {'a': {'b': {'c': ...}}}.

    Raises
    ------
    ValueError
        If key not fóund in adata.uns sub dictionary.

    Returns
    -------
    Any
        Any value stored in adata.uns
    """
    d = adata.uns
    path = ""
    for k in key:
        if k in d:
            d = d[k]
            path += f"['{k}']"
        else:
            raise ValueError(f"Key {k} not found in adata.uns{path}")
    return d


@beartype
def get_cell_values(adata: sc.AnnData,
                    element: str) -> np.ndarray:
    """Get the values of a given element in adata.obs or adata.var per cell in adata. Can for example be used to extract gene expression values.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object.
    element : str
        The element to extract from adata.obs or adata.var, e.g. a column in adata.obs or an index in adata.var.

    Returns
    -------
    np.ndarray
        Array of values per cell in adata.

    Raises
    ------
    ValueError
        If element is not found in adata.obs or adata.var.
    """

    if element in adata.obs:
        values = np.array(adata.obs[element].values)
    elif element in adata.var.index:
        idx = list(adata.var.index).index(element)
        values = adata.X[:, idx]
        values = values.todense().A1 if issparse(values) else values
    else:
        raise ValueError(f"Element '{element}' not found in adata.obs or adata.var.")

    return values


@beartype
def prepare_for_cellxgene(adata: sc.AnnData,
                          keep_obs: Optional[list[str]] = None,
                          keep_var: Optional[list[str]] = None,
                          delete_obs: Optional[list[str]] = None,
                          delete_var: Optional[list[str]] = None,
                          rename_obs: Optional[dict[str, str]] = None,
                          rename_var: Optional[dict[str, str]] = None,
                          embedding_names: Optional[list[str]] = ["pca", "umap", "tsne"],
                          cmap: Optional[str] = None,
                          palette: Optional[str | Sequence[str]] = None,
                          layer: Optional[str] = None,
                          inplace: bool = False) -> Optional[sc.AnnData]:
    """
    Prepare the given adata for cellxgene deployment.

    Parameters
    ----------
    adata : sc.Anndata
        Anndata object.
    keep_obs : Optional[list[str]], default None
        adata.obs columns that should be kept. None to keep all.
        'keep_obs' and 'delete_obs' are mutually exclusive.
    keep_var : Optional[list[str]], default None
        adata.var columns that should be kept. None to keep all.
        'keep_var' and 'delete_var' are mutually exclusive.
    delete_obs : Optional[list[str]], default None
        adata.obs columns that should be deleted. None to keep all.
        'keep_obs' and 'delete_obs' are mutually exclusive.
    delete_var : Optional[list[str]], default None
        adata.var columns that should be deleted. None to keep all.
        'keep_var' and 'delete_var' are mutually exclusive.
    rename_obs : Optional[dict[str, str]], default None
        Dictionary of .obs columns to rename. Key is the old name, value the new one.
    rename_var : Optional[dict[str, str]], default None
        Dictionary of .var columns to rename. Key is the old name, value the new one.
    embedding_names : Optional[list[str]], default ["pca", "umap", "tsne"]
        List of embeddings to check for. Will raise an error if none of the embeddings are found. Set None to disable check. Embeddings are stored in `adata.obsm`.
    cmap : Optional[str], default None
        Color map to use for continous variables.
        Use this replacement color map for broken color maps.
        If None will use scanpy default, which uses `mpl.rcParams["image.cmap"]`. See `sc.pl.embedding`.
    palette : Optional[str | Sequence[str]], default None
        Color map to use for categorical annotation groups.
        Use this replacement color map for broken color maps.
        If None will use scanpy default, which uses `mpl.rcParams["axes.prop_cycle"]`. See `sc.pl.embedding`.
    layer : Optional[str], default None

    inplace : bool, default False

    Raises
    ------
    ValueError
        1. If mutally exclusive parameters keep_obs/keep_var abd delete_obs/delete_var are both set.
        2. If not at least one of the named embeddings are found in the adata.
        3. If there is no layer with the given name.

    Returns
    -------
    Optional[sc.AnnData]
        Returns the deployment ready Anndata object.
    """
    if layer and layer not in adata.layers:
        raise ValueError(f"No layer named '{layer}' found in the AnnData. Available layers are {','.join(adata.layers.keys())}.")

    def clean_section(obj, axis="obs", keep=None, delete=None, rename=None) -> None:
        """Clean either obs or var section of given adata object."""
        if keep is not None and delete is not None:
            raise ValueError(f"'keep_{axis}' and 'delete_{axis}' are mutually exclusive. Please configure only one to proceed.")

        if axis == "obs":
            sec_table = obj.obs
        elif axis == "var":
            sec_table = obj.var

        # drop columns
        if keep is not None:
            drop = set(sec_table.columns) - set(keep)
        elif delete is not None:
            drop = set(delete)
        else:
            drop = False

        if drop is not False:
            sec_table.drop(columns=drop, inplace=True)

            # drop matching color maps
            for col in drop:
                if f"{col}_colors" in obj.uns.keys():
                    obj.uns.pop(f"{col}_colors")

        # rename columns
        if rename:
            sec_table.rename(columns=rename, inplace=True)

            # rename color maps
            for old, new in rename.items():
                if f"{old}_colors" in obj.uns.keys():
                    obj.uns[f"{new}_colors"] = obj.uns.pop(f"{old}_colors")

        # convert Int32 to float64 columns
        for c in sec_table:
            if sec_table[c].dtype == 'Int32':
                sec_table[c] = sec_table[c].astype('float64')

        sec_table.index.names = ['index']

    out = adata if inplace else adata.copy()

    # TODO remove more adata internals not needed for cellxgene

    # ----- .obsm -----
    if embedding_names:
        if not any(f"X_{e}" == k for e in embedding_names for k in out.obsm.keys()):
            raise ValueError(f"Unable to find any of the embeddings {embedding_names}. At least one is needed for cellxgene.")

    # ----- .obs -----
    clean_section(out, axis="obs", keep=keep_obs, delete=delete_obs, rename=rename_obs)
    out.obs_names_make_unique()

    # ----- .var -----
    clean_section(out, axis="var", keep=keep_var, delete=delete_var, rename=rename_var)
    out.var_names_make_unique()

    # ----- .X -----
    # overwrite .X with another layer
    if layer:
        out.X = out.layers[layer].copy()

    # convert .X to sparse matrix if needed
    if not scipy.sparse.isspmatrix(out.X):
        out.X = scipy.sparse.csc_matrix(out.X)

    out.X = out.X.astype("float32")

    # ----- .uns -----
    for key in list(out.uns):  # avoid RuntimeError by forcing a copy of dict keys.
        if key.endswith('colors'):
            obs_key = key.split("_colors")[0]
            # delete colors if they don't match a .obs column.
            if obs_key not in out.obs.columns:
                out.uns.pop(key)

                logger.warning(f"Deleted .uns[{key}] since it did not match a .obs column.")
                continue

            # fix colors not in 6-digit hex format
            # https://github.com/chanzuckerberg/cellxgene/issues/2598
            out.uns[key] = np.array([(c if len(c) <= 7 else c[:-2]) for c in out.uns[key]])

            # fix number of colors < number of categories
            if len(out.uns[key]) != len(set(out.obs[obs_key])):
                logger.warning(f"Coloring for adata.obs['{obs_key}'] broken. Reverting to {cmap if cmap else 'scanpy default'} color map.")

                # scanpy replaces broken colormap before plotting
                basis = list(out.obsm.keys())[0]
                sc.pl.embedding(adata=out, basis=basis, color=obs_key, palette=palette, color_map=cmap, show=False)
                plt.close()  # prevent that plot is shown

    if not inplace:
        return out


@beartype
def concadata(adatas: Union[Collection[sc.AnnData], Mapping[str, sc.AnnData]], label: Optional[str] = "batch") -> sc.AnnData:
    """
    Concatenate several anndata objects by appending cells.

    Essentially `sc.concat(adatas, join="outer", axis=0)` but retains adata.var information.

    Parameters
    ----------
    adatas: Union[Collection[sc.AnnData], Mapping[str, sc.AnnData]]
        A combination of AnnData objects to concatenate. Forwarded to the `adatas` parameter of [scanpy.concat](https://anndata.readthedocs.io/en/stable/generated/anndata.concat.html#anndata.concat).
    label: Optional[str], default "batch"
        Name of the `adata.obs` column to place the batch information in. Forwarded to the `label` parameter of [scanpy.concat](https://anndata.readthedocs.io/en/stable/generated/anndata.concat.html#anndata.concat)

    Returns
    -------
    sc.AnnData
        Returns the combined AnnData object.
    """
    # create adata
    adata = sc.concat(adatas, join="outer", axis=0, label=label)

    # manually combine var table, then add it to the adata
    var = pd.concat(
        [a.var for a in (adatas.values() if isinstance(adatas, Mapping) else adatas)],
        join="outer"
    )

    # remove duplicates
    # temporarily set index as column to use this as column for duplicate removal
    ind_name = var.index.name
    tmp_name = "_".join(var.columns) + "_" if len(var.columns) else "index"  # create a name that is not present in the var columns
    var = var.reset_index(names=tmp_name).drop_duplicates(subset=tmp_name).set_index(tmp_name)
    var.index.name = ind_name  # revert to the original index name

    # add the var table to the adata while ensuring the correct order
    adata.var = var.loc[adata.var_names]

    return adata
