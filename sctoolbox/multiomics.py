import copy
import anndata as ad
import pandas as pd
from functools import reduce
import numpy as np
import warnings

import sctoolbox.utilities as utils


def merge_anndata(anndata_dict, join="inner"):
    """
    Merge two h5ad files for dual cellxgene deplyoment

    Parameters
    ----------
    anndata_dict : dict
        dictionary with labels as keys and anndata objects as values
    join : string, deafult 'inner'
        set how to join cells of the adata objects: ['inner', 'outer']
        This only affects the cells since the var/gene section is simply added
        'inner': only keep overlapping cells
        'outer': keep all cells. This will add placeholder cells/dots to plots

    Returns
    -------
    merged anndata.AnnData object
    """

    if join not in ["inner", "outer"]:
        raise ValueError(f"Invalid join value: {join}. Set to 'inner' or 'outer'")

    # Copy dict to prevent changes in original anndata objects
    anndata_dict = copy.deepcopy(anndata_dict)

    obs_list = list()
    # Add prefix to obsm dict keys, var index and obs columns
    for label, adata in anndata_dict.items():
        # prefix to obsm
        adata.obsm = {f"X_{label}_{key.removeprefix('X_')}": val for key, val in adata.obsm.items()}
        # prefix to var index
        adata.var.index = label + "_" + adata.var.index
        # prefix to obs columns
        adata.obs.columns = label + "_" + adata.obs.columns
        # save obs in list
        obs_list.append(adata.obs)

    # Merge X and var
    merged_X_var = ad.concat(anndata_dict, join=join, label="source", axis=1)

    # Merge obs
    merged_X_var.obs = reduce(lambda left, right: pd.merge(left, right,
                                                           how=join,
                                                           left_index=True,
                                                           right_index=True), obs_list)
    utils.fill_na(merged_X_var.obs)
    utils.fill_na(merged_X_var.var)

    # Build new obsm
    obs_len = merged_X_var.shape[0]
    for adata in anndata_dict.values():
        for obsm_key, value in dict(adata.obsm).items():
            if len(value) < obs_len:
                dummy_coordinates = [value[0] * [0] for _ in range(abs(obs_len - len(value)))]
                merged_X_var.obsm[obsm_key] = np.append(value[0:obs_len], np.asarray(dummy_coordinates), axis=0)
            else:
                merged_X_var.obsm[obsm_key] = value[0:obs_len]  

    if len(merged_X_var.var) <= 50:
        warnings.warn("The adata object contains less than 51 genes/var entries. " +
                      "CellxGene will not work. Please add dummy genes to the var table.")

    return merged_X_var
