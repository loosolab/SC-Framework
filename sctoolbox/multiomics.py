import copy
import anndata as ad
import pandas as pd
from functools import reduce


def merge_anndata(anndata_dict):
    """
    Merge two h5ad files for dual cellxgene deplyoment

    Parameters
    ----------
    anndata_dict : dict
        dictionary with labels as keys and anndata objects as values

    Returns
    -------
    merged anndata.AnnData object
    """
    # ToDo Add outer merge for obs

    # Copy dict to prevent changes in original anndata objects
    anndata_dict = copy.deepcopy(anndata_dict)

    obs_list = list()
    # Add prefix to obsm dict keys, var index and obs columns
    for label, adata in anndata_dict.items():
        # prefix to obsm
        adata.obsm = {f"X_{label}_{key.removeprefix('X_')}": val for key, val in adata.obsm.items()}
        # prefix to var index
        adata.var.index = [f"{label}_{i}" for i in adata.var.index]
        # prefix to obs columns
        adata.obs.columns = label + "_" + adata.obs.columns
        # save obs in list
        obs_list.append(adata.obs)

    # Merge X and var
    merged_X_var = ad.concat(anndata_dict, join="inner", label="source", axis=1)

    # Merge obs
    merged_X_var.obs = reduce(lambda left, right: pd.merge(left, right,
                                                           how='inner',
                                                           left_index=True,
                                                           right_index=True), obs_list)
    # Build new obsm
    obs_len = merged_X_var.shape[0]
    for adata in anndata_dict.values():
        for obsm_key, value in dict(adata.obsm).items():
            merged_X_var.obsm[obsm_key] = value[0:obs_len]

    return merged_X_var