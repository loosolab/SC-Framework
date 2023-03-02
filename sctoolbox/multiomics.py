import anndata as ad
import pandas as pd
from functools import reduce
import warnings

import sctoolbox.utilities as utils


def merge_anndata(anndata_dict, join="inner"):
    """
    Merge two h5ad files for dual cellxgene deplyoment.
    Important: Depending on the size of the anndata objects the function takes
    around 60 to 300 GB of RAM!

    To save RAM and runtime the function generates a minimal anndata object.
    Only .X, .var, .obs and .obsm are kept. Layers, .varm, etc is removed.

    Parameters
    ----------
    anndata_dict : dict
        dictionary with labels as keys and anndata objects as values
    join : string, deafult 'inner'
        set how to join cells of the adata objects: ['inner', 'outer']
        This only affects the cells since the var/gene section is simply added
        'inner': only keep overlapping cells
        'outer': keep all cells. This will add placeholder cells/dots to plots
                 currently disabled

    Returns
    -------
    merged anndata.AnnData object
    """
    if join == "outer":
        warnings.warn("'outer' join is currently disabled. Set to inner merge")
        join = "inner"

    if join not in ["inner", "outer"]:
        raise ValueError(f"Invalid join value: {join}. Set to 'inner' or 'outer'")

    # Generate minimal anndata objects
    for label, adata in anndata_dict.items():

        if not adata.obs.index.is_unique:
            warnings.warn(f"Obs index of {label} dataset is not unqiue. Running .obs_names_make_unique()..")
            adata.obs_names_make_unique()

        if not adata.var.index.is_unique:
            warnings.warn(f"Var index of {label} dataset is not unqiue. Running .var_names_make_unique()..")
            adata.var_names_make_unique()

        anndata_dict[label] = ad.AnnData(X=adata.X, obs=adata.obs, var=adata.var, obsm=adata.obsm)

    # Get cell barcode (obs) intersection
    all_obs_indices = [single_adata.obs.index for single_adata in list(anndata_dict.values())]
    obs_intersection = set.intersection(*map(set, all_obs_indices))
    obs_intersection = sorted(list(obs_intersection))

    obs_list = list()
    obsm_dict = dict()

    # Add prefix to obsm dict keys, var index and obs columns
    for label, adata in anndata_dict.items():

        # prefix to obsm
        adata.obsm = {f"X_{label}_{key.removeprefix('X_')}": val for key, val in adata.obsm.items()}
        # prefix to var index
        adata.var.index = label + "_" + adata.var.index
        adata.var.columns = label + "_" + adata.var.columns
        # prefix to obs columns
        adata.obs.columns = label + "_" + adata.obs.columns

        # Reorder adata cells
        adata = adata[obs_intersection, :]

        for obsm_key, matrix in adata.obsm.items():
            obsm_dict[obsm_key] = matrix

        # save new adata to dict
        anndata_dict[label] = adata
        # save obs in list
        obs_list.append(adata.obs)

    # Merge X and var
    merged_X_var = ad.concat(anndata_dict, join="outer", label="source", axis=1)

    # Merge obs
    merged_X_var.obs = reduce(lambda left, right: pd.merge(left, right,
                                                           how=join,
                                                           left_index=True,
                                                           right_index=True), obs_list)
    merged_X_var.obsm = obsm_dict

    utils.fill_na(merged_X_var.obs)
    utils.fill_na(merged_X_var.var)

    if len(merged_X_var.var) <= 50:
        warnings.warn("The adata object contains less than 51 genes/var entries. "
                      + "CellxGene will not work. Please add dummy genes to the var table.")

    return merged_X_var
