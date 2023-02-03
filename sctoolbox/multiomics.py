import copy
import anndata as ad
import pandas as pd
from functools import reduce
import warnings
import scipy.sparse

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
                 currently disabled

    Returns
    -------
    merged anndata.AnnData object
    """
    if join == "outer":
        warnings.warn("'outer' join is currently disabled. SEt to inner merge")
        join = "inner"

    if join not in ["inner", "outer"]:
        raise ValueError(f"Invalid join value: {join}. Set to 'inner' or 'outer'")

    # Copy dict to prevent changes in original anndata objects
    anndata_dict = copy.deepcopy(anndata_dict)

    f_adata = list(anndata_dict.values())[0]

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

        # With this line outer join not possible anymore
        # Revise to enable outer join
        adata = adata[adata.obs.index.isin(f_adata.obs.index)]

        # Reorder obs
        adata_obs_order = [list(adata.obs.index).index(i) for i in f_adata.obs.index]
        adata.obs = adata.obs.reindex(list(f_adata.obs.index))

        # Reorder obsm
        new_order_obsm = dict(adata.obsm)
        for obsm_key, value in new_order_obsm.items():
            new_order_obsm[obsm_key] = value[adata_obs_order]
        obsm_dict |= new_order_obsm

        # Reorder X
        # ToDo work on sparse directly instead of converting
        X_df = pd.DataFrame.sparse.from_spmatrix(adata.X)
        X_df_re = X_df.reindex(adata_obs_order)
        X_df_re.index = list(range(len(X_df_re)))
        adata.X = scipy.sparse.csr_matrix(X_df_re.values)

        anndata_dict[label] = adata

        # save obs in list
        obs_list.append(adata.obs)

    # Merge X and var
    merged_X_var = ad.concat(anndata_dict, join=join, label="source", axis=1)

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
