import copy
import anndata as ad
import pandas as pd
from functools import reduce
import warnings

import sctoolbox.utilities as utils


def merge_anndata(anndata_dict, join="inner"):
    """
    Merge multiple anndata objects to enable an easy comparsion via cellxgene.
    This function is designed for multiomics data. All input anndata objects need to have
    identical cell barcodes stored in the obs index or at least have some barcodes overlapping.
    The genes and column names are labeled based on their source anndata object.
    This allows for an easy comparison between the different -omics datasets.

    ######################################################################################
    WARNIG: After merging the anndata object can only be used for cellxgene deplyoments!!!
            The merged anndata object cannot be used for further analysis!!!
    ######################################################################################

    Parameters
    ----------
    anndata_dict : dict
        dictionary with labels as keys and anndata objects as values
    join : string, default 'inner'
        set how to join cells of the adata objects: ['inner', 'outer']
        This only affects the cells since the var/gene section is simply added
        'inner': only keep overlapping cells
        'outer': keep all cells. This will add placeholder cells/dots to plots
                 NOT IMPLEMENTED

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

    # Keep until not fixed
    # https://github.com/chanzuckerberg/cellxgene/issues/2597
    if len(merged_X_var.var) <= 50:
        warnings.warn("The adata object contains less than 51 genes/var entries. "
                      + "CellxGene will not work. Please add dummy genes to the var table.")

    return merged_X_var
