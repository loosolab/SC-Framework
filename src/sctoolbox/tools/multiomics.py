"""Tools for multiomics analysis."""
import scanpy as sc
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from functools import reduce
import warnings
import muon as mu
from scipy.stats import zscore

from beartype import beartype
from beartype.typing import Literal, Optional, Tuple, List, Iterable
from numpy.typing import NDArray

import sctoolbox.utils as utils
from sctoolbox.utils.general import remove_suffix


@beartype
def merge_anndata(anndata_dict: dict[str, sc.AnnData],
                  join: Literal["inner", "outer"] = "inner") -> sc.AnnData:
    """
    Merge two h5ad files for dual cellxgene deployment.

    Parameters
    ----------
    anndata_dict : dict[str, sc.AnnData]
        Dictionary with labels as keys and anndata objects as values.
    join : Literal['inner', 'outer'], default 'inner'
        Set how to join cells of the adata objects: ['inner', 'outer'].
        This only affects the cells since the var/gene section is simply added.

        - 'inner': only keep overlapping cells.
        - 'outer': keep all cells. This will add placeholder cells/dots to plots currently disabled.

    Returns
    -------
    sc.AnnData
        Merged anndata object.

    Raises
    ------
    ValueError
        If no indices of both adata.obs tables are overlapping.

    Notes
    -----
    Important: Depending on the size of the anndata objects the function takes
    around 60 to 300 GB of RAM!
    To save RAM and runtime the function generates a minimal anndata object.
    Only .X, .var, .obs and .obsm are kept. Layers, .varm, etc is removed.
    """

    if join == "outer":
        warnings.warn("'outer' join is currently not supported. Proceeding with 'inner' ...")
        join = "inner"

    # Generate minimal anndata objects
    minimal_adata_dict = dict()
    for label, adata in anndata_dict.items():
        minimal_adata_dict[label] = sc.AnnData(X=adata.X, obs=adata.obs, var=adata.var, obsm=dict(adata.obsm))
        if not adata.obs.index.is_unique:
            warnings.warn(f"Obs index of {label} dataset is not unique. Running .obs_names_make_unique()..")
            minimal_adata_dict[label].obs_names_make_unique()

        if not adata.var.index.is_unique:
            warnings.warn(f"Var index of {label} dataset is not unique. Running .var_names_make_unique()..")
            minimal_adata_dict[label].var_names_make_unique()

    # Get cell barcode (obs) intersection
    all_obs_indices = [single_adata.obs.index for single_adata in list(anndata_dict.values())]
    obs_intersection = set.intersection(*map(set, all_obs_indices))
    obs_intersection = sorted(list(obs_intersection))

    if not obs_intersection:
        raise ValueError("No overlapping indices among the .obs tables. "
                         + "The barcodes of the cells must be identical for all datasets")

    obs_list = list()
    obsm_dict = dict()

    # Add prefix to obsm dict keys, var index and obs columns
    for label, adata in minimal_adata_dict.items():
        # prefix to var index
        adata.var.index = label + "_" + adata.var.index
        adata.var.columns = label + "_" + adata.var.columns
        # prefix to obs columns
        adata.obs.columns = label + "_" + adata.obs.columns
        # Reorder adata cells
        adata = adata[obs_intersection, :]
        for obsm_key, matrix in dict(adata.obsm).items():
            obsm_dict[f"X_{label}_{obsm_key.removeprefix('X_')}"] = matrix
        # save new adata to dict
        minimal_adata_dict[label] = adata
        # save obs in list
        obs_list.append(adata.obs)
    # Merge X and var
    merged_adata = sc.concat(minimal_adata_dict, join="outer", label="source", axis=1)

    # Merge obs
    merged_adata.obs = reduce(lambda left, right: pd.merge(left, right,
                                                           how=join,
                                                           left_index=True,
                                                           right_index=True), obs_list)
    merged_adata.obsm = obsm_dict

    utils.tables.fill_na(merged_adata.obs)
    utils.tables.fill_na(merged_adata.var)

    if len(merged_adata.var) <= 50:
        warnings.warn("The adata object contains less than 51 genes/var entries. "
                      + "CellxGene will not work. Please add dummy genes to the var table.")

    return merged_adata


@beartype
def match_barcodes(adata_mod1: sc.AnnData,
                   adata_mod2: sc.AnnData,
                   barcode_map: str,
                   sample_delimiter: str = "-",
                   sample_mod1: str = "Sample",
                   sample_mod2: str = "Sample",
                   inplace: bool = True) -> Optional[Tuple[sc.AnnData, sc.AnnData]]:
    """
    Match cellbarcodes for multiomics anndatas.

    Parameters
    ----------
    adata_mod1 : sc.AnnData
        Anndata of modality 1.
    adata_mod2 : sc.AnnData
        Anndata of modality 2.
    barcode_map : str
        Path to barcode map file.
        The file contains a tab seperated table of two columns.
        The first column contains the barcodes of modality 1.
        The second column the coresponding barcode of modality 2.
        Example:
        TAGCCCTGATATTGGG        AGGTCTTAGTGAACCT
        GGATCCAGATATTGGG        AGGTCTTAGTGAACGA
    sample_delimiter : str, default '-'
        Often the sample name is added to the barcode to generate unique IDs.
        For mapping the raw barcodes are required wihtout the sample name.
    sample_mod1 : String, default="Sample"
        Sample column in the modality 1 anndata object.
    sample_mod2 : String, default="Sample"
        Sample column in the modality 2 anndata object.
    inplace : bool, default True
        If true, matches barcodes inplace.

    Returns
    -------
    Optional[Tuple[sc.Anndata, sc.Anndata]]
        If not inplace, returns a tuple of both matched anndatas.
    """
    if not inplace:
        adata_mod1 = adata_mod1.copy()
        adata_mod2 = adata_mod2.copy()

    # Check if cell tags already match between modalities else convert
    raw_barcodes_mod1 = [i.split(sample_delimiter)[0] if sample_delimiter in i else i for i in adata_mod1.obs.index]
    raw_barcodes_mod2 = [i.split(sample_delimiter)[0] if sample_delimiter in i else i for i in adata_mod2.obs.index]

    if raw_barcodes_mod1 != raw_barcodes_mod2:

        # Get table to convert ATAC to RNA barcodes
        bc_conversion = pd.read_csv(barcode_map, sep='\t', header=None, names=['mod1', 'mod2'])
        bc_conversion["barcode_id"] = bc_conversion.index.astype(str)

        # Add unique barcode IDs as indeces to the two andata objects
        adata_mod1.obs["raw_barcode"] = raw_barcodes_mod1
        adata_mod1.obs = adata_mod1.obs.merge(bc_conversion, left_on="raw_barcode", right_on="mod1", how="left")
        # should be batch not sample column and only used if multiple batches?
        adata_mod1.obs.index = adata_mod1.obs["barcode_id"] + "-" + adata_mod1.obs[sample_mod1].astype(str)

        adata_mod2.obs["raw_barcode"] = raw_barcodes_mod2
        adata_mod2.obs = adata_mod2.obs.merge(bc_conversion, left_on="raw_barcode", right_on="mod2", how="left")
        # should be batch not sample column and only used if multiple batches?
        adata_mod2.obs.index = adata_mod2.obs["barcode_id"] + "-" + adata_mod2.obs[sample_mod2].astype(str)

    else:
        # Alter barcodes to match <cell tag>-<Sample> in both modalities
        adata_mod1.obs.index = raw_barcodes_mod1 + "-" + adata_mod1.obs[sample_mod1].astype(str)
        adata_mod2.obs.index = raw_barcodes_mod2 + "-" + adata_mod2.obs[sample_mod2].astype(str)

    # Remove possible NaN indeces from anndatas after cell tag conversion
    adata_mod1 = adata_mod1[adata_mod1.obs.index.notnull()].copy()
    adata_mod2 = adata_mod2[adata_mod2.obs.index.notnull()].copy()

    if not inplace:
        return adata_mod1, adata_mod2


@beartype
def add_multiome_prefix(
        adata: sc.AnnData,
        prefix: str,
        ignore_obs_col: list[str] = [],
        inplace: bool = True
        ) -> Optional[sc.AnnData]:
    """
    Add prefix to obsm, obs column names and var.index.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object to add multiome prefix to.
    prefix : str
        Prefix to be added, e.g. 'RNA'
    ignore_obs_col : list[str], default []
        List of obs column names to not add the prefix to.
    inplace : bool, default True
        If True, modifies adata.obsm, adata.obs.colnames, adata.var.index in place. Otherwise, returns a copy of adata.

    Returns
    -------
    Optional[sc.Anndata]
        If inplace = True, returns None.
        Otherwise, returns a copy of adata.
    """
    if not inplace:
        adata = adata.copy()

    # Add prefix to obsm keys, between "X_" and the rest
    # e.g. "X_pca" -> "X_RNA_pca"
    if adata.obsm:
        new_obsm = {}
        for key, value in adata.obsm.items():
            new_key = key[2:] if key.startswith("X_") else key
            new_obsm[f"X_{prefix}_{new_key}"] = value
        adata.obsm = new_obsm

    # Add prefix to obs column names, except those in ignore_obs_col
    if adata.obs.columns is not None:
        new_obs_cols = []
        for col in adata.obs.columns:
            if col in ignore_obs_col:
                new_obs_cols.append(col)
            else:
                new_obs_cols.append(f"{prefix}_{col}")
        adata.obs.columns = new_obs_cols

    # Add prefix to var.index
    if adata.var.index is not None:
        adata.var.index = [f"{prefix}_{idx}" for idx in adata.var.index]

    if not inplace:
        return adata


@beartype
def join_modalities(adata_mod1: sc.AnnData,
                    adata_mod2: sc.AnnData,
                    modality_1: str,
                    modality_2: str,
                    keep_outer: bool = False) -> mu.MuData:
    """
    Take two anndata objects, one for each of two modalities and join them to one new mudata object.

    Parameters
    ----------
    adata_mod1 : anndata Object
        Object containing all information related to modality 1.
    adata_mod2 : anndata Object
        Object containing all information related to modality 2.
    modality_1 : str
        Value with the name of the first modality that will be used as a key for the anndata object within the
        joint mudata object.
    modality_2 : str
        Value with the name of the second modality that will be used as a key for the anndata object within
        the joint mudata object.
    keep_outer : bool, default False
        If True, keep cells that are present in only one modality.

    Returns
    -------
    mudata :
        Muon mudata object containnig both modalities.
    """
    # Create mudata object from anndata objects for modality 1 and modality 2
    mdata = mu.MuData({modality_1: adata_mod1, modality_2: adata_mod2})

    # Check if cells that are not present in both modalities are to be kept or filtered out
    if not keep_outer:
        # Filter to keep only cells that exist in both modalities
        mu.pp.intersect_obs(mdata)

    return mdata


@beartype
def mean_percent_data_frame(data_frames: Iterable[pd.DataFrame],
                            modalities: Iterable[str]) -> Styler:
    """
    Calculate mean percentage overlap between the modality clusters.

    Takes two data frames with modality cluster overlap percentages and generates a new dataframe with the mean
    percentage overlap for each of the modality clusters from both sides. The best match is marked
    for each cluster of each modality.

    Parameters
    ----------
    data_frames : List[pd.DataFrame]
        List of data frames containing information on cluster overlap between modalities from both sides.
    modalities : List[str]
        List of values with the names of the modalities in the dataset.

    Returns
    -------
    pd.io.formats.style.Styler
        A pandas data frame styler as described above.
    """
    # Generate copies of the data frames so as not to alter the originals
    dfs = np.empty(len(data_frames), dtype=pd.DataFrame)

    for i in range(0, len(data_frames)):
        dfs[i] = data_frames[i].copy()

    # Prepare data frames for merging by setting the correct index and renaming columns
    for i in range(0, len(data_frames)):
        dfs[i] = dfs[i].reset_index().set_index([f"{modalities[0]}_clusters", f"{modalities[1]}_clusters"])
        dfs[i] = dfs[i].rename(columns={"Cells_per_cluster_pct": f"Cells_per_cluster_pct_{modalities[i]}"})

    # Join data frames
    df_joined = (dfs[0].join(dfs[1][f"Cells_per_cluster_pct_{modalities[1]}"],
                             on=[f"{modalities[0]}_clusters", f"{modalities[1]}_clusters"]))

    # Calculate mean cluster overlap percentage
    df_joined["Mean_clusters_pct"] = (df_joined[[f"Cells_per_cluster_pct_{modalities[0]}",
                                                 f"Cells_per_cluster_pct_{modalities[1]}"]].
                                      mean(axis=1))

    # Set correct index
    df_joined = df_joined.reset_index().set_index(f"{modalities[0]}_clusters")

    # Pivot table for desired format
    df_joined = df_joined.pivot(columns=df_joined.columns[0], values="Mean_clusters_pct")

    # Get rid of clusters/conditions in data frame with no cells in joint data set:
    # remove rows with all zeroes
    mean_df = df_joined.loc[~(df_joined == 0).all(axis=1)]

    # remove columns with all zeroes
    for col in mean_df.columns:
        if (mean_df[col] == 0.0).all():
            mean_df.drop(labels=col, axis=1)

    # copy mean data frame
    z_score_df = mean_df.copy()

    # iterate across each cell in data frames and replace values in copy with z-scores
    for i in range(0, len(mean_df.index)):
        for j in range(0, len(mean_df.columns)):
            mean_df.iloc[[i], [j]]

            # Get row and column around current position in data frame as lists
            row = [val for vals in mean_df.iloc[[i]].values.tolist() for val in vals]
            column = [val for vals in mean_df.iloc[:, [j]].values.tolist() for val in vals]

            # Remove current value for current position from row to not have it double
            del column[i]

            # Concat row and column lists
            scores = row + column

            # Compute z score for row + column
            z_scores = zscore(scores)

            # Add z score for cell to z score data frame
            z_score_df.iat[i, j] = z_scores[j]

    # Mark highest value per row and highest value per column
    z_score_df = (z_score_df.style.
                 apply(lambda x: ["border-style: solid" if v == x.max() and x.max() > 0.0 else "" for v in x], axis=1).
                 apply(lambda x: ["border-style: solid" if v == x.max() and x.max() > 0.0 else "" for v in x], axis=0))

    return z_score_df


@beartype
def cluster_comparison_data_frames(data_frame: pd.DataFrame,
                                   modalities: List[str],
                                   clustercols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate three comparison three matrices.

    One that can be used to generate the heatmap for visualization, one that can be used
    to generate the sankey diagram and one for display of cluster comparison between modalities.
    The second matrix shows per row:
        - Cluster name from modality one as index.
        - Number of cells total assigned to modality 1 cluster.
        - A dictionary showing how many of the cells in modality 1 cluster have been assigend to each modality 2 cluster.
        - A dictionary showing the percentage of cells in modality 1 cluster assigned to each modality 2 cluster.
        - The name of the best match modality 2 cluster for the modality 1 cluster.
        - The percentage of cells from the modality 1 cluster assigned to the best match modality 2 cluster.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Data frame containing information on both modalities with clustering columns.
    modalities : List[str]
        List of values with the names of the modalities in the dataset.
    clustercols : List[str]
        List of value with the names of the modality clustering columns in the modality matrices or in the joint matrix.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        df_final, df_heatmap, df_sankey
        Three data frames as described above.
    """
    # New name for index and other columns
    index = "_".join([modalities[0], "clusters"])
    clusters_mod2 = "_".join([modalities[1], "clusters_cnts"])
    clusters_pct_mod2 = "_".join([modalities[1], "clusters_pct"])

    # Caclulate number of cells per modality 1 cluster
    df_tmp = data_frame.groupby(clustercols[0]).size().reset_index(name="Cells_total").set_index(clustercols[0])
    # Calculate number of cells per modality 1 cluster that has been assigned to each modality 2 cluster
    df_tmp = (data_frame.groupby([clustercols[0], clustercols[1]]).size().
                 reset_index(name="Cells_per_cluster").set_index(clustercols[0]).join(df_tmp))

    # Calculate fractions of cells from each modality 1 cluster assigned to each modality 2 cluster
    df_heatmap = df_tmp.assign(Cells_per_cluster_pct=lambda x: round(x.Cells_per_cluster / x.Cells_total, 2))
    # Rename index column
    df_heatmap.index.name = index

    # Reset index and assign sankey data frame
    df_sankey = df_tmp.reset_index()

    # Group by cluster names of modality 1 clusters
    df_final = df_heatmap.groupby(index).agg(list)
    # Generate column with modality 2 cluster names as keys and number of cells per modality 2 cluster as values in dictionary
    df_final.insert(3, clusters_mod2,
                    df_final.apply(lambda x: dict(zip(x[clustercols[1]], x["Cells_per_cluster"])), axis=1))
    # Generate column with modality 2 cluster names as keys and percentage of cells per modality 2 cluster as values in dictionary
    df_final.insert(4, clusters_pct_mod2,
                    df_final.apply(lambda x: dict(zip(x[clustercols[1]], x["Cells_per_cluster_pct"])), axis=1))
    # Drop columns no longer needed
    df_final = df_final.drop(["Cells_per_cluster", "Cells_per_cluster_pct", clustercols[1]], axis=1)
    # Add column with best match modality 2 cluster per modality 1 cluster
    df_final["Best_match"] = df_final[clusters_pct_mod2].apply(lambda x: max(x, key=x.get) if x[max(x, key=x.get)] > 0.0
                                                               else None)
    # Add column with best match percentage per modality 1 cluster
    df_final["Best_match_pct"] = df_final[clusters_pct_mod2].apply(lambda x: max(x.values()))
    # Unpack one-element-lists in column "Cells_total"
    df_final["Cells_total"] = df_final.agg(lambda x: x["Cells_total"][0], axis=1)

    # Rename column with modality 2 clusters in heatmap data frame
    df_heatmap.rename(columns={clustercols[1]: clusters_mod2},
                      inplace=True)
    # Fix column names in heatmap data frame
    df_heatmap.rename(columns={df_heatmap.columns[0]: remove_suffix(df_heatmap.columns[0], "_cnts")},
                      inplace=True)

    return df_heatmap, df_final, df_sankey


def compare_clusters(mdata: mu.MuData,
                     clusters_mod1: str,
                     clusters_mod2: str,) -> Tuple[NDArray[pd.DataFrame], NDArray[pd.DataFrame], NDArray[pd.DataFrame]]:
    """
    Calculate comparison matricies of clusters between modalities.

    - Give a score for each pairwise cluster comparison that rates how well they match -> Mean of percentage of cells of one cluster
      in other cluster from both sides.
    - Generate a matrix that lists the scores between the modalities.
    - Give out best match for each cluster.

    Parameters
    ----------
    mdata : mu.MuData
        Muon object containing both modalities with clustering information.
    clusters_mod1 : str
        Clustering column of modality 1
    clusters_mod2 : str
        Clustering column of modality 2

    Returns
    -------
    Tuple[NDArray[pd.DataFrame], NDArray[pd.DataFrame], NDArray[pd.DataFrame]]
        Tuple of comparison matricies
    """
    modality_1, modality_2 = list(mdata.mod.keys())

    # Assure that values for parameters clustercol_mod1 and clustercol_mod2 have the correct prefixes for mdata.obs
    clusters_mod1 = ":".join([modality_1, clusters_mod1]) if not clusters_mod1.startswith(modality_1) else clusters_mod1
    clusters_mod2 = ":".join([modality_2, clusters_mod2]) if not clusters_mod2.startswith(modality_2) else clusters_mod2

    # Initialize lists for modality data frames, heatmap data frames and sankey data frames
    dfs_mods = np.empty(4, dtype=pd.DataFrame)
    dfs_heatmaps = np.empty(2, dtype=pd.DataFrame)
    dfs_sankey = np.empty(2, dtype=pd.DataFrame)

    # Generate data frames for modality 1 and 2
    for i in range(0, 2):
        dfs_heatmaps[i], dfs_mods[i], dfs_sankey[i] = cluster_comparison_data_frames(mdata.obs,
                                                                                     [modality_1, modality_2] if i == 0
                                                                                     else [modality_2, modality_1],
                                                                                     [clusters_mod1, clusters_mod2] if i == 0
                                                                                     else [clusters_mod2, clusters_mod1])
    # Generate mean percent data frame
    dfs_mods[2] = mean_percent_data_frame(dfs_heatmaps, [modality_1, modality_2])

    # Save mean percent data frame to mdata.uns layer
    mdata.uns["cluster_comparison_best_matches"] = dfs_mods[2].data

    # Add data frame for sankey diagramm
    dfs_mods[3] = dfs_sankey[0]

    return dfs_heatmaps, dfs_mods, dfs_sankey
