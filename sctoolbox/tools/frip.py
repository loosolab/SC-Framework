"""Module containing functions for calculating the FRiP score."""
from tqdm import tqdm
import os
import pandas as pd
import scanpy as sc

import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings

from beartype import beartype

from typing import Tuple
logger = settings.logger


@beartype
def _count_fragments(df: pd.DataFrame,
                     barcodes_col: str = 'barcode',
                     n_fragments_col: str = 'n') -> pd.DataFrame:
    """
    Count the number of fragments per barcode.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing the columns barcode and n
    barcodes_col : str, default "barcode"
        name of the column containing the barcodes
    n_fragments_col : str, default "n"
        name of the column containing the number of fragments

    Returns
    -------
    counts : pd.DataFrame
        dataframe containing the columns barcode and n
    """

    logger.info('counting fragments per barcode')
    counts = df.groupby(barcodes_col)[n_fragments_col].sum().reset_index()

    return counts


@deco.log_anndata
@beartype
def calc_frip_scores(adata: sc.AnnData,
                     fragments: str,
                     temp_dir: str = '') -> Tuple[sc.AnnData, float]:
    """
    Calculate the FRiP score for each barcode and adds it to adata.obs.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing the fragments
    fragments : str
        path to fragments bedfile
    temp_dir : str, default ""
        path to temp directory

    Returns
    -------
    Tuple[sc.AnnData, float]
        AnnData object containing the fragments
        total FRiP score
    """

    # init tempfiles
    tempfiles = []
    # create path to tempfiles
    regions_bed = os.path.join(temp_dir, "regions.bed")
    overlap = os.path.join(temp_dir, "overlap.bed")
    # add tempfiles to list
    tempfiles.append(regions_bed)
    tempfiles.append(overlap)

    # write regions to bedfile
    logger.info("writing regions to bedfile")
    with open(regions_bed, "w") as out_file:
        for region in tqdm(adata.var.iterrows(), desc='extract adata.var regions'):
            line = str(region[1][0]) + '\t' + str(region[1][1]) + '\t' + str(region[1][2]) + '\n'
            out_file.write(line)
    out_file.close()

    # overlap fragments with regions
    logger.info("overlapping bedfiles")
    if ~utils._bed_is_sorted(fragments):
        # sort fragments
        sorted_fragments = os.path.join(temp_dir, "sorted_fragments.bed")
        utils._sort_bed(fragments, sorted_fragments)
        fragments = sorted_fragments
        # add sorted fragments to tempfiles
        tempfiles.append(sorted_fragments)

    # overlap sorted fragments with regions
    utils._overlap_two_bedfiles(fragments, regions_bed, overlap)

    # read in bedfiles
    logger.info('reading in bedfiles')
    overlap_list = utils._read_bedfile(overlap)
    fragments_list = utils._read_bedfile(fragments)

    logger.info("calculating total number of fragments and overlaps")
    # calculate total number of fragments and overlaps
    fragments_df = pd.DataFrame(fragments_list, columns=['chr', 'start', 'stop', 'barcode', 'n'])
    total_fragments = fragments_df['n'].sum()

    # calculate total number of overlaps
    overlaps_df = pd.DataFrame(overlap_list, columns=['chr', 'start', 'stop', 'barcode', 'n'])
    total_overlaps = overlaps_df['n'].sum()

    # calculate total FRiP score
    total_frip = total_overlaps / total_fragments
    logger.info('total_frip: ' + str(total_frip))

    # calculate FRiP score per barcode
    logger.info("calculating FRiP scores per barcode")
    # count fragments per barcode
    logger.info('count fragments per barcode')
    counts_ov = _count_fragments(overlaps_df)
    counts_total = _count_fragments(fragments_df)

    # merge counts
    counts_ov = counts_ov.set_index('barcode')
    counts_total = counts_total.set_index('barcode')

    counts_ov = counts_ov.rename(columns={'n': 'n_ov'})
    counts_total = counts_total.rename(columns={'old_name': 'n_total'})

    combined = counts_ov.join(counts_total)

    # calculate FRiP score per barcode
    logger.info('calculating FRiP score per barcode and adding it to adata.obs')
    combined['frip'] = combined['n_ov'] / combined['n']

    # add FRiP score to adata.obs
    adata.obs = adata.obs.join(combined['frip'])

    # remove tempfiles
    for file in tempfiles:
        os.remove(file)

    return adata, total_frip
