"""Module to calculate TSS enrichment scores."""
import scanpy as sc
import pysam
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import sctoolbox.tools as tools
import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings

from beartype.typing import Tuple, Optional
from beartype import beartype
import numpy.typing as npt

logger = settings.logger


@beartype
def write_TSS_bed(gtf: str,
                  custom_TSS: str,
                  negativ_shift: int = 2000,
                  positiv_shift: int = 2000,
                  temp_dir: Optional[str] = None) -> Tuple[list, list]:
    """
    Write a custom TSS file from a gene gtf file.

    negativ_shift and positiv_shift are the number of bases of the flanks.
    The output is a bed file with the following columns:
    1: chr, 2: start, 3:stop
    and a list of lists with the following columns:
    1: chr, 2: start, 3:stop

    Parameters
    ----------
    gtf : str
        path to gtf file
    custom_TSS : str
        path to output file
    negativ_shift : int, default 2000
        number of bases to shift upstream
    positiv_shift : int, default 2000
        number of bases to shift downstream
    temp_dir : Optional[str], default None
        path to temporary directory

    Returns
    -------
    Tuple[list, list]
        list of lists with the following columns:
        1: chr, 2: start, 3:stop
    """

    # Unzip, sort and index gtf if necessary
    gtf, tempfiles = tools._prepare_gtf(gtf, temp_dir)
    # Open tabix file
    tabix_obj = pysam.TabixFile(gtf)
    # Create list of TSS
    tss_list = []
    # Write TSS to file and list
    logger.info("building temporary TSS file")
    with open(custom_TSS, "w") as out_file:
        for gene in tabix_obj.fetch(parser=pysam.asGTF()):
            tss_list.append([gene.contig, gene.start - negativ_shift, gene.start + positiv_shift])
            line = str(gene.contig) + '\t' + str(max(0, gene.start - negativ_shift)) + '\t' + str(
                gene.start + positiv_shift) + '\n'
            out_file.write(line)

    return tss_list, tempfiles


@beartype
def overlap_and_aggregate(fragments: str,
                          custom_TSS: str,
                          overlap: str,
                          tss_list: list[list[str | int]],
                          negativ_shift: int = 2000,
                          positiv_shift: int = 2000) -> Tuple[dict[str, list], list[str]]:
    """
    Overlap the fragments with the custom TSS file and aggregates the fragments in a dictionary.

    The dictionary has the following structure:
    {barcode: [tss_agg, n_fragments]}
    tss_agg is a numpy array with the aggregated fragments around the TSS with flanks of negativ_shift and positiv_shift.
    n_fragments is the number of fragments overlapping the TSS.

    Parameters
    ----------
    fragments : str
        path to fragments file
    custom_TSS : str
        path to custom TSS file
    overlap : str
        path to output file
    tss_list : list[list[str | int]]
        list of lists with the following columns:
        1: chr, 2: start, 3:stop
    negativ_shift : int, default 2000
        number of bases to shift upstream
    positiv_shift : int, default 2000
        number of bases to shift downstream

    Returns
    -------
    Tuple[dict[str, list], list[str]]
        dictionary with the following structure:
        {barcode: [tss_agg, n_fragments]}
        tss_agg is a numpy array with the aggregated fragments around the TSS with flanks of negativ_shift and positiv_shift.
        n_fragments is the number of fragments overlapping the TSS.
        list[str] contains a list of temporary files.

    """
    tempfiles = []
    # Check if fragments are sorted
    if utils._bed_is_sorted(fragments):
        logger.info("fragments are sorted")
    else:
        logger.info("sorting fragments")
        # build path to sorted fragments
        base_name = os.path.basename(fragments)
        file_name = os.path.splitext(base_name)[0]
        sorted_bedfile = os.path.join(os.path.split(fragments)[0], (file_name + '_sorted.bed'))
        tempfiles.append(sorted_bedfile)
        # sort fragments
        utils._sort_bed(fragments, sorted_bedfile)
        # sorted fragments to be used for overlap
        fragments = sorted_bedfile

    # Overlap two bedfiles (FRAGMENTS / TSS_sites)
    logger.info("overlapping fragments with TSS")
    bedtools = os.path.join('/'.join(sys.executable.split('/')[:-1]), 'bedtools')
    intersect_cmd = f'{bedtools} intersect -a {fragments} -b {custom_TSS} -u -sorted > {overlap}'
    # run command
    os.system(intersect_cmd)

    # Read in overlap file
    logger.info("opening overlap file")

    overlap_list = utils._read_bedfile(overlap)

    # initialize dictionary
    tSSe_cells = {}
    # initialize counter for overlap_list
    k = 0

    # Aggregate Overlap
    logger.info("aggregating fragments")
    for tss in tqdm(tss_list, desc='Aggregating'):
        for i in range(k, len(overlap_list)):

            fragment = overlap_list[i]
            # update counter
            k = i
            # check if fragment is on the same chromosome and start of fragment is smaller than end of tss
            overlap_condition = (fragment[0] == tss[0]) & \
                                (fragment[1] <= tss[2])

            # break if not overlap_condition
            if not overlap_condition:
                break

            # calculate start and stop of the overlap
            start = max(0, fragment[1] - tss[1])
            stop = min(-1, fragment[2] - tss[2])

            # calculate number of fragments
            n_fragments = fragment[4]

            # check if barcode is already in dictionary. If yes, update tss_agg and n_fragments
            if fragment[3] in tSSe_cells:
                new_tss_agg = tSSe_cells[fragment[3]][0]
                new_tss_agg[start:stop] = tSSe_cells[fragment[3]][0][start:stop] + n_fragments
                new_count = tSSe_cells[fragment[3]][1] + n_fragments
                tSSe_cells[fragment[3]] = [new_tss_agg, new_count]
            # if not, create new entry in dictionary
            else:
                tSS_agg = np.zeros(negativ_shift + positiv_shift, dtype=int)
                tSS_agg[start:stop] = n_fragments
                tSSe_cells[fragment[3]] = [tSS_agg, n_fragments]

    return tSSe_cells, tempfiles


@beartype
def calc_per_base_tsse(tSSe_df: pd.DataFrame,
                       min_bias: float = 0.01,
                       edge_size: int = 100) -> npt.ArrayLike:
    """
    Calculate per base tSSe by dividing the tSSe by the bias. The bias is calculated by averaging the edges of the tSSe.

    The edges are defined by the edge_size.

    Parameters
    ----------
    tSSe_df : pd.DataFrame
        dataframe with the following columns:
        1: barcode, 2: tSSe, 3: n_fragments
    min_bias : float, default 0.01
        minimum bias to avoid division by zero
    edge_size : int, default 100
        number of bases to use for the edges

    Returns
    -------
    npt.ArrayLike
        numpy array with the per base tSSe
    """
    # make np.array for calculation
    tSS_agg_arr = np.array(tSSe_df['TSS_agg'].to_list())

    # get the edges as bias
    logger.info("calculating bias")
    border_upstream = tSS_agg_arr[:, 0:edge_size]
    border_downstream = tSS_agg_arr[:, -edge_size:]

    # Concatenate along the second axis
    edges = np.concatenate((border_upstream, border_downstream), axis=1)

    # calculate bias
    bias = np.sum(edges, axis=1) / 200
    bias[bias == 0] = min_bias  # Can I do this?

    # calculate per base tSSe
    logger.info("calculating per base tSSe")
    per_base_tsse = tSS_agg_arr / bias[:, None]

    return per_base_tsse


@beartype
def global_tsse_score(per_base_tsse: npt.ArrayLike,
                      negativ_shift: int,
                      edge_size: int = 50) -> npt.ArrayLike:
    """
    Calculate the global tSSe score by averaging the per base tSSe around the TSS.

    Parameters
    ----------
    per_base_tsse : npt.ArrayLike
        numpy array with the per base tSSe
    negativ_shift : int
        number of bases to shift upstream
    edge_size : int, default 50
        number of bases to use for the edges

    Returns
    -------
    npt.ArrayLike
        numpy array with the global tSSe score
    """
    # calculate global tSSe score
    logger.info("calculating global tSSe score")
    center = per_base_tsse[:, negativ_shift - edge_size:negativ_shift + edge_size]
    global_tsse_score = np.sum(center, axis=1) / (edge_size * 2)

    return global_tsse_score


@beartype
def tsse_scoring(fragments: str,
                 gtf: str,
                 negativ_shift: int = 2000,
                 positiv_shift: int = 2000,
                 edge_size_total: int = 100,
                 edge_size_per_base: int = 50,
                 min_bias: float = 0.01,
                 keep_tmp: bool = False,
                 temp_dir: str = "",
                 plot: bool = False) -> pd.DataFrame:
    """
    Calculate the tSSe score for each cell.

    Calculating the TSSe score is done like described in: "Chromatin accessibility profiling by ATAC-seq" Fiorella et al. 2022

    Parameters
    ----------
    fragments : str
        path to fragments file
    gtf : str
        path to gtf file
    negativ_shift : int, default 2000
        number of bases to shift upstream
    positiv_shift : int, default 2000
        number of bases to shift downstream
    edge_size_total : int, default 100
        number of bases to use for the edges for the global tSSe score
    edge_size_per_base : int, default 50
        number of bases to use for the edges for the per base tSSe score
    min_bias : float, default 0.01
        minimum bias to avoid division by zero
    keep_tmp : bool, default False
        keep temporary files
    temp_dir : str, default ""
        path to temporary directory
    plot : bool, default False
        plot a single aggregate as reference

    Returns
    -------
    pd.DataFrame
        dataframe with the following columns:
        1: barcode, 2: tSSe, 3: n_fragments
    """

    tmp_files = []
    # create temporary file paths
    custom_TSS = os.path.join(temp_dir, "custom_TSS.bed")
    overlap = os.path.join(temp_dir, "overlap.bed")
    # add temporary file paths to list
    tmp_files.append(custom_TSS)
    tmp_files.append(overlap)

    # write custom TSS bed file
    tss_list, temp = write_TSS_bed(gtf, custom_TSS, negativ_shift=negativ_shift, positiv_shift=positiv_shift,
                                   temp_dir=temp_dir)
    # add temporary file paths to list
    tmp_files.extend(temp)
    # overlap fragments with custom TSS
    tSSe_cells, temp = overlap_and_aggregate(fragments, custom_TSS, overlap, tss_list)
    # add temporary file paths to list
    tmp_files.extend(temp)
    # calculate per base tSSe
    tSSe_df = pd.DataFrame.from_dict(tSSe_cells, orient='index', columns=['TSS_agg', 'total_ov'])
    # Plot a single aggregate as reference
    if plot:
        plt.plot(tSSe_df['TSS_agg'].to_numpy()[0])
    # calculate per base tSSe
    per_base_tsse = calc_per_base_tsse(tSSe_df, min_bias=min_bias, edge_size=edge_size_total)
    # calculate global tSSe score
    tsse_score = global_tsse_score(per_base_tsse, negativ_shift, edge_size=edge_size_per_base)
    # add tSSe score to adata
    tSSe_df['tsse_score'] = tsse_score

    # remove temporary files
    if not keep_tmp:
        logger.info("cleaning up temporary files")
        for tmp_file in tmp_files:
            os.remove(tmp_file)

    return tSSe_df


@deco.log_anndata
@beartype
def add_tsse_score(adata: sc.AnnData,
                   fragments: str,
                   gtf: str,
                   negativ_shift: int = 2000,
                   positiv_shift: int = 2000,
                   edge_size_total: int = 100,
                   edge_size_per_base: int = 50,
                   min_bias: float = 0.01,
                   keep_tmp: bool = False,
                   temp_dir: str = "",
                   plot: bool = False,
                   return_aggs: bool = False) -> sc.AnnData | Tuple[sc.AnnData, pd.DataFrame]:
    """
    Add the tSSe score to the adata object.

    Calculating the TSSe score is done like described in: "Chromatin accessibility profiling by ATAC-seq" Fiorella et al. 2022

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object
    fragments : str
        path to fragments.bed
    gtf : str
        path to gtf file
    negativ_shift : int, default 2000
        number of bases to shift upstream for the flanking regions
    positiv_shift : int, default 2000
        number of bases to shift downstream for the flanking regions
    edge_size_total : int, default 100
        number of bases to use for the edges for the global tSSe score
    edge_size_per_base : int, default 50
        number of bases to use for the edges for the per base tSSe score
    min_bias : float, default 0.01
        minimum bias to avoid division by zero
    keep_tmp : bool, default False
        keep temporary files
    temp_dir : str, default ""
        path to temporary directory
    plot : bool, default False
        plot a single aggregate as reference
    return_aggs : bool, default False
        return the aggregated fragments

    Returns
    -------
    sc.AnnData | Tuple[sc.AnnData, pd.DataFrame]
        AnnData object with added tSSe score
    """

    logger.info("adding tSSe score to adata object")

    tSSe_df = tsse_scoring(fragments,
                           gtf,
                           negativ_shift=negativ_shift,
                           positiv_shift=positiv_shift,
                           edge_size_total=edge_size_total,
                           edge_size_per_base=edge_size_per_base,
                           min_bias=min_bias,
                           keep_tmp=keep_tmp,
                           temp_dir=temp_dir,
                           plot=plot)

    # add tSSe score to adata
    if 'tsse_score' in adata.obs.columns:
        logger.warning("tsse_score already in adata.obs. Overwriting tsse_score.")
        adata.obs = adata.obs.drop(columns='tsse_score')
    adata.obs = adata.obs.join(tSSe_df['tsse_score'])

    if return_aggs:
        return adata, tSSe_df
    else:
        return adata
