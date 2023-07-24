import pysam
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import sctoolbox.tools as tools
import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


def write_TSS_bed(gtf, custom_TSS, negativ_shift=2000, positiv_shift=2000, temp_dir=None):
    """
    This writes a custom TSS file from a gene gtf file.
    negativ_shift and positiv_shift are the number of bases of the flanks.
    The output is a bed file with the following columns:
    1: chr, 2: start, 3:stop
    and a list of lists with the following columns:
    1: chr, 2: start, 3:stop

    Parameters
    ----------
    gtf: str
        path to gtf file
    custom_TSS: str
        path to output file
    negativ_shift: int
        number of bases to shift upstream
    positiv_shift: int
        number of bases to shift downstream

    Returns
    -------
    tss_list: list
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


def overlap_and_aggregate(fragments, custom_TSS, overlap, tss_list, negativ_shift=2000, positiv_shift=2000):
    """
    This function overlaps the fragments with the custom TSS file and aggregates the fragments in a dictionary.
    The dictionary has the following structure:
    {barcode: [tss_agg, n_fragments]}
    tss_agg is a numpy array with the aggregated fragments around the TSS with flanks of negativ_shift and positiv_shift.
    n_fragments is the number of fragments overlapping the TSS.

    Parameters
    ----------
    fragments: str
        path to fragments file
    custom_TSS: str
        path to custom TSS file
    overlap: str
        path to output file
    tss_list: list
        list of lists with the following columns:
        1: chr, 2: start, 3:stop
    negativ_shift: int
        number of bases to shift upstream
    positiv_shift: int
        number of bases to shift downstream

    Returns
    -------
    tSSe_cells: dict
        dictionary with the following structure:
        {barcode: [tss_agg, n_fragments]}
        tss_agg is a numpy array with the aggregated fragments around the TSS with flanks of negativ_shift and positiv_shift.
        n_fragments is the number of fragments overlapping the TSS.

    """
    # Overlap two bedfiles (FRAGMENTS / TSS_sites)
    logger.info("overlapping fragments with TSS")
    bedtools = os.path.join('/'.join(sys.executable.split('/')[:-1]), 'bedtools')
    intersect_cmd = f'{bedtools} intersect -a {fragments} -b {custom_TSS} -u -sorted > {overlap}'
    # run command
    os.system(intersect_cmd)

    # Read in overlap file
    logger.info("opening overlap file")
    #overlap_list = []
    #with open(overlap, 'rb') as file:
    #    for row in file:
    #        row = row.decode("utf-8")
    #        row = row.split('\t')
    #        line = [str(row[0]), int(row[1]), int(row[2]), str(row[3]), int(row[4])]
    #        overlap_list.append(line)

    overlap_list = utils._read_bedfile(overlap)

    # initialize dictionary
    tSSe_cells = {}
    # initialize counter for overlap_list
    k = 0

    # fragments:
    # 1: chr, 2: start, 3:stop, 4:barcode, 5:n

    # tss:
    # 1: chr, 2: start, 3:stop


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

    return tSSe_cells


def calc_per_base_tsse(tSSe_df, min_bias=0.01, edge_size=100):
    """
    calculate per base tSSe by dividing the tSSe by the bias. The bias is calculated by averaging the edges of the tSSe.
    The edges are defined by the edge_size.

    Parameters
    ----------
    tSSe_df: pandas.DataFrame
        dataframe with the following columns:
        1: barcode, 2: tSSe, 3: n_fragments
    min_bias: float
        minimum bias to avoid division by zero
    edge_size: int
        number of bases to use for the edges

    Returns
    -------
    per_base_tsse: numpy.array
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


def global_tsse_score(per_base_tsse, negativ_shift, edge_size=50):
    """
    This function calculates the global tSSe score by averaging the per base tSSe around the TSS.

    Parameters
    ----------
    per_base_tsse: numpy.array
        numpy array with the per base tSSe
    negativ_shift: int
        number of bases to shift upstream
    edge_size: int
        number of bases to use for the edges

    Returns
    -------
    global_tsse_score: numpy.array
        numpy array with the global tSSe score
    """
    # calculate global tSSe score
    logger.info("calculating global tSSe score")
    center = per_base_tsse[:,negativ_shift-edge_size:negativ_shift+edge_size]
    global_tsse_score = np.sum(center, axis=1) / (edge_size*2)

    return global_tsse_score


@deco.log_anndata
def add_tsse_score(adata,
                   fragments,
                   gtf,
                   negativ_shift=2000,
                   positiv_shift=2000,
                   edge_size_total=100,
                   edge_size_per_base=50,
                   min_bias=0.01,
                   keep_tmp=False,
                   temp_dir=""):
    """
    This function adds the tSSe score to the adata object.
    Calculating the TSSe score is done like described in: "Chromatin accessibility profiling by ATAC-seq" Fiorella et al. 2022
    Parameters
    ----------
    adata: AnnData
        AnnData object
    fragments: str
        path to fragments.bed
    gtf: str
        path to gtf file
    custom_TSS: str
        path to custom TSS bed file
    overlap: str
        path to overlap file
    negativ_shift: int
        number of bases to shift upstream for the flanking regions
    positiv_shift: int
        number of bases to shift downstream for the flanking regions
    edge_size_total: int
        number of bases to use for the edges for the global tSSe score
    edge_size_per_base: int
        number of bases to use for the edges for the per base tSSe score
    min_bias: float
        minimum bias to avoid division by zero
    keep_tmp: bool
        keep temporary files

    Returns
    -------
    adata: AnnData
        AnnData object with added tSSe score
    """
    logger.info("adding tSSe score to adata object")
    # create list for temporary files
    tmp_files = []
    # create temporary file paths
    custom_TSS = os.path.join(temp_dir, "custom_TSS.bed")
    overlap = os.path.join(temp_dir, "overlap.bed")
    # add temporary file paths to list
    tmp_files.append(custom_TSS)
    tmp_files.append(overlap)

    # write custom TSS bed file
    tss_list, tmp_files_tss = write_TSS_bed(gtf, custom_TSS, negativ_shift=negativ_shift, positiv_shift=positiv_shift, temp_dir=temp_dir)
    # add temporary file paths to list
    tmp_files.extend(tmp_files_tss)
    # overlap fragments with custom TSS
    tSSe_cells = overlap_and_aggregate(fragments, custom_TSS, overlap, tss_list)
    # calculate per base tSSe
    tSSe_df = pd.DataFrame.from_dict(tSSe_cells, orient='index', columns=['TSS_agg', 'total_ov'])
    # calculate per base tSSe
    per_base_tsse = calc_per_base_tsse(tSSe_df, min_bias=min_bias, edge_size=edge_size_total)
    # calculate global tSSe score
    tsse_score = global_tsse_score(per_base_tsse, negativ_shift, edge_size=edge_size_per_base)
    # add tSSe score to adata
    tSSe_df['TSSe_score'] = tsse_score
    # add tSSe score to adata
    adata.obs = adata.obs.join(tSSe_df['TSSe_score'])

    # remove temporary files
    if not keep_tmp:
        logger.info("cleaning up temporary files")
        for tmp_file in tmp_files:
            os.remove(tmp_file)

    return adata