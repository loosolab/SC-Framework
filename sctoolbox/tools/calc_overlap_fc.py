"""Module to calculate fold change of reads or fragments from a BAM or fragments file that overlap specified regions."""

import os
import pandas as pd
from pathlib import Path
import scanpy as sc

from beartype import beartype
from beartype.typing import Optional, Tuple

import sctoolbox.utils as utils
from sctoolbox.tools.bam import create_fragment_file
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


@beartype
def _convert_gtf_to_bed(gtf: str,
                        out: Optional[str] = None,
                        temp_files: list[str] = []) -> Tuple[str, list[str]]:
    """
    Convert GTF-file to BED file.

    Extract 'chr', 'start' and 'stop' from .gtf file and convert it to sorted BED file.
    BED file will be sorted by chromosome name and start position.

    Parameters
    ----------
    gtf : str
        Path to .gtf file.
    out : Optional[str], default None
        Path to save new BED file. If none, the file will be saved in the same folder as tha BAM file.
    temp_files : list[str], default []
        List of temporary files.

    Returns
    -------
    Tuple[str, list[str]]
        Path to fragments and temp files.
    """

    name = os.path.basename(gtf)

    if not out:
        out = os.getcwd()

    out_unsorted = os.path.join(out, f'{name}_tmp.bed')
    out_sorted = os.path.join(out, f'{name}_sorted.bed')

    # add temp file to list
    temp_files.append(out_sorted)

    with open(gtf, 'rb') as file:
        with open(out_unsorted, "w") as out_file:
            for row in file:
                row = row.decode("utf-8")  # convert binary to string
                if row.startswith('#'):  # skip if header
                    continue
                row = row.split('\t')
                line = row[0] + '\t' + row[3] + '\t' + row[4] + '\n'
                out_file.write(line)
    # sort gtf
    utils.bioutils._sort_bed(out_unsorted, out_sorted)

    # remove unsorted
    os.remove(out_unsorted)

    # return the path to sorted bed
    return out_sorted, temp_files


@deco.log_anndata
@beartype
def fc_fragments_in_regions(adata: sc.AnnData,
                            regions_file: str,
                            bam_file: Optional[str] = None,
                            fragments_file: Optional[str] = None,
                            cb_col: Optional[str] = None,
                            cb_tag: str = 'CB',
                            regions_name: str = 'list',
                            threads: Optional[int] = 4,
                            temp_dir: Optional[str] = None) -> None:
    """
    Calculate the fold change of fragments in a region against the background.

    This function calculates the fold change of fragments overlapping a region against the background for each cell.
    The regions are specified in a BED or GTF file and the fragments should be provided by a fragments or BAM file.
    The results are added to the anndata object as new column.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object containig cell barcodes in adata.obs.
    regions_file : str
        Path to BED or GTF file containing regions of interest.
    bam_file : Optional[str], default None
        Path to BAM file. If None, a fragments file must be provided in the parameter 'fragments_file'.
    fragments_file : Optional[str], default None
        Path to fragments file. If None, a BAM file must be provided in the parameter 'bam_file'. The
        BAM file will be converted into fragments file.
    cb_col : Optional[str], default None
        The column in adata.obs containing cell barcodes. If None, adata.obs.index will be used.
    cb_tag : str, default 'CB'
        The tag where cell barcodes are saved in the bam file. Set to None if the barcodes are in read names.
    regions_name : str, default 'list'
        The name of the regions in the BED or GTF file (e.g. Exons). The name will be used as columns' name
        added to the anndata object (e.g. pct_fragments_in_{regions_name}).
    threads : Optional[int], default 4
        Number of threads for parallelization. Will be used to convert BAM to fragments file. None to use settings.get_threads.
    temp_dir : Optional[str], default None
        Path to temporary directory. Will use the current working directory by default.

    Raises
    ------
    ValueError
        If bam_file and fragment file is not provided.
    """
    if threads is None:
        threads = settings.get_threads()

    if temp_dir:
        utils.io.create_dir(temp_dir)
    else:
        temp_dir = os.getcwd()

    if not bam_file and not fragments_file:
        raise ValueError("Either BAM file or fragments file has to be provided!")

    # check for column in adata.obs where barcodes are
    if cb_col:
        try:
            adata.obs.set_index(cb_col)
        except KeyError:
            logger.error(f"{cb_col} is not in adata.obs!")
            return

    temp_files = []
    # check if regions file is gtf or bed
    file_ext = Path(regions_file).suffix
    if file_ext.lower() == '.gtf':
        logger.info("Converting GTF to BED...")
        # convert gtf to bed with columns chr, start, end
        bed_file, temp_files = _convert_gtf_to_bed(regions_file, temp_files=temp_files, out=temp_dir)

    elif file_ext.lower() == '.bed':
        bed_file = os.path.join(temp_dir, os.path.splitext(os.path.split(regions_file)[1])[0]) + '_sorted.bed'
        temp_files.append(bed_file)
        if not utils.bioutils._bed_is_sorted(regions_file):
            logger.info("Sorting BED file...")
            utils.bioutils._sort_bed(regions_file, bed_file)
        else:
            bed_file = regions_file

    # if only bam file is available -> convert to fragments
    if bam_file and not fragments_file:
        logger.info('Converting BAM to fragments file! This may take a while...')
        fragments_file = create_fragment_file(bam=bam_file,
                                              barcode_tag=cb_tag,
                                              outdir=temp_dir,
                                              nproc=threads)
        temp_files.append(fragments_file)

    # Check if fragments file is sorted
    if not utils.bioutils._bed_is_sorted(fragments_file):
        # sort fragments file
        logger.info('Sorting fragments file...')
        unsorted_fragments_file = fragments_file
        fragments_file = os.path.join(temp_dir, os.path.splitext(os.path.split(unsorted_fragments_file)[1])[0]) + '_sorted.bed'
        utils.bioutils._sort_bed(unsorted_fragments_file, fragments_file)

    # overlap reads in fragments with promoter regions, return path to overlapped file
    logger.info('Finding overlaps...')
    overlap_file = os.path.join(temp_dir, 'overlap_fragments_gtf.bed')
    utils.bioutils._overlap_two_bedfiles(fragments_file, bed_file, overlap=overlap_file, wa=True, wb=True, sorted=True)
    temp_files.append(overlap_file)

    # read overlap bedfile as dataframe
    overlap_df = pd.read_csv(overlap_file,
                             delimiter='\t',
                             header=None,
                             names=['chr_f', 'start_f', 'stop_f', 'barcode', 'count', 'chr_g', 'start_g', 'stop_g'])

    # read fragments bedfile as dataframe
    fragments_df = pd.read_csv(fragments_file,
                               delimiter='\t',
                               header=None,
                               names=['chr', 'start', 'stop', 'barcode', 'count'])

    # count fragments per cell
    counts_ov = pd.DataFrame(count_fragments_per_cell(overlap_df, barcode_col='barcode', frag_count='count'))  # count fragments per cell in overlap
    counts_all = pd.DataFrame(count_fragments_per_cell(fragments_df, barcode_col='barcode', frag_count='count'))  # count fragments per cell in all

    # merge counts
    merged_df = counts_ov.join(counts_all, how='outer', lsuffix='_ov', rsuffix='_all')  # merge outer to keep all cells

    # calculate fold change
    logger.info('Calculating fold change...')
    column_name = f'fold_change_{regions_name}_fragments'  # name of the new column
    merged_df[column_name] = merged_df['count_ov'] / merged_df['count_all']  # calculate fold change (Maybe better to use log2 fold change?)

    # fill NaN with 0 as NaN means no fragments in that cell
    merged_df = merged_df.fillna(0)

    # add to adata.obs
    logger.info('Adding results to adata object...')
    adata.obs = adata.obs.join(merged_df[column_name], how='left')  # merge left to keep only cells in adata.obs

    # clean up temp files
    logger.info("cleaning up...")
    utils.io.rm_tmp(temp_dir=temp_dir,
                    temp_files=temp_files,
                    rm_dir=False if temp_dir else True)


def count_fragments_per_cell(df: pd.DataFrame, barcode_col: str = 'barcode', frag_count: str = 'count') -> pd.DataFrame:
    """
    Get counts per cell from a dataframe containing fragments.

    The dataframe must have a column with cell barcodes and a column with fragment counts.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe of a bedfile containing the fragments.
    barcode_col : str, default 'barcode'
        The column name containing the cell barcodes.
    frag_count : str, default 'count'
        The column name containing the fragment counts.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the number of fragments per cell.
    """

    fragments_per_cell = df.groupby(barcode_col)[frag_count].sum()

    return fragments_per_cell
