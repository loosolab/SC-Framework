"""
Module to calculate percentage of reads from a BAM or fragments file that overlap promoter regions.

Module to calculate percentage of reads from a BAM or fragments file that overlap promoter regions specified
in a GTF file using 'pct_reads_in_promoters' function.
pct_reads_overlap() calculates percentage of
reads that overlap with regions specified in any BED file. The BED file must have three columns ['chr','start','end']
as first columns.
"""

from collections import Counter
import os
import pkg_resources
import pandas as pd
from pathlib import Path
import multiprocessing as mp
import scanpy as sc

from beartype import beartype
from beartype.typing import Optional, Tuple, Union, Literal, Any

import sctoolbox.utils.bioutils as utils
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
    temp_dir : str, default '.'
        Path to temporary directory.

    Returns
    -------
    Tuple[str, list[str]]
        Path to fragments and temp files.
    """

    if not out:
        path = os.path.splitext(gtf)
        out_unsorted = f"{path[0]}_tmp.bed"
        out_sorted = f"{path[0]}.gtf_sorted.bed"
    else:
        name = os.path.basename(gtf)
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
    utils._sort_bed(out_unsorted, out_sorted)

    # remove unsorted
    os.remove(out_unsorted)

    # return the path to sorted bed
    return out_sorted, temp_files


@deco.log_anndata
@beartype
def pct_fragments_in_promoters(adata: sc.AnnData,
                               gtf_file: Optional[str] = None,
                               bam_file: Optional[str] = None,
                               fragments_file: Optional[str] = None,
                               cb_col: Optional[str] = None,
                               cb_tag: str = 'CB',
                               species: Optional[Literal['bos_taurus', 'caenorhabditis_elegans',
                                                         'canis_lupus_familiaris', 'danio_rerio',
                                                         'drosophila_melanogaster', 'gallus_gallus',
                                                         'homo_sapiens', 'mus_musculus', 'oryzias_latipes',
                                                         'rattus_norvegicus', 'sus_scrofa', 'xenopus_tropicalis']] = None,
                               nproc: int = 1,
                               sort_bam: bool = False) -> None:
    """
    Calculate the percentage of fragments in promoters.

    This function calculates for each cell, the percentage of fragments in a BAM alignment file
    that overlap with a promoter region specified in a GTF file. The results are added to the anndata object
    as new columns (n_total_fragments, n_fragments_in_promoters and pct_fragments_in_promoters).
    This is a wrapper function for pct_fragments_overlap.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object containig cell barcodes in adata.obs.
    gtf_file : str, default None
        Path to GTF file for promoters regions. if None, the GTF file in flatfiles directory will be used.
    bam_file : str, default None
        Path to BAM file. If None, a fragments file must be provided in the parameter 'fragments_file'.
    fragments_file : str, default None
        Path to fragments file. If None, a BAM file must be provided in the parameter 'bam_file'. The
        BAM file will be converted into fragments file.
    cb_col : str, default None
        The column in adata.obs containing cell barcodes. If None, adata.obs.index will be used.
    cb_tag : str, default 'CB
        The tag where cell barcodes are saved in the bam file. Set to None if the barcodes are in read names.
    species : str, default None
        Name of the species, will only be used if gtf_file is None to use internal GTF files.
        Species are {bos_taurus, caenorhabditis_elegans, canis_lupus_familiaris, danio_rerio, drosophila_melanogaster,
        gallus_gallus, homo_sapiens, mus_musculus, oryzias_latipes, rattus_norvegicus, sus_scrofa, xenopus_tropicalis}
    nproc : int, default 1
        Number of threads for parallelization. Will be used to convert BAM to fragments file.
    sort_bam : bool, default False
        Set to True if the provided BAM file is not sorted.

    Raises
    ------
    ValueError
        If no species and no gtf_file is given.
    """

    # exit if no gtf file and no species
    if not gtf_file and not species:
        raise ValueError('Please provide a GTF file or specify a species!')
    if not gtf_file:
        promoters_gtf = pkg_resources.resource_filename("sctoolbox", f"data/promoters_gtf/{species}.104.promoters2000.gtf")
    else:
        promoters_gtf = gtf_file

    # call function
    pct_fragments_overlap(adata, regions_file=promoters_gtf, bam_file=bam_file, fragments_file=fragments_file,
                          cb_col=cb_col, cb_tag=cb_tag, regions_name='promoters', nproc=nproc, sort_bam=sort_bam)


@deco.log_anndata
@beartype
def pct_fragments_overlap(adata: sc.AnnData,
                          regions_file: str,
                          bam_file: Optional[str] = None,
                          fragments_file: Optional[str] = None,
                          cb_col: Optional[str] = None,
                          cb_tag: str = 'CB',
                          regions_name: str = 'list',
                          nproc: int = 1,
                          sort_bam: bool = False,
                          sort_regions: bool = False,
                          keep_fragments: bool = False,
                          temp_dir: Optional[str] = None) -> None:
    """
    Calculate the percentage of fragments.

    This function calculates for each cell, the percentage of fragments in a BAM alignment file
    that overlap with regions specified in a BED or GTF file. The results are added to the anndata object
    as new columns.

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
        to be added to the anndata object (e.g. pct_fragments_in_{regions_name}).
    nproc : int, default 1
        Number of threads for parallelization. Will be used to convert BAM to fragments file.
    sort_bam : bool, default False
        Set to True if the provided BAM file is not sorted.
    sort_regions : bool, default False
        If True sort bed file on regions.
    keep_fragments : bool, default False
        If True keep fragment files.
    temp_dir : Optional[str], default None
        Path to temporary directory.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If bam_file and fragment file is not provided.
    """

    if not bam_file and not fragments_file:
        raise ValueError("Either BAM file or fragments file has to be provided!")

    # check for column in adata.obs where barcodes are
    if cb_col:
        try:
            barcodes = adata.obs[cb_col].to_list()
        except KeyError:
            logger.error(f"{cb_col} is not in adata.obs!")
            return
    else:
        barcodes = adata.obs.index.to_list()

    # check if temp_dir is given
    if not temp_dir:
        # if not, use current working directory
        temp_dir = os.getcwd()

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
        if not utils._bed_is_sorted(regions_file):
            logger.info("Sorting BED file...")
            utils._sort_bed(regions_file, bed_file)
        else:
            bed_file = regions_file

    # if only bam file is available -> convert to fragments
    if bam_file and not fragments_file:
        logger.info('Converting BAM to fragments file! This may take a while...')
        fragments_file = create_fragment_file(bam=bam_file,
                                              barcode_tag=cb_tag,
                                              outdir=temp_dir,
                                              nproc=nproc)
        temp_files.append(fragments_file)

    # Check if fragments file is sorted
    if not utils._bed_is_sorted(fragments_file):
        # sort fragments file
        logger.info('Sorting fragments file...')
        unsorted_fragments_file = fragments_file
        fragments_file = os.path.join(temp_dir, os.path.splitext(os.path.split(unsorted_fragments_file)[1])[0]) + '_sorted.bed'
        utils._sort_bed(unsorted_fragments_file, fragments_file)

    # overlap reads in fragments with promoter regions, return path to overlapped file
    logger.info('Finding overlaps...')
    overlap_file = os.path.join(temp_dir, 'overlap_fragments_gtf.bed')
    utils._overlap_two_bedfiles(fragments_file, bed_file, overlap=overlap_file, wa=True, wb=True, sorted=True)
    temp_files.append(overlap_file)

    #
    #mp_calc_pct = MPOverlapPct()
    #mp_calc_pct.calc_pct(overlap_file, fragments_file, barcodes, adata, regions_name=regions_name, n_threads=8)

    #
    #logger.info('Adding results to adata object...')
    #logger.info("cleaning up...")
    #for f in temp_files:
    #    os.remove(f)
    #logger.info('Done')


@beartype
class MPOverlapPct():
    """
    Class to calculate percentage of fragments overlapping with regions of interest.

    Notes
    -----
    This class will be removed in the future and replaced by a function due to over engineering.
    Therefore this will be documented sparsely.
    """

    def __init__(self):
        """Init class variables."""

        self.merged_dict = None

    def calc_pct(self,
                 overlap_file: str,
                 fragments_file: str,
                 barcodes: list[str],
                 adata: sc.AnnData,
                 regions_name: str = 'list',
                 n_threads: int = 8) -> sc.AnnData:
        """Calculate percentage of fragments overlapping with regions of interest."""

        # check if there was an overlap
        if not overlap_file:
            logger.info("There was no overlap!")
            return

        # get unique barcodes from adata.obs
        barcodes = set(barcodes)

        # make columns names that will be added to adata.obs
        col_total_fragments = 'n_total_fragments'
        # if no name is given or None, set default name
        if not regions_name:
            col_n_fragments_in_list = 'n_fragments_in_list'
            col_pct_fragments = 'pct_fragments_in_list'
        else:
            col_n_fragments_in_list = 'n_fragments_in_' + regions_name
            col_pct_fragments = 'pct_fragments_in_' + regions_name

        # calculating percentage
        logger.info('Calculating percentage...')
        # read overlap file as dataframe
        ov_fragments = pd.read_csv(overlap_file, sep="\t", header=None, chunksize=1000000)
        merged_ov_dict = self.mp_counter(ov_fragments, barcodes=barcodes, column=col_n_fragments_in_list, n_threads=n_threads)
        fragments = pd.read_csv(fragments_file, sep="\t", header=None, chunksize=1000000)
        merged_fl_dict = self.mp_counter(fragments, barcodes=barcodes, column=col_total_fragments, n_threads=n_threads)

        # add to adata.obs
        adata.obs[col_n_fragments_in_list] = adata.obs.index.map(merged_ov_dict).fillna(0)
        adata.obs[col_total_fragments] = adata.obs.index.map(merged_fl_dict).fillna(0)

        # calc pct
        adata.obs[col_pct_fragments] = adata.obs[col_n_fragments_in_list] / adata.obs[col_total_fragments]

        return adata

    def get_barcodes_sum(self,
                         df: pd.DataFrame,
                         barcodes: list[str],
                         col_name: str) -> dict:
        """Get the sum of reads counts in each cell barcode."""

        # drop columns we dont need
        df.drop(df.iloc[:, 5:], axis=1, inplace=True)
        df.columns = ['chr', 'start', 'end', 'barcode', col_name]
        # remove barcodes not found in adata.obs
        df = df.loc[df['barcode'].isin(barcodes)]
        # drop chr start end columns
        df.drop(['chr', 'start', 'end'], axis=1, inplace=True)
        # get the sum of reads counts in each cell barcode
        df = df.groupby('barcode').sum()

        count_dict = df[col_name].to_dict()

        return count_dict

    def log_result(self, result: Any) -> None:
        """Log results from mp_counter."""

        if self.merged_dict:
            self.merged_dict = dict(Counter(self.merged_dict) + Counter(result))
            # print('merging')
        else:
            self.merged_dict = result

    def mp_counter(self, fragments: list[str], barcodes: list[str], column: str, n_threads: int = 8):
        """Count reads for each cell barcode in parallel."""

        pool = mp.Pool(n_threads, maxtasksperchild=48)
        jobs = []
        # split fragments into chunks
        for chunk in fragments:
            # apply async job wit callback function
            job = pool.apply_async(self.get_barcodes_sum, args=(chunk, barcodes, column), callback=self.log_result)
            jobs.append(job)
        # monitor progress
        #utils.monitor_jobs(jobs, description="Progress")
        # close pool
        pool.close()
        # wait for all jobs to finish
        pool.join()
        # reset settings
        returns = self.merged_dict
        self.merged_dict = None

        return returns

if __name__ == '__main__':

    promoters_gtf = "/mnt/flatfiles/organisms/new_organism/mus_musculus/104/mus_musculus.104.promoters2000.gtf"
    fragments_file = '/mnt/workspace2/jdetlef/experimental/mm10_atac_fragemnts.bed'
    h5ad_file = '/mnt/workspace2/jdetlef/experimental/mm10_atac.h5ad'

    adata = sc.read_h5ad(h5ad_file)

    pct_fragments_overlap(adata, regions_file=promoters_gtf, fragments_file=fragments_file, temp_dir='/mnt/workspace2/jdetlef/experimental/temp')

    print('Done')