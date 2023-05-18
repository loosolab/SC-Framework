"""
    Module to calculate percentage of reads from a BAM or fragments file that overlap promoter regions specified
    in a GTF file using 'pct_reads_in_promoters' function. The function 'pct_reads_overlap' calculates percentage of
    reads that overlap with regions specified in any BED file. The BED file must have three columns ['chr','start','end']
    as first columns.
"""
from collections import Counter
import sctoolbox.utils as utils
import os
import pkg_resources
import pandas as pd
import pybedtools
import sys
from pathlib import Path
from sinto.fragments import fragments
import multiprocessing as mp


def check_pct_fragments_in_promoters(adata, qc_col):
    """
    Check if percentage of reads in promoters is in adata.obs.
    :param adata: anndata.AnnData
        AnnData object
    :return boolean
        True if pct_reads_in_promoters column is in adata.obs, otherwise False.
    """

    if qc_col in adata.obs.columns:
        return True
    else:
        return False


def create_fragment_file(bam, cb_tag='CB', out=None, nproc=1, sort_bam=False, keep_temp=False, temp_files=[]):
    """
    Create fragments file out of a BAM file using the package sinto

    :param bam: str
        Path to .bam file.
    :param cb_tag: str
        The tag where cell barcodes are saved in the bam file. Set to None if the barcodes are in read names.
    :param nproc: int
        Number of threads for parallelization.
    :param out: str
        Path to save fragments file. If none, the file will be saved in the same folder as tha BAM file.
    :param sort_bam: boolean
        Set to True if the provided BAM file is not sorted.
    :return: str
        Path to fragments file.
    """

    utils.check_module("pysam")
    import pysam

    # extract bam file name and path

    if not out:
        path = os.path.splitext(bam)
        out_unsorted = f"{path[0]}_fragments.bed"
        out_sorted = f"{path[0]}_fragments_sorted.bed"
        if sort_bam:
            bam_sorted = f"{path[0]}_sorted.bam"
            temp_files.append(bam_sorted)
    else:
        name = os.path.basename(bam).split('.')[0]
        out_unsorted = os.path.join(out, f"{name}_fragments.bed")
        out_sorted = os.path.join(out, f"{name}_fragments_sorted.bed")
        if sort_bam:
            bam_sorted = os.path.join(out, f"{name}_sorted.bam")
            temp_files.append(bam_sorted)

    if not keep_temp:
        temp_files.append(out_sorted)
    # sort bam if not sorted
    if sort_bam:
        print("Sorting BAM file...")
        pysam.sort("-o", bam_sorted, bam)
        bam = bam_sorted
        pysam.index(bam)

    # check for bam index file
    if not os.path.exists(bam + ".bai"):
        print("Bamfile has no index - trying to index with pysam...")
        pysam.index(bam)

    # sinto = os.path.join('/'.join(sys.executable.split('/')[:-1]),'sinto')
    # create_cmd = f'''{sinto} fragments -b {bam} -p {nproc} -f {out} --barcode_regex "[^:]*"'''
    # execute command
    # os.system(create_cmd)

    # use tag 'CB' if barcodes are stored in a tag, otherwise extract barcodes from read names
    readname_bc = None if cb_tag else "[^:]*"
    # run sinto
    fragments(bam, out_unsorted, nproc=nproc, cellbarcode=cb_tag, readname_barcode=readname_bc)
    print('Finished creating fragments file. Now sorting...')

    # sort
    sort_cmd = f'sort -k1,1 -k2,2n {out_unsorted} > {out_sorted}'
    os.system(sort_cmd)
    print('Finished sorting fragments')

    # remove unsorted
    os.remove(out_unsorted)

    # return path to sorted fragments file
    return out_sorted, temp_files


def _convert_gtf_to_bed(gtf, out=None, temp_files=[]):
    """
    Extract 'chr', 'start' and 'stop' from .gtf file and convert it to sorted BED file.
    BED file will be sorted by chromosome name and start position.

    :param gtf: str
        Path to .gtf file.
    :param out: str
        Path to save new BED file. If none, the file will be saved in the same folder as tha BAM file.
    :return: out_sorted: str
        Path to .bed file.
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
    sort_cmd = f'sort -k1,1 -k2,2n {out_unsorted} > {out_sorted}'
    os.system(sort_cmd)

    # remove unsorted
    os.remove(out_unsorted)

    # return the path to sorted bed
    return out_sorted, temp_files


def _overlap_two_beds(bed1, bed2, out=None, temp_files=[]):
    """
    Overlap two BED files using Bedtools Intersect.
    The result is a BED file containing regions in bed1 that overlaps with at least one region in bed2.

    :param bed1: str
        Path to first BED file.
    :param bed2: str
        Path to second BED file.
    :param out: str
        Path to save overlapped file. If none, the file will be saved in the same folder as tha BAM file.
    :return: out_overlap: str
        Path to new .bed file.
    """

    from pathlib import Path

    # get names of the bed files
    name_1 = Path(bed1).stem
    name_2 = Path(bed2).stem

    if not out:
        path = os.path.splitext(bed1)
        out_overlap = f"{path[0]}_{name_2}_overlap.bed"
    else:
        out_overlap = os.path.join(out, f'{name_1}_{name_2}_overlap.bed')

    temp_files.append(out_overlap)
    # a = pybedtools.BedTool(bed1)
    # b = pybedtools.BedTool(bed2)
    # a.intersect(b, u=True, sorted=True, output=out_overlap)

    bedtools = os.path.join('/'.join(sys.executable.split('/')[:-1]), 'bedtools')
    intersect_cmd = f'{bedtools} intersect -a {bed1} -b {bed2} -u -sorted > {out_overlap}'
    # run command
    os.system(intersect_cmd)

    # check if there is an overlap
    bed_file = pybedtools.BedTool(out_overlap)
    if len(bed_file) == 0:
        return False

    # return path to overlapped file
    return out_overlap, temp_files


def pct_fragments_in_promoters(adata, gtf_file=None, bam_file=None, fragments_file=None,
                               cb_col=None, cb_tag='CB', species=None, nproc=1, sort_bam=False):
    """
    A wrapper function for pct_fragments_overlap.
    This function calculates for each cell, the percentage of fragments in a BAM alignment file
    that overlap with a promoter region specified in a GTF file. The results are added to the anndata object
    as new columns (n_total_fragments, n_fragments_in_promoters and pct_fragments_in_promoters).

    :param adata: anndata.AnnData
        The anndata object containig cell barcodes in adata.obs.
    :param gtf_file: str
        Path to GTF file for promoters regions. if None, the GTF file in flatfiles directory will be used.
    :param bam_file: str
        Path to BAM file. If None, a fragments file must be provided in the parameter 'fragments_file'.
    :param fragments_file: str
        Path to fragments file. If None, a BAM file must be provided in the parameter 'bam_file'. The
        BAM file will be converted into fragments file.
    :param cb_col: str
        The column in adata.obs containing cell barcodes. If None, adata.obs.index will be used.
    :param cb_tag: str
        The tag where cell barcodes are saved in the bam file. Set to None if the barcodes are in read names.
     :param species: str
        Name of the species, will only be used if gtf_file is None to use internal GTF files.
        Species are {bos_taurus, caenorhabditis_elegans, canis_lupus_familiaris, danio_rerio, drosophila_melanogaster,
        gallus_gallus, homo_sapiens, mus_musculus, oryzias_latipes, rattus_norvegicus, sus_scrofa, xenopus_tropicalis}
    :param nproc: int
        Number of threads for parallelization. Will be used to convert BAM to fragments file.
    :param sort_bam: boolean
        Set to True if the provided BAM file is not sorted.
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


def pct_fragments_overlap(adata, regions_file, bam_file=None, fragments_file=None, cb_col=None,
                          cb_tag='CB', regions_name='list', nproc=1, sort_bam=False, sort_regions=False, keep_fragments=False):
    """
    This function calculates for each cell, the percentage of fragments in a BAM alignment file
    that overlap with regions specified in a BED or GTF file. The results are added to the anndata object
    as new columns.

    :param adata: anndata.AnnData
        The anndata object containig cell barcodes in adata.obs.
    :param regions_file: str
        Path to BED or GTF file containing regions of interest.
    :param bam_file: str
        Path to BAM file. If None, a fragments file must be provided in the parameter 'fragments_file'.
    :param fragments_file: str
        Path to fragments file. If None, a BAM file must be provided in the parameter 'bam_file'. The
        BAM file will be converted into fragments file.
    :param cb_col: str
        The column in adata.obs containing cell barcodes. If None, adata.obs.index will be used.
    :param cb_tag: str
        The tag where cell barcodes are saved in the bam file. Set to None if the barcodes are in read names.
    :param regions_name: int
        The name of the regions in the BED or GTF file (e.g. Exons). The name will be used as columns' name
        to be added to the anndata object (e.g. pct_fragments_in_{regions_name}). Defaults to 'list'.
    :param nproc: int
        Number of threads for parallelization. Will be used to convert BAM to fragments file.
    :param sort_bam: boolean
        Set to True if the provided BAM file is not sorted.
    """

    if not bam_file and not fragments_file:
        raise ValueError("Either BAM file or fragments file has to be provided!")

    # check for column in adata.obs where barcodes are
    if cb_col:
        try:
            barcodes = adata.obs[cb_col].to_list()
        except KeyError:
            print(f"{cb_col} is not in adata.obs!")
            return
    else:
        barcodes = adata.obs.index.to_list()

    temp_files = []
    # check if regions file is gtf or bed
    file_ext = Path(regions_file).suffix
    if file_ext.lower() == '.gtf':
        print("Converting GTF to BED...")
        # convert gtf to bed with columns chr, start, end
        bed_file, temp_files = _convert_gtf_to_bed(regions_file, temp_files=temp_files, out=None)
    elif file_ext.lower() == '.bed':
        bed_file = os.path.splitext(regions_file)[0] + '_sorted.bed'
        temp_files.append(bed_file)
        if sort_regions:
            sort_cmd = f'sort -k1,1 -k2,2n {regions_file} > {bed_file}'
            os.system(sort_cmd)

    # if only bam file is available -> convert to fragments
    if bam_file and not fragments_file:
        print('Converting BAM to fragments file! This may take a while...')
        fragments_file, temp_files = create_fragment_file(bam_file,
                                                          cb_tag=cb_tag,
                                                          out=None,
                                                          nproc=nproc,
                                                          sort_bam=sort_bam,
                                                          keep_temp=keep_fragments,
                                                          temp_files=temp_files)

    # overlap reads in fragments with promoter regions, return path to overlapped file
    print('Finding overlaps...')
    overlap_file, temp_files = _overlap_two_beds(fragments_file, bed_file, out=None, temp_files=temp_files)

    #
    mp_calc_pct = MPOverlapPct()
    mp_calc_pct.calc_pct(overlap_file, fragments_file, barcodes, adata, regions_name=regions_name, n_threads=8)

    #
    print('Adding results to adata object...')
    print("cleaning up...")
    for f in temp_files:
        os.remove(f)
    print('Done')


class MPOverlapPct():

    def __init__(self):

        self.merged_dict = None

    def calc_pct(self,
                 overlap_file,
                 fragments_file,
                 barcodes,
                 adata,
                 regions_name='list',
                 n_threads=8):

        # check if there was an overlap
        if not overlap_file:
            print("There was no overlap!")
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
        print('Calculating percentage...')
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

    def get_barcodes_sum(self, df, barcodes, col_name):
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

    def log_result(self, result):
        if self.merged_dict:
            self.merged_dict = dict(Counter(self.merged_dict) + Counter(result))
            # print('merging')
        else:
            self.merged_dict = result

    def mp_counter(self, fragments, barcodes, column, n_threads=8):

        pool = mp.Pool(n_threads, maxtasksperchild=48)
        jobs = []
        for chunk in fragments:
            job = pool.apply_async(self.get_barcodes_sum, args=(chunk, barcodes, column), callback=self.log_result)
            jobs.append(job)
        utils.monitor_jobs(jobs, description="Progress")
        pool.close()
        pool.join()
        # reset settings
        returns = self.merged_dict
        self.merged_dict = None

        return returns
