"""
    Module to calculate percentage of reads from a BAM or fragments file that overlap promoter regions specified 
    in a GTF file using 'pct_reads_in_promoters' function. The function 'pct_reads_overlap' calculates percentage of
    reads that overlap with regions specified in any BED file. The BED file must have three columns ['chr','start','end']
    as first columns.
"""


import sctoolbox
import sctoolbox.utilities as utils
import os
import sys 
import pandas as pd
import anndata as ad
import pybedtools
from pathlib import Path
from sinto.fragments import fragments


def check_pct_fragments_in_promoters(adata):
    """
    Check if percentage of reads in promoters is in adata.obs.
    :param adata: AnnData
        AnnData object
    :return boolean
        True if pct_reads_in_promoters column is in adata.obs, otherwise False.
    """

    if 'pct_fragments_in_promoters' in adata.obs.columns:
        return True
    else:
        return False


def create_fragment_file(bam, nproc=1, out=None):
    """
    Create fragments file out of a BAM file using the package sinto

    :param bam: str
        Path to .bam file.
    :param nproc: int
        Number of threads for parallelization.
    :param out: str
        Path to save fragments file. If none, the file will be saved in the same folder as tha BAM file.
    
    :return: str
        Path to fragments file.
    """

    if not out:
        path = os.path.splitext(bam)
        out = f"{path[0]}_fragments.bed"
        out_sorted = f"{path[0]}_fragments_sorted.bed"
        
    #sinto = os.path.join('/'.join(sys.executable.split('/')[:-1]),'sinto')
    #create_cmd = f'''{sinto} fragments -b {bam} -p {nproc} -f {out} --barcode_regex "[^:]*"'''

    fragments(bam, out, readname_barcode="[^:]*")
    
    # execute command
    #os.system(create_cmd)
    print('Finished creating fragments file. Now sorting...')

    # sort
    sort_cmd = f'sort -k1,1 -k2,2n {out} > {out_sorted}'
    os.system(sort_cmd)
    print('Finished sorting fragments')
    
    # remove unsorted
    os.remove(out)
    
    # return path to sorted fragments file
    return out_sorted


def _convert_gtf_to_bed(gtf, out=None):
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
    
    import csv
    
    if not out:
        path = os.path.splitext(gtf)
        out_unsorted = f"{path[0]}_tmp.bed"
        out_sorted = f"{path[0]}.gtf_sorted.bed"
    else:
        name = os.path.basename(gtf)
        out_unsorted = os.path.join(out, f'{name}_tmp.bed')
        out_sorted = os.path.join(out, f'{name}_sorted.bed')
        
    with open(gtf, 'rb') as file:
        with open(out_unsorted, "w") as out_file:
            for row in file:
                row = row.decode("utf-8")  # convert binary to string
                row = row.split('\t')
                line = row[0] + '\t' + row[3] + '\t' + row[4] + '\n'
                out_file.write(line)
    # sort gtf
    sort_cmd = f'sort -k1,1 -k2,2n {out_unsorted} > {out_sorted}'
    os.system(sort_cmd)
    
    # remove unsorted
    os.remove(out_unsorted)
    
    # return the path to sorted bed
    return out_sorted


def _overlap_two_beds(bed1, bed2, out=None):
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
        out_overlap = os.path.join(out, f'{name_1}_{name_2}.bed')
    
    a = pybedtools.BedTool(bed1)
    b = pybedtools.BedTool(bed2)
    
    overlap = a.intersect(b, u=True, sorted=True, output=out_overlap)

    #bedtools = os.path.join('/'.join(sys.executable.split('/')[:-1]),'bedtools')
    #intersect_cmd = f'{bedtools} intersect -a {bed1} -b {bed2} -u -sorted > {out_overlap}'
    
    # run command
    #os.system(intersect_cmd)
    
    # return path to overlapped file
    return out_overlap


def pct_fragments_in_promoters(adata, gtf_file=None, bam_file=None, fragments_file=None, 
                               cb_col=None, species=None, nproc=1):
    """
    A wrapper function for pct_fragments_overlap.
    This function calculates for each cell, the percentage of fragments in a BAM alignment file 
    that overlap with a promoter region specified in a GTF file. The results are added to the anndata object
    as new columns (n_total_fragments, n_fragments_in_promoters and pct_fragments_in_promoters). 
    
    :param adata: AnnData
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
     :param species: str
        Name of the species, will only be used if gtf_file is None to use internal GTF files.
        Species are {bos_taurus, caenorhabditis_elegans, canis_lupus_familiaris, danio_rerio, drosophila_melanogaster, 
        gallus_gallus, homo_sapiens, mus_musculus, oryzias_latipes, rattus_norvegicus, sus_scrofa, xenopus_tropicalis}
    :param nproc: int
        Number of threads for parallelization. Will be used to convert BAM to fragments file.
    """
    # exit if no gtf file and no species 
    if not gtf_file and not species:
        print('Please provide a GTF file or specify a species!')
        return
    if not gtf_file:
        promoters_gtf = pkg_resources.resource_filename("sctoolbox", f"data/promoters_gtf/{species}.104.promoters2000.gtf")
    else:
        promoters_gtf = gtf_file
    
    # call function
    pct_fragments_overlap(adata, regions_file=promoters_gtf, bam_file=bam_file, fragments_file=fragments_file, 
                          cb_col=cb_col, nproc=nproc, regions_name='promoters')


def pct_fragments_overlap(adata, regions_file, bam_file=None, fragments_file=None, cb_col=None, nproc=1, regions_name='list'):
    """
    This function calculates for each cell, the percentage of fragments in a BAM alignment file 
    that overlap with regions specified in a BED or GTF file. The results are added to the anndata object
    as new columns. 
    
    :param adata: AnnData
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
    :param nproc: int
        Number of threads for parallelization. Will be used to convert BAM to fragments file.
    :param regions_name: int
        The name of the regions in the BED or GTF file (e.g. Exons). The name will be used as columns' name 
        to be added to the anndata object (e.g. pct_fragments_in_{regions_name}). Defaults to 'list'
    """
    
    if not bam_file and not fragments_file:
        print("Either BAM file or fragments file has to be given!")
        return
    
    # check for column in adata.obs where barcodes are
    if cb_col:
        try:
            barcodes = list(adata.obs[cb_col])
        except KeyError:
            print(f"{cb_col} is not in adata.obs!")
            return 
    else:
        barcodes = list(adata.obs.index)
        
    # check if regions file is gtf or bed
    file_ext = Path(regions_file).suffix
    if file_ext.lower() == '.gtf':
        print("Converting GTF to BED...")
        # convert gtf to bed with columns chr, start, end
        bed_file = _convert_gtf_to_bed(regions_file, out=None)
    elif file_ext.lower() == '.bed':
        bed_file = regions_file
        
    # if only bam file is available -> convert to fragments
    if bam_file and not fragments_file:
        print('Converting BAM to fragments file! This may take a while...')
        fragments_file = create_fragment_file(bam_file, nproc=nproc, out=None)

    # overlap reads in fragments with promoter regions, return path to overlapped file
    print('Finding overlaps...')
    overlap_file = _overlap_two_beds(fragments_file, bed_file, out=None)
    
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
    
    ### calculating percentage ###
    print('Calculating percentage...')
    # read overlap file as dataframe
    df_overlap = pd.read_csv(overlap_file, sep='\t', header=None)
    df_overlap.columns=['chr','start','end','barcode', col_n_fragments_in_list]
    # remove barcodes not found in adata.obs
    df_overlap = df_overlap.loc[df_overlap['barcode'].isin(barcodes)]
    # drop chr start end columns
    df_overlap.drop(['chr','start','end'], axis=1, inplace=True)
    # get the sum of reads counts in each cell barcode
    df_overlap = df_overlap.groupby('barcode').sum()
    # convert dataframe to dictionary
    overlap_count = df_overlap[col_n_fragments_in_list].to_dict()


    # read fragments file as dataframe
    fragments_df = pd.read_csv(fragments_file, sep='\t', header=None)
    # rename columns, remove barcodes not in adata.obs, drop unwanted columns and sum read counts for each cell
    
    fragments_df.columns=['chr','start','end','barcode', col_total_fragments]
    fragments_df = fragments_df.loc[fragments_df['barcode'].isin(barcodes)]
    fragments_df.drop(['chr','start','end'], axis=1, inplace=True)
    fragments_df = fragments_df.groupby('barcode').sum()
    # add column for reads in promoters from promoters_count dict
    fragments_df[col_n_fragments_in_list] = fragments_df.index.map(overlap_count).fillna(0)
    # calculate percentage
    fragments_df[col_pct_fragments] = fragments_df[col_n_fragments_in_list] / fragments_df[col_total_fragments]

    print('Adding results to adata object...')
    # add results to adata.obs
    if cb_col:
        adata.obs = adata.obs.merge(fragments_df, left_on=cb_col, right_index=True, how='inner')
    else:
        adata.obs = adata.obs.merge(fragments_df, how='inner', left_index=True, right_index=True)

    print('Done')
