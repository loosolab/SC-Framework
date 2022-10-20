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



def check_pct_reads_in_promoters(adata):
    """
    Check if percentage of reads in promoters is in adata.obs.
    :param adata: AnnData
        AnnData object
    :return boolean
        True if pct_reads_in_promoters column is in adata.obs, otherwise False.
    """

    if 'pct_reads_in_promoters' in adata.obs.columns:
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
        
    sinto = os.path.join('/'.join(sys.executable.split('/')[:-1]),'sinto')
    create_cmd = f'''{sinto} fragments -b {bam} -p {nproc} -f {out} --barcode_regex "[^:]*"'''

    
    # execute command
    os.system(create_cmd)
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
    
    bedtools = os.path.join('/'.join(sys.executable.split('/')[:-1]),'bedtools')
    intersect_cmd = f'{bedtools} intersect -a {bed1} -b {bed2} -u -sorted > {out_overlap}'
    
    # run command
    os.system(intersect_cmd)
    
    # return path to overlapped file
    return out_overlap


def pct_reads_in_promoters(adata, gtf_file, bam_file=None, fragments_file=None, cb_col=None, nproc=1):
    """
    This function calculates for each cell, the percentage of reads in a BAM alignment file 
    that overlap with a promoter region specified in a GTF file. The results are added to the anndata object
    as a new column 'pct_reads_in_promoters'. 
    
    :param adata: AnnData
        The anndata object containig cell barcodes in adata.obs.
    :param gtf_file: str
        Path to GTF file for promoters regions.
    :param bam_file: str
        Path to BAM file. If None, a fragments file must be provided in the parameter 'fragments_file'.
    :param fragments_file: str
        Path to fragments file. If None, a BAM file must be provided in the parameter 'bam_file'. The
        BAM file will be converted into fragments file.
    :param cb_col: str
        The column in adata.obs containing cell barcodes. If None, adata.obs.index will be used.
    :param nproc: int
        Number of threads for parallelization. Will be used to convert BAM to fragments file.
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
        
    # if only bam file is available -> convert to fragments
    if bam_file and not fragments_file:
        print('Converting BAM to fragments file! This may take a while...')
        fragments_file = create_fragment_file(bam_file, nproc=nproc, out=None)
        
    # convert gtf to bed and get the path string
    print('Converting GTF to sorted BED...')
    promoters_bed_file = _convert_gtf_to_bed(gtf_file, out=None)

    # overlap reads in fragments with promoter regions, return path to overlapped file
    print('Finding overlaps...')
    overlap_file = _overlap_two_beds(fragments_file, promoters_bed_file, out=None)
    
    # get unique barcodes from adata.obs
    barcodes = set(barcodes)
    
    print('Calculating percentage...')
    # read overlap file as dataframe
    df_overlap = pd.read_csv(overlap_file, sep='\t', header=None)
    df_overlap.columns=['chr','start','end','barcode','n_reads_in_promoter']
    # remove barcodes not found in adata.obs
    df_overlap = df_overlap.loc[df_overlap['barcode'].isin(barcodes)]
    # drop chr start end columns
    df_overlap.drop(['chr','start','end'], axis=1, inplace=True)
    # get the sum of reads counts in each cell barcode
    df_overlap = df_overlap.groupby('barcode').sum()
    # convert dataframe to dictionary
    promoters_count = df_overlap['n_reads_in_promoter'].to_dict()
    
    
    # read fragments file as dataframe
    fragments_df = pd.read_csv(fragments_file, sep='\t', header=None)
    # rename columns, remove barcodes not in adata.obs, drop unwanted columns and sum read counts for each cell
    fragments_df.columns=['chr','start','end','barcode','n_total_reads']
    fragments_df = fragments_df.loc[fragments_df['barcode'].isin(barcodes)]
    fragments_df.drop(['chr','start','end'], axis=1, inplace=True)
    fragments_df = fragments_df.groupby('barcode').sum()
    # add column for reads in promoters from promoters_count dict
    fragments_df['n_reads_in_promoters'] = fragments_df.index.map(promoters_count).fillna(0)
    # calculate percentage
    fragments_df['pct_reads_in_promoters'] = fragments_df['n_reads_in_promoters'] / fragments_df['n_total_reads']
    
    print('Adding results to adata object...')
    # add results to adata.obs
    if cb_col:
        adata.obs['pct_reads_in_promoters'] = adata.obs[cb_col].map(fragments_df['pct_reads_in_promoters'].to_dict())
    else:
        adata.obs['pct_reads_in_promoters'] = adata.obs.index.map(fragments_df['pct_reads_in_promoters'].to_dict())
        
    print('Done')


def pct_reads_overlap(adata, bed_file, bam_file=None, fragments_file=None, cb_col=None, nproc=1, 
                      col_added='pct_reads_overlap'):
    """
    This function calculates for each cell, the percentage of reads in a BAM alignment file 
    that overlap with regions specified in a BED file. The results are added to the anndata object
    as a new column. 
    
    :param adata: AnnData
        The anndata object containig cell barcodes in adata.obs.
    :param bed_file: str
        Path to BED file containing regions of interest.
    :param bam_file: str
        Path to BAM file. If None, a fragments file must be provided in the parameter 'fragments_file'.
    :param fragments_file: str
        Path to fragments file. If None, a BAM file must be provided in the parameter 'bam_file'. The
        BAM file will be converted into fragments file.
    :param cb_col: str
        The column in adata.obs containing cell barcodes. If None, adata.obs.index will be used.
    :param nproc: int
        Number of threads for parallelization. Will be used to convert BAM to fragments file.
    :param col_added: int
        The name of the column to be added to the anndata object. Defaults to 'pct_reads_overlap'
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
        
    # if only bam file is available -> convert to fragments
    if bam_file and not fragments_file:
        print('Converting BAM to fragments file! This may take a while...')
        fragments_file = create_fragment_file(bam_file, nproc=nproc, out=None)

    # overlap reads in fragments with promoter regions, return path to overlapped file
    print('Finding overlaps...')
    overlap_file = _overlap_two_beds(fragments_file, bed_file, out=None)
    
    # get unique barcodes from adata.obs
    barcodes = set(barcodes)
    
    print('Calculating percentage...')
    # read overlap file as dataframe
    df_overlap = pd.read_csv(overlap_file, sep='\t', header=None)
    df_overlap.columns=['chr','start','end','barcode','n_reads_in_promoter']
    # remove barcodes not found in adata.obs
    df_overlap = df_overlap.loc[df_overlap['barcode'].isin(barcodes)]
    # drop chr start end columns
    df_overlap.drop(['chr','start','end'], axis=1, inplace=True)
    # get the sum of reads counts in each cell barcode
    df_overlap = df_overlap.groupby('barcode').sum()
    # convert dataframe to dictionary
    promoters_count = df_overlap['n_reads_in_promoter'].to_dict()
    
    
    # read fragments file as dataframe
    fragments_df = pd.read_csv(fragments_file, sep='\t', header=None)
    # rename columns, remove barcodes not in adata.obs, drop unwanted columns and sum read counts for each cell
    fragments_df.columns=['chr','start','end','barcode','n_total_reads']
    fragments_df = fragments_df.loc[fragments_df['barcode'].isin(barcodes)]
    fragments_df.drop(['chr','start','end'], axis=1, inplace=True)
    fragments_df = fragments_df.groupby('barcode').sum()
    # add column for reads in promoters from promoters_count dict
    fragments_df['n_reads_in_promoters'] = fragments_df.index.map(promoters_count).fillna(0)
    # calculate percentage
    fragments_df['pct_reads_in_promoters'] = fragments_df['n_reads_in_promoters'] / fragments_df['n_total_reads']
    
    print('Adding results to adata object...')
    # add results to adata.obs
    if cb_col:
        adata.obs['pct_reads_in_promoters'] = adata.obs[cb_col].map(fragments_df['pct_reads_in_promoters'].to_dict())
    else:
        adata.obs['pct_reads_in_promoters'] = adata.obs.index.map(fragments_df['pct_reads_in_promoters'].to_dict())
    
    print('Done')

if __name__ == '__main__':
    # test
    import scanpy as sc

    fragments_file = '/home/jan/python-workspace/sc-atac/preprocessing/data/bamfiles/fragments_cropped_146.bed'
    promoters_gtf = '/home/jan/python-workspace/sc-atac/preprocessing/data/homo_sapiens.104.promoters2000.gtf'

    adata = sc.read_h5ad('/home/jan/python-workspace/sc-atac/preprocessing/data/anndata/cropped_146.h5ad')
    adata.obs = adata.obs.set_index('barcode')
    pct_reads_overlap(adata, promoters_gtf,fragments_file=fragments_file, nproc=1)