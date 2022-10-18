import math
from collections import defaultdict

import pandas as pd
import episcanpy as epi
import pysam
import multiprocessing as mp
import sctoolbox.bam

def mean_fragment_length(bam_obj):
    """
    Calculate the mean fragment length of barcodes in a bam file

    :param bam:
    :return:
    """
    mean_fragment_lengths = {}
    q = None
    previous_barcode = None
    for read in bam_obj:
        l = read.template_length
        l = math.sqrt(l**2)
        barcode = read.qname.split(":")[0].upper()
        if barcode != previous_barcode:
            if q is not None:
                mean_fragment_lengths[previous_barcode] = sum(fragment_lengths) / len(fragment_lengths)
            fragment_lengths = []
            fragment_lengths.append(l)
            previous_barcode = barcode
        else:
            if q != read.query_name:
                fragment_lengths.append(l)

        q = read.query_name

    mean_fragment_lengths[previous_barcode] = sum(fragment_lengths) / len(fragment_lengths)
    mean_fragment_lengths_df = pd.DataFrame.from_dict(mean_fragment_lengths, orient="index")

    return mean_fragment_lengths_df

def mean_fragment_length_fragment_file(fragment_file):
    """
    Calculate the mean fragment length of barcodes in a fragment file

    :param bam:
    :return:
    """
    fragments = pd.read_csv(fragment_file, sep="\t", header=None, chunksize=10000)
    mean_fl_df = None

    for chunk in fragments:

        fl_df = None

        for index, fragment in chunk.iterrows():
            start = fragment[1]
            stop = fragment[2]
            barcode = fragment[3]
            l = stop - start
            length = math.sqrt(l**2)

            if fl_df is None:
                fl_df = pd.DataFrame({'barcode': [barcode], 'length': [length], 'count': [0]}).set_index('barcode')
            else:
                if barcode in fl_df.index:
                    fl_df.loc[barcode, 'length'] += length
                    fl_df.loc[barcode, 'count'] += 1
                else:
                    fl_df.loc[barcode] = [length, 1]

        fl_df['mean'] = fl_df['length'] / fl_df['count']

        if mean_fl_df is None:
            mean_fl_df = fl_df
        else:
            mean_fl_df = mean_fl_df.add(fl_df, fill_value=0)

    return mean_fl_df

def mean_fl_multi_thread(fragment_file, n_threads=4):
    """
    Calculate the mean fragment length of barcodes in a fragment file (multi-threaded)
    :param fragment_file:
    :param n_threads:
    :return:
    """
    fragments = pd.read_csv(fragment_file, sep="\t", header=None, chunksize=10000)
    mean_fl_df = None
    # init pool
    pool = mp.Pool(n_threads)
    jobs = []
    # loop over chunks
    for chunk in fragments:
        fl_df = None
        job = pool.apply_async(calc_chunk, args=(chunk, fl_df))
        jobs.append(job)
    pool.close()

    #collect results
    for job in jobs:
        fl_df = job.get()
        if mean_fl_df is None:
            mean_fl_df = fl_df
        else:
            mean_fl_df = mean_fl_df.add(fl_df, fill_value=0)


    return mean_fl_df

def calc_chunk(chunk, fl_df):
    '''
    Calculate the mean fragment length of barcodes in a fragment file (worker function)
    :param chunk:
    :param fl_df:
    :return:
    '''
    for index, fragment in chunk.iterrows():
        start = fragment[1]
        stop = fragment[2]
        barcode = fragment[3]
        l = stop - start
        length = math.sqrt(l ** 2)

        if fl_df is None:
            fl_df = pd.DataFrame({'barcode': [barcode], 'length': [length], 'count': [0]}).set_index('barcode')
        else:
            if barcode in fl_df.index:
                fl_df.loc[barcode, 'length'] += length
                fl_df.loc[barcode, 'count'] += 1
            else:
                fl_df.loc[barcode] = [length, 1]

    return fl_df


def add_mfl_bam(bam, adata):
    """
    Merge mean fragment lengths with adata.obs
    """
    # load bam file
    bam_obj = sctoolbox.bam.open_bam(bam, "rb", verbosity=0)
    # compute mean fragment lengths
    mean_fragment_lengths = mean_fragment_length(bam_obj)
    # set barcode as index
    adata.obs = adata.obs.set_index("barcode")
    # Merge with adata.obs
    mean_fragment_lengths.columns=['mean']
    adata.obs = adata.obs.merge(mean_fragment_lengths, left_index=True, right_index=True, how="left")
    adata.obs.rename(columns={'mean': 'mean_fragment_length'}, inplace=True)

    return adata

def add_mfl_fragment(fragment, adata):
    """
    Merge mean fragment lengths with adata.obs
    """
    # compute mean fragment lengths
    mean_fragment_lengths = mean_fl_multi_thread(fragment, n_threads=4)
    # set barcode as index
    adata.obs = adata.obs.set_index("barcode")
    # Merge with adata.obs
    mean_fragment_lengths.columns=['mean']
    adata.obs = adata.obs.merge(mean_fragment_lengths, left_index=True, right_index=True, how="left")
    adata.obs.rename(columns={'mean': 'mean_fragment_length'}, inplace=True)

    return adata

if __name__ == "__main__":

    import time
    bam = "/home/jan/python-workspace/sc-atac/preprocessing/data/bamfiles/cropped_146.bam"
    adata = epi.read("/home/jan/python-workspace/sc-atac/preprocessing/data/anndata/cropped_146.h5ad")
    fragment_file = "/home/jan/python-workspace/sc-atac/preprocessing/data/bamfiles/fragments.bed"

    start = time.time()
    mean_fl = mean_fragment_length_fragment_file(fragment_file)
    stop = time.time()
    print("Benchmark single core: ", stop - start)

    start = time.time()
    mean_fl = mean_fl_multi_thread(fragment_file, 8)
    stop = time.time()
    print("Benchmark multi core: ", stop - start)

    #adata = add_mean_fragment_length(bam, adata)

# write adata to h5ad
    #adata.write("/home/jan/python-workspace/sc-atac/preprocessing/data/anndata/cropped_146_fl.h5ad")

#for read1, read2 in read_pair_generator(bam_obj):
#    print(read1, read2)