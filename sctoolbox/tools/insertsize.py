import os
import re
import pandas as pd
import gzip
import datetime

import sctoolbox.utils as utils
import sctoolbox.tools.bam
from sctoolbox._settings import settings
logger = settings.logger


# --------------------------------------------------------------------- #
# --------------------- Insertsize distribution ----------------------- #
# --------------------------------------------------------------------- #

def _check_in_list(element, alist):
    return element in alist


def _check_true(element, alist):  # true regardless of input
    return True


def add_insertsize(adata,
                   bam=None,
                   fragments=None,
                   barcode_col=None,
                   barcode_tag="CB",
                   regions=None):
    """
    Add information on insertsize to the adata object using either a .bam-file or a fragments file.
    Adds columns "insertsize_count" and "mean_insertsize" to adata.obs and a key "insertsize_distribution" to adata.uns containing the
    insertsize distribution as a pandas dataframe.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to add insertsize information to.
    bam : str, default None
        Path to bam file containing paired-end reads. If None, the fragments file is used instead.
    fragments : str, default None
        Path to fragments file containing fragments information. If None, the bam file is used instead.
    barcode_col : str, default None
        Column in adata.obs containing the name of the cell barcode. If barcode_col is None, it is assumed that the index of adata.obs contains the barcode.
    barcode_tag : str, default 'CB'
        Only for bamfiles: Tag used for the cell barcode for each read.
    regions : str, default None
        Only for bamfiles: A list of regions to obtain reads from, e.g. ['chr1:1-2000000']. If None, all reads in the .bam-file are used.

    Returns
    -------
    None - adata is adjusted in place.
    """

    adata_barcodes = adata.obs.index.tolist() if barcode_col is None else adata.obs[barcode_col].tolist()

    if bam is not None and fragments is not None:
        raise ValueError("Please provide either a bam file or a fragments file - not both.")

    elif bam is not None:
        table = _insertsize_from_bam(bam, barcode_tag=barcode_tag, regions=regions, barcodes=adata_barcodes)

    elif fragments is not None:
        table = _insertsize_from_fragments(fragments, barcodes=adata_barcodes)

    else:
        raise ValueError("Please provide either a bam file or a fragments file.")

    # Merge table to adata.obs and uns
    mean_table = table[["insertsize_count", "mean_insertsize"]]
    distribution_table = table[[c for c in table.columns if isinstance(c, int)]]

    distribution_barcodes = set(table.index)
    common = set(adata_barcodes).intersection(distribution_barcodes)
    missing = set(adata_barcodes) - distribution_barcodes

    if len(common) == 0:
        raise ValueError("No common barcodes")

    elif len(missing) > 0:
        logger.warning("not all barcodes in adata.obs were represented in the input fragments. The values for these barcodes are set to NaN.")
        missing_table = pd.DataFrame(index=list(missing), columns=distribution_table.columns)
        distribution_table = pd.concat([distribution_table, missing_table])

    # Merge table to adata.obs and uns
    if barcode_col is None:
        adata.obs = adata.obs.merge(mean_table, left_index=True, right_index=True, how="left")
    else:
        adata.obs = adata.obs.merge(mean_table, left_on=barcode_col, right_index=True, how="left")

    adata.uns["insertsize_distribution"] = distribution_table.loc[adata_barcodes]
    adata.uns['insertsize_distribution'].columns = adata.uns['insertsize_distribution'].columns.astype(str)  # ensures correct order of barcodes in table

    logger.info("Added insertsize information to adata.obs[[\"insertsize_count\", \"mean_insertsize\"]] and adata.uns[\"insertsize_distribution\"].")


def _insertsize_from_bam(bam,
                         barcode_tag="CB",
                         barcodes=None,
                         regions='chr1:1-2000000',
                         chunk_size=100000):
    """
    Get fragment insertsize distributions per barcode from bam file.

    Parameters
    -----------
    bam : str
        Path to bam file
    barcode_tag : str, default "CB"
        The read tag representing the barcode.
    barcodes : list, default None
        List of barcodes to include in the analysis. If None, all barcodes are included.
    regions : str or list of str, default 'chr1:1-2000000'
        Regions to include in the analysis. If None, all reads are included.
    chunk_size : int, default 500000
        Size of bp chunks to read from bam file.

    Returns
    --------
    pandas.DataFrame
        DataFrame with insertsize distributions per barcode.
    """

    # Load modules
    utils.check_module("pysam")
    import pysam

    if utils._is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    if isinstance(regions, str):
        regions = [regions]

    # Prepare function for checking against barcodes list
    if barcodes is not None:
        barcodes = set(barcodes)
        check_in = _check_in_list
    else:
        check_in = _check_true

    # Open bamfile
    logger.info("Opening bam file...")
    if not os.path.exists(bam + ".bai"):
        logger.warning("Bamfile has no index - trying to index with pysam...")
        pysam.index(bam)

    bam_obj = sctoolbox.tools.bam.open_bam(bam, "rb", require_index=True)
    chromosome_lengths = dict(zip(bam_obj.references, bam_obj.lengths))

    # Create chunked genome regions:
    logger.info(f"Creating chunks of size {chunk_size}bp...")

    if regions is None:
        regions = [f"{chrom}:0-{length}" for chrom, length in chromosome_lengths.items()]
    elif isinstance(regions, str):
        regions = [regions]

    # Create chunks from larger regions
    regions_split = []
    for region in regions:
        chromosome, start, end = re.split("[:-]", region)
        start = int(start)
        end = int(end)
        for chunk_start in range(start, end, chunk_size):
            chunk_end = chunk_start + chunk_size
            if chunk_end > end:
                chunk_end = end
            regions_split.append(f"{chromosome}:{chunk_start}-{chunk_end}")

    # Count insertsize per chunk using multiprocessing
    logger.info(f"Counting insertsizes across {len(regions_split)} chunks...")
    count_dict = {}
    read_count = 0
    pbar = tqdm(total=len(regions_split), desc="Progress: ", unit="chunks")
    for region in regions_split:
        chrom, start, end = re.split("[:-]", region)
        for read in bam_obj.fetch(chrom, int(start), int(end)):
            read_count += 1
            try:
                barcode = read.get_tag(barcode_tag)
            except Exception:  # tag was not found
                barcode = "NA"

            # Add read to dict
            if check_in(barcode, barcodes) is True:
                size = abs(read.template_length) - 9  # length of insertion
                count_dict = _add_fragment(count_dict, barcode, size)

        # Update progress
        pbar.update(1)
    pbar.close()  # close progress bar

    bam_obj.close()

    # Check if any reads were read at all
    if read_count == 0:
        raise ValueError("No reads found in bam file. Please check bamfile or adjust the 'regions' parameter to include more regions.")

    # Check if any barcodes were found
    if len(count_dict) == 0 and barcodes is not None:
        raise ValueError("No reads found in bam file for the barcodes given in 'barcodes'. Please adjust the 'barcodes' or 'barcode_tag' parameters.")

    # Convert dict to pandas dataframes
    logger.info("Converting counts to dataframe")
    table = pd.DataFrame.from_dict(count_dict, orient="index")
    table = table[["insertsize_count", "mean_insertsize"] + sorted(table.columns[2:])]
    table["mean_insertsize"] = table["mean_insertsize"].round(2)

    logger.info("Done getting insertsizes from bam!")

    return table


def _insertsize_from_fragments(fragments, barcodes=None):
    """
    Get fragment insertsize distributions per barcode from fragments file.

    Parameters
    -----------
    fragments : str
        Path to fragments.bed(.gz) file.
    barcodes : list of str, default None
        Only collect fragment sizes for the barcodes in barcodes

    Returns
    --------
    pandas.DataFrame
        DataFrame with insertsize distributions per barcode.
    """

    # Open fragments file
    if utils._is_gz_file(fragments):
        f = gzip.open(fragments, "rt")
    else:
        f = open(fragments, "r")

    # Prepare function for checking against barcodes list
    if barcodes is not None:
        barcodes = set(barcodes)
        check_in = _check_in_list
    else:
        check_in = _check_true

    # Read fragments file and add to dict
    logger.info("Counting fragment lengths from fragments file...")
    start_time = datetime.datetime.now()
    count_dict = {}
    for line in f:
        columns = line.rstrip().split("\t")
        start = int(columns[1])
        end = int(columns[2])
        barcode = columns[3]
        count = int(columns[4])
        size = end - start - 9  # length of insertion (-9 due to to shifted cutting of Tn5)

        # Only add fragment if check is true
        if check_in(barcode, barcodes) is True:
            count_dict = _add_fragment(count_dict, barcode, size, count)

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    f.close()
    logger.info("Done reading file - elapsed time: {0}".format(str(elapsed).split(".")[0]))

    # Convert dict to pandas dataframe
    logger.info("Converting counts to dataframe...")
    table = pd.DataFrame.from_dict(count_dict, orient="index")
    table = table[["insertsize_count", "mean_insertsize"] + sorted(table.columns[2:])]
    table["mean_insertsize"] = table["mean_insertsize"].round(2)

    logger.info("Done getting insertsizes from fragments!")

    return table


def _add_fragment(count_dict, barcode, size, count=1):
    """
    Add fragment of size 'size' to count_dict.

    Parameters
    -----------
    count_dict : dict
        Dictionary containing the counts per insertsize.
    barcode : str
        Barcode of the read.
    size : int
        Insertsize to add to count_dict.
    count : int, default 1
        Number of reads to add to count_dict.

    Returns
    --------
    count_dict : dict
        Updated count_dict
    """

    # Initialize if barcode is seen for the first time
    if barcode not in count_dict:
        count_dict[barcode] = {"mean_insertsize": 0, "insertsize_count": 0}

    # Add read to dict
    if size >= 0 and size <= 1000:  # do not save negative insertsize, and set a cap on the maximum insertsize to limit outlier effects

        count_dict[barcode]["insertsize_count"] += count

        # Update mean
        mu = count_dict[barcode]["mean_insertsize"]
        total_count = count_dict[barcode]["insertsize_count"]
        diff = (size - mu) / total_count
        count_dict[barcode]["mean_insertsize"] = mu + diff

        # Save to distribution
        if size not in count_dict[barcode]:  # first time size is seen
            count_dict[barcode][size] = 0
        count_dict[barcode][size] += count

    return count_dict
