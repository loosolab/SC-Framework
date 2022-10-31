import pandas as pd
import numpy as np
import copy
import os
import argparse
import glob
import multiprocessing as mp
import re

import sctoolbox.utilities as utils
import psutil
import subprocess
import gzip
import shutil
import scanpy as sc
import sctoolbox.creators as cr
import warnings


def add_cellxgene_annotation(adata, csv):
    """
    Add columns from cellxgene annotation to the adata .obs table.

    Parameters
    ----------
    adata : anndata.AnnData
        Adata object to add annotations to.
    csv : str
        Path to the annotation file from cellxgene containing cell annotation.

    Returns
    --------
    None - the annotation is added to adata in place.
    """
    anno_table = pd.read_csv(csv, sep=",", comment='#')
    anno_table.set_index("index", inplace=True)
    anno_name = anno_table.columns[-1]
    adata.obs.loc[anno_table.index, anno_name] = anno_table[anno_name].astype('category')


#################################################################################
# ------------------------ Uropa annotation of peaks -------------------------- #
#################################################################################

def gtf_integrity(gtf):
    '''
    Checks the integrity of a gtf file by examining:
        - file-ending
        - header ##format: gtf
        - number of columns == 9
        - regex pattern of column 9 matches gtf specific format

    :param gtf: str
        Path to .gtf file containing genomic elements for annotation.
    :return: boolean
        True if the file passed all tests
    '''

    regex_header = '#+.*'
    regex_format_column = '#+format: gtf.*'  # comment can start with one or more '#'

    # Initialize parameters
    first_entry = []
    header = False
    format_gtf = False

    if _is_gz_file(gtf):
        fp = gzip.open(gtf, 'rt')  # read text (rt) mode
    else:
        fp = open(gtf)

    # Check header (if present) and first entry
    for line in fp:
        if re.match(regex_header, line):
            header = True
            if re.match(regex_format_column, line):
                # Check if format information in the header matches gtf
                format_gtf = True
        else:
            first_entry = line.split(sep='\t')
            break

    fp.close()  # done reading from file

    # Check if number of columns matches 9
    if len(first_entry) != 9:
        raise argparse.ArgumentTypeError('Number of columns in the gtf file unequal 9')

    # If there was a header, check that the header matches gtf
    if header and not format_gtf:
        raise argparse.ArgumentTypeError('Header in gtf file does not match gtf format')

    # Extract gene_id information from column 9 and check if the format is gtf (not gff)
    column_9 = first_entry[8]
    column_9_split = column_9.split(sep=';')
    gene_id = column_9_split[0]  # TODO; it cannot always be assumed that the gene_id is the first entry in column 9
    # gtf specific format of the gene_id column (gff3: gene_id="xxxxx"; gtf: gene_id "xxxxx")
    regex_gene_id = 'gene_id ".*"'

    # check match of the pattern
    if not re.match(regex_gene_id, gene_id):
        raise argparse.ArgumentTypeError('gtf file is corrupted')

    return True  # the function only returns of no error is raised


def _is_gz_file(filepath):
    with open(filepath, 'rb') as test_f:
        return test_f.read(2) == b'\x1f\x8b'


def gunzip_file(f_in, f_out):
    with gzip.open(f_in, 'rb') as h_in:
        with open(f_out, 'wb') as h_out:
            shutil.copyfileobj(h_in, h_out)


def rm_tmp(temp_dir, tempfiles=None):
    """
    1. Running with tempfiles list:
    Removing temporary directory by previously removing temporary files from the tempfiles list.
    If the temporary directory is not empty it will not be removed.
    2. Running without tempfiles list:
    All gtf related files will be removed automatically no list of them required.
    The directory is then removed afterwards.

    :param temp_dir: str
        Path to the temporary directory.
    :param tempfiles: list of str
        Paths to files to be deleted before removing the temp directory.
    :return: None
    """
    try:
        if tempfiles is None:
            for f in glob.glob(temp_dir + "/*gtf*"):
                os.remove(f)
        else:
            for f in tempfiles:
                os.remove(f)

        os.rmdir(temp_dir)

    except OSError as error:
        print(error)


def validate_regions(adata, coordinate_columns):
    """ Checks if the regions in adata.var are valid.

    Parameters
    -----------
    adata : AnnData
        AnnData object containing the regions to be checked.
    coordinate_columns : list of str
        List of length 3 for column names in adata.var containing chr, start, end coordinates. """

    # Test whether the first three columns are in the right format
    chr, start, end = coordinate_columns

    # Test if coordinate columns are in adata.var
    utils.check_columns(adata.var, coordinate_columns, "adata.var")

    # Test whether the first three columns are in the right format
    for _, line in adata.var.to_dict(orient="index").items():
        valid = False

        if isinstance(line[chr], str) and isinstance(line[start], int) and isinstance(line[end], int):
            if line[start] <= line[end]:  # start must be smaller than end
                valid = True  # if all tests passed, the line is valid

        if valid is False:
            raise ValueError("The region {0}:{1}-{2} is not a valid genome region. Please check the format of columns: {3}".format(line[chr], line[start], line[end], coordinate_columns))


def format_adata_var(adata,
                     coordinate_columns=None,
                     columns_added=["chr", "start", "end"]):
    '''
    Formats the index of adata.var and adds peak_chr, peak_start, peak_end columns to adata.var if needed.
    If coordinate_columns are given, the function will check if these columns already contain the information needed. If the coordinate_columns are in the correct format, nothing will be done.
    If the coordinate_columns are invalid (or coordinate_columns is not given) the index is checked for the following format:
    "*[_:-]start[_:-]stop"

    If the index can be formatted, the formatted columns (columns_added) will be added.
    If the index cannot be formatted, an error will be raised.

    :param adata: AnnData
        The anndata object containing features to annotate.
    :param coordinate_columns: list of str or None
        List of length 3 for column names in adata.var containing chr, start, end coordinates to check.
        If None, the index will be formatted.
    :param columns_added: list of str
        List of length 3 for column names in adata.var containing chr, start, end coordinates to add.
    '''

    # Test whether the first three columns are in the right format
    format_index = True
    if coordinate_columns is not None:
        try:
            validate_regions(adata, coordinate_columns)
            format_index = False
        except KeyError:
            print("The coordinate columns are not found in adata.var. Trying to format the index.")
        except ValueError:
            print("The regions in adata.var are not in the correct format. Trying to format the index.")

    # Format index if needed
    if format_index:
        print("formatting adata.var index to coordinate columns:")
        regex = r'[^_:\-]+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+'  # matches chr_start_end / chr-start-end / chr:start-end and variations

        # Prepare lists to insert
        peak_chr_list = []
        peak_start_list = []
        peak_end_list = []

        names = adata.var.index
        for name in names:
            if re.match(regex, name):  # test if name can be split by regex

                # split the name into chr, start, end
                split_name = re.split(r'[\_\:\-]', name)
                peak_chr_list.append(split_name[0])
                peak_start_list.append(int(split_name[1]))
                peak_end_list.append(int(split_name[2]))

            else:
                raise ValueError("Index does not match the format *_start_stop or *:start-stop. Please check your index.")

        adata.var.drop(columns_added, axis=1,
                       errors='ignore', inplace=True)

        adata.var.insert(0, columns_added[2], peak_end_list)
        adata.var.insert(0, columns_added[1], peak_start_list)
        adata.var.insert(0, columns_added[0], peak_chr_list)

        # Check whether the newly added columns are in the right format
        validate_regions(adata, columns_added)


def annotate_adata(adata,
                   gtf,
                   config=None,
                   best=True,
                   threads=1,
                   coordinate_cols=None,
                   temp_dir="",
                   remove_temp=True,
                   verbose=True,
                   inplace=True):

    """
    Annotate adata .var features with genes from .gtf using UROPA [1]_. The function assumes that the adata.var contains genomic coordinates in the first
    three columns, e.g. "chromosome", "start", "end". If specific columns should be used, please adjust the names via 'coordinate_cols'.

    Parameters
    ----------
    adata : anndata.AnnData
        The anndata object containing features to annotate.
    gtf : str
        Path to .gtf file containing genomic elements for annotation.
    config : dict, optional
        A dictionary indicating how regions should be annotated. Default is to annotate feature 'gene' within -10000;1000bp of the gene start. See 'Examples' of how to set up a custom configuration dictionary.
    best : boolean
        Whether to return the best annotation or all valid annotations. Default: True (only best are kept).
    coordinate_cols : list of str, optional
        A list of column names in the regions DataFrame that contain the chromosome, start and end coordinates. Default: None (the first three columns are taken).
    threads : int, optional
        Number of threads to use for multiprocessing. Default: 1.
    remove_temp : boolean, optional
        option to remove temporary directory after execution. Default: True (clean up)
    verbose : boolean
        Whether to write output to stdout. Default: True.
    temp_dir : str, optional
        Path to a directory to store files. Is only used if input .gtf-file needs sorting. Default: "" (current working dir).
    inplace : boolean
        Whether to add the annotations to the adata object in place. Default: True.

    Returns
    --------
    If inplace == True, the annotation is added to adata.var in place.
    Else, a copy of the adata object is returned with the annotations added.

    References
    ----------
        .. [1] Kondili M, Fust A, Preussner J, Kuenne C, Braun T, and Looso M. UROPA: a tool for Universal RObust Peak Annotation. Scientific Reports 7 (2017), doi: 10.1038/s41598-017-02464-y

    Examples
    ---------
    >>> custom_config = {"queries": [{"distance": [10000, 1000],
                                      "feature_anchor": "start",
                                      "feature": "gene"}],
                                 "priority": True,
                                 "show_attributes": "all"}

    >>> annotate_regions(adata, gtf="genes.gtf",
                                config=custom_config)
    """
    # Setup verbose print function
    print = utils.vprint(verbose)

    # Make temporary directory if needed
    utils.create_dir(temp_dir)

    # Check that packages are installed
    utils.check_module("uropa")  # will raise an error if not installed
    import uropa.utils

    # TODO: Check input types
    # check_type(gtf, str, "gtf")
    # check_type(config, [type(None), dict], "config")
    # check_value(threads, vmin=1, name="threads")

    # Establish configuration dict
    print("Setting up annotation configuration...")
    if config is None:
        cfg_dict = {"queries": [{"distance": [10000, 1000],
                                 "feature_anchor": "start",
                                 "feature": "gene",
                                 "name": "promoters"}],
                    "priority": True,
                    "show_attributes": "all"}
    else:
        cfg_dict = config

    cfg_dict = copy.deepcopy(cfg_dict)  # make sure that config is not being changed in place
    logger = uropa.utils.UROPALogger()
    cfg_dict = uropa.utils.format_config(cfg_dict, logger=logger)
    print("Config dictionary: {0}".format(cfg_dict))

    # Read regions from .var table
    print("Setting up genomic regions to annotate...")
    regions = adata.var.copy()

    # Establish columns for coordinates
    if coordinate_cols is None:
        coordinate_cols = adata.var.columns[:3]  # first three columns are coordinates
    else:
        utils.check_columns(adata.var, coordinate_cols, "coordinate_cols")  # Check that coordinate_cols are in adata.var)

    # Test the coordinate columns
    format_adata_var(adata, coordinate_cols, coordinate_cols)  # will raise an error if not valid or try to convert from index

    # Convert regions to dict for uropa
    idx2name = {i: name for i, name in enumerate(regions.index)}
    regions.reset_index(inplace=True, drop=True)  # drop index to get unique order for peak id
    region_dicts = []
    for idx, row in regions.iterrows():
        d = {"peak_chr": row[coordinate_cols[0]],
             "peak_start": int(row[coordinate_cols[1]]),
             "peak_end": int(row[coordinate_cols[2]]),
             "peak_id": idx}
        region_dicts.append(d)

    # Unzip, sort and index gtf if necessary
    gtf, tempfiles = prepare_gtf(gtf, temp_dir, print)

    annotations_table = annotate_features(region_dicts, threads, gtf, cfg_dict, best)

    # Preparation of adata.var update

    # Drop some columns already present
    drop_columns = ["peak_chr", "peak_start", "peak_end"]
    annotations_table.drop(columns=drop_columns, inplace=True)

    # Rename feat -> gene to prevent confusion with input features
    rename_dict = {c: c.replace("feat", "gene") for c in annotations_table.columns}
    rename_dict.update({'peak_id': '_peak_id',
                        'feature': 'annotation_feature',
                        'distance': 'distance_to_gene',
                        'relative_location': 'relative_location_to_gene',
                        'query': 'annotation_query'})
    annotations_table.rename(columns=rename_dict, inplace=True)

    # Set columns as categorical
    cols = rename_dict.values()
    cols = [col for col in cols if col not in ["distance_to_gene", "gene_ovl_peak", "peak_ovl_gene"]]
    for col in cols:
        annotations_table[col] = annotations_table[col].astype('category')

    # Check if columns already exist
    existing = set(regions.columns).intersection(annotations_table.columns)
    if len(existing) > 0:
        print("WARNING: The following annotation columns already exist in adata.var: {0}".format(existing))
        print("These columns will be overwritten by the annotation")
        regions.drop(columns=existing, inplace=True)

    # Merge original sites with annotations
    regions_annotated = regions.merge(annotations_table, how="outer", left_index=True, right_on="_peak_id")
    regions_annotated.index = [idx2name[i] for i in regions_annotated["_peak_id"]]  # put index name back into regions
    regions_annotated.drop(columns=["_peak_id"], inplace=True)

    # Duplicate peaks in adata if best is False
    if best is False:
        adata = adata[:, regions_annotated.index]  # duplicates rows of adata.X and .var

    # Save var input object
    adata.var = regions_annotated

    print("Finished annotation of features! The results are found in the .var table.")

    # Remove temporary directory
    if remove_temp:
        rm_tmp(temp_dir, tempfiles)

    if inplace is False:
        return adata  # else returns None


def annotate_narrowPeak(filepath,
                        gtf,
                        config=None,
                        best=True,
                        threads=1,
                        temp_dir="",
                        remove_temp=True,
                        verbose=True):

    """
    Annotates narrowPeak files with genes from .gtf using UROPA.

    :param filepath: str
        Path to the narrowPeak file to be annotated
    :param gtf: str
        Path to the .gtf file containing the genes to be annotated
    :param config: dict, optional
        A dictionary indicating how regions should be annotated. Default is to annotate feature 'gene' within -10000;1000bp of the gene start. See 'Examples' of how to set up a custom configuration dictionary.
    :param best: boolean
        Whether to return the best annotation or all valid annotations. Default: True (only best are kept).
    :param threads: int, optional
        Number of threads to perform the annotation.
    :param temp_dir: str, optional
        Path to the directory where the temporary files should be written. Default: location of the script
    :param remove_temp: boolean, optional
        option to remove temporary directory after execution. Default: True (clean up)
    :param verbose: boolean
        Whether to write output to stdout. Default: True.
    :return: annotation_table: pandas.Dataframe
        Dataframe containing the annotations.
    """
    # Setup verbose print function
    print = utils.vprint(verbose)

    # Make temporary directory
    utils.create_dir(temp_dir)

    # Check that packages are installed
    utils.check_module("uropa")  # will raise an error if not installed
    import uropa.utils

    # TODO: Check input types
    # check_type(gtf, str, "gtf")
    # check_type(config, [type(None), dict], "config")
    # check_value(threads, vmin=1, name="threads")

    # Establish configuration dict
    print("Setting up annotation configuration...")
    if config is None:
        cfg_dict = {"queries": [{"distance": [10000, 1000],
                                 "feature_anchor": "start",
                                 "feature": "gene",
                                 "name": "promoters"}],
                    "priority": True,
                    "show_attributes": "all"}
    else:
        cfg_dict = config

    cfg_dict = copy.deepcopy(cfg_dict)  # make sure that config is not being changed in place
    logger = uropa.utils.UROPALogger()
    cfg_dict = uropa.utils.format_config(cfg_dict, logger=logger)
    print("Config dictionary: {0}".format(cfg_dict))

    region_dicts = load_narrowPeak(filepath, print)

    # Unzip, sort and index gtf if necessary
    gtf, tempfiles = prepare_gtf(gtf, temp_dir, print)

    annotation_table = annotate_features(region_dicts, threads, gtf, cfg_dict, best)

    print("annotation done")

    # Remove temporary directory
    if remove_temp:
        rm_tmp(temp_dir, tempfiles)

    return annotation_table


def load_narrowPeak(filepath, print):
    '''
    Load narrowPeak file to annotate.

    :param filepath: str
        Path to the narrowPeak file.
    :param print: function
        Print function for verbose printing.
    :return: region_dicts: dictionary
        Dictionary with the peak information.
    '''
    print("load regions_dict from: " + filepath)
    peaks = pd.read_csv(filepath, header=None, sep='\t')
    peaks = peaks.drop([3, 4, 5, 6, 7, 8, 9], axis=1)
    peaks.columns = ["peak_chr", "peak_start", "peak_end"]
    peaks["peak_id"] = peaks.index

    region_dicts = []
    for i in range(len(peaks.index)):
        entry = peaks.iloc[i]
        dict = entry.to_dict()
        region_dicts.append(dict)

    return region_dicts


def prepare_gtf(gtf,
                temp_dir,
                print):
    """
    Prepares the .gtf file to use it in the annotation process. Therefore the file properties are checked and if necessary it is sorted, indexed and compressed.

    :param gtf: str
        Path to the .gtf file containing the genes to be annotated.
    :param temp_dir: str
        Path to the temporary directory for storing .gtf files.
    :param print: function
        Function for verbose printing.

    :return: gtf: str
        Path to the gtf file to use in the annotation.
    :return: tempfiles: list of str
        List of temporary files created.
    """
    utils.check_module("pysam")
    import pysam

    # input_gtf = gtf
    # Check integrity of the gtf file
    gtf_integrity(gtf)  # will raise an error if gtf is not valid

    tempfiles = []

    # Prepare .gtf file in terms of index and sorting
    print("Preparing gtf file for annotation...")
    success = 0
    sort_done = 0
    while success == 0:
        try:  # try to open gtf with Tabix
            print("- Reading gtf with Tabix")
            g = pysam.TabixFile(gtf)  # gtf can be .gz or not
            g.close()
            success = 1
            print("Done preparing gtf!")

        except Exception:  # if not possible, try to sort gtf
            print("- Index of gtf not found - trying to index gtf")

            # First check if gtf was already gzipped
            if not _is_gz_file(gtf):
                base = os.path.basename(gtf)
                gtf_gz = os.path.join(temp_dir, base + ".gz")
                pysam.tabix_compress(gtf, gtf_gz, force=True)
                tempfiles.append(gtf_gz)
                gtf = gtf_gz

            # Try to index
            try:
                gtf_index = gtf + ".tbi"
                gtf = pysam.tabix_index(gtf, seq_col=0, start_col=3, end_col=4, keep_original=True, force=True, meta_char='#', index=gtf_index)
                tempfiles.append(gtf_index)

            except Exception:
                print("- Indexing failed - the GTF is probably unsorted")

                # Start by uncompressing file if file is gz
                is_gz = _is_gz_file(gtf)
                if is_gz:
                    gtf_uncompressed = os.path.join(temp_dir, "uncompressed.gtf")
                    print(f"- Uncompressing {gtf} to: {gtf_uncompressed}")
                    try:
                        gunzip_file(gtf, gtf_uncompressed)
                    except Exception:
                        raise ValueError("Could not uncompress gtf file to sort. Please ensure that the input gtf is sorted.")
                    gtf = gtf_uncompressed
                    tempfiles.append(gtf_uncompressed)

                # Try to sort gtf
                if sort_done == 0:  # make sure sort was not already performed
                    gtf_sorted = os.path.join(temp_dir, "sorted.gtf")
                    sort_call = "grep -v \"^#\" {0} | sort -k1,1 -k4,4n > {1}".format(gtf, gtf_sorted)
                    print("- Attempting to sort gtf with call: '{0}'".format(sort_call))

                    try:
                        _ = subprocess.check_output(sort_call, shell=True)
                        gtf = gtf_sorted  # this gtf will now go to next loop in while
                        sort_done = 1

                    except subprocess.CalledProcessError:
                        raise ValueError("Could not sort gtf file using command-line call: {0}".format(sort_call))
                else:
                    raise ValueError("Could not read input gtf - please check for the correct format.")

    # Force close of gtf file left open; pysam issue 1038
    proc = psutil.Process()
    for f in proc.open_files():
        if f.path == os.path.abspath(gtf):
            os.close(f.fd)

    return gtf, tempfiles


def annotate_features(region_dicts,
                      threads,
                      gtf,
                      cfg_dict,
                      best):

    '''

    :param region_dicts: dictionary
        dictionary with peak information.
    :param threads: int
        number of threads to perform the annotation.
    :param gtf: str
        Path to the .gtf file
    :param cfg_dict:
        A dictionary indicating how regions should be annotated. Default is to annotate feature 'gene' within -10000;1000bp of the gene start. See 'Examples' of how to set up a custom configuration dictionary.
    :param best: boolean
        Whether to return the best annotation or all valid annotations. Default: True (only best are kept).

    :return: pandas.Dataframe
        Dataframe with the annotation
    '''

    # split input regions into cores
    n_reg = len(region_dicts)
    per_chunk = int(np.ceil(n_reg / float(threads)))
    region_dict_chunks = [region_dicts[i:i + per_chunk] for i in range(0, n_reg, per_chunk)]

    # calculate annotations for each chunk
    print("Annotating regions...")
    annotations = []
    if threads == 1:
        print("NOTE: Increase --threads to speed up computation")
        for region_chunk in region_dict_chunks:
            chunk_annotations = _annotate_peaks_chunk(region_chunk, gtf, cfg_dict)
            annotations.extend(chunk_annotations)
    else:

        # Start multiprocessing pool
        pool = mp.Pool(threads)

        # Start job for each chunk
        jobs = []
        for region_chunk in region_dict_chunks:
            job = pool.apply_async(_annotate_peaks_chunk, (region_chunk, gtf, cfg_dict, ))
            jobs.append(job)
        pool.close()

        # TODO: Print progress
        """
        n_jobs = len(jobs)
        n_done = 0
        #pbar = tqdm.tqdm(total=n_jobs)
        #pbar.set_description("Annotating regions")
        #while n_done < n_jobs:
            #n_current_done = sum([job.ready() for job in jobs])
            #time.wait(1)
            #if n_current_done != n_done:
                #pbar.update()
            #n_done = n_current_done
        """

        # Collect results:
        for job in jobs:
            chunk_annotations = job.get()
            annotations.extend(chunk_annotations)

        pool.join()

    print("Formatting annotations...")

    # Select best annotations
    if best is True:
        annotations = [annotations[i] for i, anno in enumerate(annotations) if anno["best_hit"] == 1]

    # Extend feat_attributes per annotation and format for output
    del_keys = ["raw_distance", "anchor_pos", "peak_center", "peak_length", "feat_length", "feat_center"]
    for anno in annotations:
        if "feat_attributes" in anno:
            for key in anno["feat_attributes"]:
                anno[key] = anno["feat_attributes"][key]
            del anno["feat_attributes"]

        # remove certain keys
        for key in del_keys:
            if key in anno:
                del anno[key]

        # Remove best_hit column if best is True
        if best is True:
            del anno["best_hit"]

        # Convert any lists to string
        for key in anno:
            if isinstance(anno[key], list):
                anno[key] = anno[key][0]

    # Convert to pandas table
    annotations_table = pd.DataFrame(annotations)

    return annotations_table


def _annotate_peaks_chunk(region_dicts, gtf, cfg_dict):
    """ Multiprocessing safe function to annotate a chunk of regions """

    import pysam
    import uropa
    from uropa.annotation import annotate_single_peak

    logger = uropa.utils.UROPALogger()

    # Open tabix file
    tabix_obj = pysam.TabixFile(gtf)

    # For each peak in input peaks, collect all_valid_annotations
    all_valid_annotations = []
    for i, region in enumerate(region_dicts):

        # Annotate single peak
        valid_annotations = annotate_single_peak(region, tabix_obj, cfg_dict, logger=logger)
        all_valid_annotations.extend(valid_annotations)

    tabix_obj.close()

    return (all_valid_annotations)


def annot_HVG(anndata, min_mean=0.0125, max_iterations=10, hvg_range=(1000, 5000), step=10, inplace=True, **kwargs):
    """
    Annotate highly variable genes (HVG). Tries to annotate in given range of HVGs, by gradually in-/ decreasing min_mean of scanpy.pp.highly_variable_genes.

    Note: Logarithmized data is expected.

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object to annotate.
    min_mean : float, default 0.0125
        Starting min_mean parameter for finding HVGs.
    max_iterations : int, default 10
        Maximum number of min_mean adjustments.
    hvg_range : int tuple, default (1000, 5000)
        Number of HVGs should be in the given range. Will issue a warning if result is not in range.
        Default limits are chosen as proposed by https://doi.org/10.15252/msb.20188746.
    step : float, default 10
        Value min_mean is adjusted by in each iteration. Will divide min_value (below range) or multiply (above range) by this value.
    inplace : boolean, default False
        Whether the anndata object is modified inplace.
    **kwargs :
        Additional arguments forwarded to scanpy.pp.highly_variable_genes().

    Returns
    -------
    anndata.Anndata or None:
        Adds annotation of HVG to anndata object. Information is added to Anndata.var["highly_variable"].
    """
    adata_m = anndata if inplace else anndata.copy()

    print("Annotating highy variable genes (HVG)")

    # adjust min_mean to get a HVG count in a certain range
    for i in range(max_iterations + 1):
        sc.pp.highly_variable_genes(adata_m, min_mean=min_mean, inplace=True, **kwargs)

        # counts True values in column
        hvg_count = sum(adata_m.var.highly_variable)

        # adjust min_mean
        # skip adjustment if in last iteration
        if i < max_iterations and hvg_count < hvg_range[0]:
            min_mean /= step
        elif i < max_iterations and hvg_count > hvg_range[1]:
            min_mean *= step + 0.00001  # This .000001 is to avoid an infinit loop if the current mean lie in the above if.
        else:
            break

    # warn if outside of range
    if hvg_count < hvg_range[0] or hvg_count > hvg_range[1]:
        warnings.warn(f"Number of HVGs not in range. Range is {hvg_range} but counted {hvg_count}.")
    else:
        sc.pl.highly_variable_genes(anndata, show=False)  # Plot dispersion of HVG
        print("Total HVG=" + str(anndata.var["highly_variable"].sum()))

    # Adding info in anndata.uns["infoprocess"]
    cr.build_infor(anndata, "Scanpy annotate HVG", "min_mean= " + str(min_mean) + "; Total HVG= " + str(hvg_count), inplace=True)

    if not inplace:
        return adata_m
