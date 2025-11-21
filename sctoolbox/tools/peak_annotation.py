"""Tools for peak annotation required by scATAC-seq."""
import pandas as pd
import numpy as np
import copy
import os
import multiprocessing as mp
import psutil
import subprocess
import scanpy as sc

from beartype.typing import Optional, Union, Tuple, Any, Literal
from beartype import beartype

import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


#################################################################################
# ------------------------ Uropa annotation of peaks -------------------------- #
#################################################################################

@deco.log_anndata
@beartype
def annotate_adata(adata: sc.AnnData,
                   gtf: str,
                   config: Optional[dict[str, Any]] = None,
                   best: bool = True,
                   threads: Optional[int] = 1,
                   coordinate_cols: Optional[list[str]] = None,
                   temp_dir: str = "",
                   remove_temp: bool = True,
                   inplace: bool = True,
                   report: bool = False) -> Optional[sc.AnnData]:
    """
    Annotate adata .var features with genes from .gtf using UROPA [1]_.

    The function assumes that the adata.var contains genomic coordinates in the first
    three columns, e.g. "chromosome", "start", "end".
    If specific columns should be used, please adjust the names via 'coordinate_cols'.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object containing features to annotate.
    gtf : str
        Path to .gtf file containing genomic elements for annotation.
    config : Optional[dict[str, Any]], default None
        A dictionary indicating how regions should be annotated.
        Default (None) is to annotate feature 'gene' within -10000;1000bp of the gene start.
        See 'Examples' of how to set up a custom configuration dictionary.
    best : bool, default True
        Whether to return the best annotation or all valid annotations.
    threads : Optional[int], default 1
        Number of threads to use for multiprocessing. None to use settings.get_threads.
    coordinate_cols : Optional[list[str]], default None
        A list of column names in the regions DataFrame that contain the chromosome, start and end coordinates.
        If None the first three columns are taken.
    temp_dir : str, default ''
        Path to a directory to store files. Is only used if input .gtf-file needs sorting.
    remove_temp : boolean, default True
        If True remove temporary directory after execution.
    inplace : boolean, default True
        Whether to add the annotations to the adata object in place.
    report : boolean, default False
        Add UROPA annotation to report (methods slide) if True and sctoolbox.settings.report_dir is not empty.

    Returns
    -------
    Optional[sc.AnnData]
        If inplace == True, the annotation is added to adata.var in place.
        Else, a copy of the adata object is returned with the annotations added.

    References
    ----------
        .. [1] Kondili M, Fust A, Preussner J, Kuenne C, Braun T, and Looso M. UROPA: a tool for Universal RObust Peak Annotation. Scientific Reports 7 (2017), doi: 10.1038/s41598-017-02464-y

    Examples
    --------
    >>> custom_config = {"queries": [{"distance": [10000, 1000],
                                      "feature_anchor": "start",
                                      "feature": "gene"}],
                                 "priority": True,
                                 "show_attributes": "all"}

    >>> annotate_regions(adata, gtf="genes.gtf",
                                config=custom_config)
    """

    if threads is None:
        threads = settings.get_threads()

    # Make temporary directory if needed
    utils.io.create_dir(temp_dir)

    # Check that packages are installed
    utils.checker.check_module("uropa")  # will raise an error if not installed
    import uropa.utils

    # TODO: Check input types
    # check_type(gtf, str, "gtf")
    # check_type(config, [type(None), dict], "config")
    # check_value(threads, vmin=1, name="threads")

    # Establish configuration dict
    logger.info("Setting up annotation configuration...")
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
    uropa_logger = uropa.utils.UROPALogger()
    cfg_dict = uropa.utils.format_config(cfg_dict, logger=uropa_logger)
    logger.info("Config dictionary: {0}".format(cfg_dict))

    # Read regions from .var table
    logger.info("Setting up genomic regions to annotate...")
    regions = adata.var.copy()

    # Establish columns for coordinates
    if coordinate_cols is None:
        coordinate_cols = adata.var.columns[:3]  # first three columns are coordinates
    else:
        utils.checker.check_columns(adata.var, coordinate_cols, name="coordinate_cols")  # Check that coordinate_cols are in adata.var)

    # Test the coordinate columns
    utils.checker.var_index_to_column(adata, coordinate_cols)  # will raise an error if not valid or try to convert from index

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
    gtf, tempfiles = _prepare_gtf(gtf, temp_dir)

    annotations_table = _annotate_features(region_dicts, threads, gtf, cfg_dict, best)

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
        logger.warning("WARNING: The following annotation columns already exist in adata.var: {0}".format(existing))
        logger.info("These columns will be overwritten by the annotation")
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

    logger.info("Finished annotation of features! The results are found in the .var table.")

    # Remove temporary directory
    if remove_temp:
        utils.io.rm_tmp(temp_dir, tempfiles)

    # enable UROPA sentence on the methods slide
    if settings.report_dir and report:
        utils.io.update_yaml({"UROPA": True}, yml="method.yml", path_prefix="report")

    if inplace is False:
        return adata  # else returns None


@beartype
def annotate_narrowPeak(filepath: str,
                        gtf: str,
                        config: Optional[dict[str, Any]] = None,
                        best: bool = True,
                        threads: Optional[int] = 1,
                        temp_dir: str = "",
                        remove_temp: bool = True) -> pd.DataFrame:
    """
    Annotate narrowPeak files with genes from .gtf using UROPA.

    Parameters
    ----------
    filepath : str
        Path to the narrowPeak file to be annotated.
    gtf : str
        Path to the .gtf file containing the genes to be annotated.
    config : Optional[dict[str, Any]], default None
        A dictionary indicating how regions should be annotated.
        Default (None) is to annotate feature 'gene' within -10000;1000bp of the gene start.
        See 'Examples' of how to set up a custom configuration dictionary.
    best : bool, default True
        Whether to return the best annotation or all valid annotations.
    threads : Optional[int], default 1
        Number of threads to perform the annotation. None to use settings.get_threads
    temp_dir : str, default ''
        Path to the directory where the temporary files should be written.
    remove_temp : bool, default True
        If True remove temporary directory after execution.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the annotations.
    """

    if threads is None:
        threads = settings.get_threads()

    # Make temporary directory
    utils.io.create_dir(temp_dir)

    # Check that packages are installed
    utils.checker.check_module("uropa")  # will raise an error if not installed
    import uropa.utils

    # Establish configuration dict
    logger.info("Setting up annotation configuration...")
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
    uropa_logger = uropa.utils.UROPALogger()
    cfg_dict = uropa.utils.format_config(cfg_dict, logger=uropa_logger)
    logger.info("Config dictionary: {0}".format(cfg_dict))

    region_dicts = _load_narrowPeak(filepath)

    # Unzip, sort and index gtf if necessary
    gtf, tempfiles = _prepare_gtf(gtf, temp_dir)

    annotation_table = _annotate_features(region_dicts, threads, gtf, cfg_dict, best)

    logger.info("annotation done")

    # Remove temporary directory
    if remove_temp:
        utils.io.rm_tmp(temp_dir, tempfiles)

    return annotation_table


@beartype
def _load_narrowPeak(filepath: str) -> list[dict[str, Union[str, int]]]:
    """
    Load narrowPeak file to annotate.

    Parameters
    ----------
    filepath : str
        Path to the narrowPeak file.

    Returns
    -------
    list[dict[str, Union[str, int]]]
        List of dictionaries containing peak range information.
    """

    logger.info("load regions_dict from: " + filepath)
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


@beartype
def _prepare_gtf(gtf: str,
                 temp_dir: str) -> Tuple[str, list[str]]:
    """
    Prepare the .gtf file to use it in the annotation process.

    Therefore the file properties are checked and if necessary it is sorted,
    indexed and compressed.

    Parameters
    ----------
    gtf : str
        Path to the .gtf file containing the genes to be annotated.
    temp_dir : str
        Path to the temporary directory for storing .gtf files.

    Returns
    -------
    Tuple[str, list[str]]
        Index 1: str : Path to the gtf file to use in the annotation.
        Index 2: list : List of paths to temporary files created.

    Raises
    ------
    ValueError
        1: If GTF-file could not be uncompressed. Whiel trying to sort the GTF-file.
        2: If one subprocess during the sorting step fails.
        3: If GTF-file could not be read. For example, due to an invalid format.
    """

    utils.checker.check_module("pysam")
    import pysam

    # input_gtf = gtf
    # Check integrity of the gtf file
    utils.bioutils._gtf_integrity(gtf)  # will raise an error if gtf is not valid

    tempfiles = []

    # Prepare .gtf file in terms of index and sorting
    logger.info("Preparing gtf file for annotation...")
    success = 0
    sort_done = 0
    while success == 0:
        try:  # try to open gtf with Tabix
            logger.info("- Reading gtf with Tabix")
            g = pysam.TabixFile(gtf)  # gtf can be .gz or not
            g.close()
            success = 1
            logger.info("Done preparing gtf!")

        except Exception:  # if not possible, try to sort gtf
            logger.info("- Index of gtf not found - trying to index gtf")

            # First check if gtf was already gzipped
            if not utils.checker._is_gz_file(gtf):
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
                logger.info("- Indexing failed - the GTF is probably unsorted")

                # Start by uncompressing file if file is gz
                is_gz = utils.checker._is_gz_file(gtf)
                if is_gz:
                    gtf_uncompressed = os.path.join(temp_dir, "uncompressed.gtf")
                    logger.info(f"- Uncompressing {gtf} to: {gtf_uncompressed}")
                    try:
                        utils.checker.gunzip_file(gtf, gtf_uncompressed)
                    except Exception:
                        raise ValueError("Could not uncompress gtf file to sort. Please ensure that the input gtf is sorted.")
                    gtf = gtf_uncompressed
                    tempfiles.append(gtf_uncompressed)

                # Try to sort gtf
                if sort_done == 0:  # make sure sort was not already performed
                    gtf_sorted = os.path.join(temp_dir, "sorted.gtf")
                    sort_call = "grep -v \"^#\" {0} | sort -k1,1 -k4,4n > {1}".format(gtf, gtf_sorted)
                    logger.info("- Attempting to sort gtf with call: '{0}'".format(sort_call))

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


@beartype
def _annotate_features(region_dicts: list[dict[Literal["peak_chr", "peak_start", "peak_end", "peak_id"], str | int]],
                       threads: int,
                       gtf: str,
                       cfg_dict: Optional[dict[str, Union[list, bool, str, int, float]]],
                       best: bool) -> pd.DataFrame:
    """
    Annotate features.

    Parameters
    ----------
    region_dicts : list[dict[Literal["peak_chr", "peak_start", "peak_end", "peak_id"], str | int]]
        List of dictionary with peak information.
    threads : int
        Number of threads to perform the annotation.
    gtf : str
        Path to the .gtf file
    cfg_dict : Optional[dict[str, Union[list, bool, str, int, float]]]
        A dictionary indicating how regions should be annotated.
        Set to None to annotate feature 'gene' within -10000;1000bp of the gene start.
    best : bool
        Whether to return the best annotation or all valid annotations.

    Returns
    -------
    pd.DataFrame
        Dataframe with the annotation

    Examples
    --------
    >>> region_dicts = [{"peak_chr": chr1, "peak_start": 100, "peak_end": 200, "peak_id": "peak1"},
                        {"peak_chr": chr1, "peak_start": 600, "peak_end": 800, "peak_id": "peak2"}]

    >>> cfg_dict = {"queries": [{"distance": [10000, 1000], "feature_anchor": "start", "feature": "gene"}],
                                 "priority": True,
                                 "show_attributes": "all"}
    """

    # split input regions into cores
    n_reg = len(region_dicts)
    per_chunk = int(np.ceil(n_reg / float(threads)))
    region_dict_chunks = [region_dicts[i:i + per_chunk] for i in range(0, n_reg, per_chunk)]

    # calculate annotations for each chunk
    logger.info("Annotating regions...")
    annotations = []
    if threads == 1:
        logger.info("NOTE: Increase --threads to speed up computation")
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

    logger.info("Formatting annotations...")

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


@beartype
def _annotate_peaks_chunk(region_dicts: list[dict[Literal["peak_chr", "peak_start", "peak_end", "peak_id"], str | int]],
                          gtf: str,
                          cfg_dict: Optional[dict[str, Union[list, bool, str, int, float]]]) -> list[dict[str, Union[str, int]]]:
    """
    Multiprocessing safe function to annotate a chunk of regions.

    Parameters
    ----------
    region_dicts : list[dict[Literal["peak_chr", "peak_start", "peak_end", "peak_id"], str | int]]
        List of dictionaryies with peak information.
    gtf : str
        Path to the .gtf file
    cfg_dict : Optional[dict[str, Union[list, bool, str, int, float]]]
        A dictionary indicating how regions should be annotated.

    Returns
    -------
    list[dict[str, Union[str, int]]]
        List of all valid annotations.
    """

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
