"""Functionality to split bam files into smaller bam files based on clustering in adata.obs."""
import re
import os
import pandas as pd
import multiprocessing as mp
from multiprocessing.managers import BaseProxy
from multiprocessing.pool import ApplyResult
import scanpy as sc
from functools import partial

from beartype.typing import TYPE_CHECKING, Iterable, Optional, Literal, Any, Sequence
from beartype import beartype

import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger

if TYPE_CHECKING:
    import pysam


@beartype
def bam_adata_ov(adata: sc.AnnData,
                 bamfile: str,
                 cb_tag: str = "CB") -> float:
    """
    Check if adata.obs barcodes existing in a column of a bamfile.

    Parameters
    ----------
    adata : sc.AnnData
        adata object where adata.obs is stored
    bamfile : str
        path of the bamfile to investigate
    cb_tag : str, default 'CB'
        bamfile column to extract the barcodes from

    Returns
    -------
    float
        hitrate of the barcodes in the bamfile
    """

    logger.info("calculating barcode overlap between bamfile and adata.obs")
    bam_obj = open_bam(bamfile, "rb")

    sample = []
    counter = 0
    iterations = 1000
    for read in bam_obj:
        tag = read.get_tag(cb_tag)
        sample.append(tag)
        if counter == iterations:
            break
        counter += 1

    barcodes_df = pd.DataFrame(adata.obs.index)
    count_table = barcodes_df.isin(sample)
    hits = count_table.sum()
    hitrate = hits[0] / iterations

    return hitrate


@deco.log_anndata
@beartype
def check_barcode_tag(adata: sc.AnnData,
                      bamfile: str,
                      cb_tag: str = "CB") -> None:
    """
    Check for the possibilty that the wrong barcode is used.

    Parameters
    ----------
    adata : sc.AnnData
        adata object where adata.obs is stored
    bamfile : str
        path of the bamfile to investigate
    cb_tag : str, default 'CB'
        bamfile column to extract the barcodes from

    Raises
    ------
    ValueError
        If barcode hit rate could not be identified
    """

    hitrate = bam_adata_ov(adata, bamfile, cb_tag)

    if hitrate == 0:
        logger.warning('None of the barcodes from the bamfile found in the .obs table.\n'
                       'Consider if you are using the wrong column cb-tag or bamfile.')
    elif hitrate <= 0.05:
        logger.warning('Only 5% or less of the barcodes from the bamfile found in the .obs table.\n'
                       'Consider if you are using the wrong column for cb-tag or bamfile.')
    elif hitrate > 0.05:
        logger.info('Barcode tag: OK')
    else:
        raise ValueError("Could not identify barcode hit rate.")

#####################################################################


@beartype
def subset_bam(bam_in: str,
               bam_out: str,
               barcodes: Iterable[str],
               read_tag: str = "CB",
               pysam_threads: Optional[int] = 4,
               overwrite: bool = False) -> None:
    """
    Subset a bam file based on a list of barcodes.

    Parameters
    ----------
    bam_in : str
        Path to input bam file.
    bam_out : str
        Path to output bam file.
    barcodes : Iterable[str]
        List of barcodes to keep.
    read_tag : str, default "CB"
        Tag in bam file to use for barcode.
    pysam_threads : Optional[int], default 4
        Number of threads to use for pysam. Set None to use settings.get_threads.
    overwrite : bool, default False
        Overwrite output file if it exists.
    """

    if pysam_threads is None:
        pysam_threads = settings.get_threads()

    # check then load modules
    utils.checker.check_module("tqdm")
    if utils.jupyter._is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
    utils.checker.check_module("pysam")

    # Create output dir if needed
    utils.io.create_dir(bam_out)

    if os.path.exists(bam_out) and overwrite is False:
        logger.warning(f"Output file {bam_out} exists. Skipping.")
        return

    # Open files
    bam_in_obj = open_bam(bam_in, "rb", verbosity=0, threads=pysam_threads)
    bam_out_obj = open_bam(bam_out, "wb", template=bam_in_obj, threads=pysam_threads, verbosity=0)

    barcodes = set(barcodes)

    # Update progress based on total number of reads
    total = get_bam_reads(bam_in_obj)
    print(' ', end='', flush=True)  # hack for making progress bars work in notebooks; https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308
    pbar_reading = tqdm(total=total, desc="Reading... ", unit="reads")
    pbar_writing = tqdm(total=total, desc="% written from input", unit="reads")
    step = int(total / 10000)  # 10000 total updates

    # Iterate over reads
    writing_i = 0
    reading_i = 0
    written = 0
    for read in bam_in_obj:
        reading_i += 1
        if read.has_tag(read_tag):
            bc = read.get_tag(read_tag)
        else:
            bc = None

        # Update step manually - there is an overhead to update per read with hundreds of million reads
        if reading_i == step:
            pbar_reading.update(step)
            pbar_reading.refresh()
            reading_i = 0

        # Write read to output bam if barcode is in barcodes
        if bc in barcodes:
            bam_out_obj.write(read)
            written += 1

            if writing_i == step:
                pbar_writing.update(step)
                pbar_writing.refresh()
                writing_i = 0

    # close progressbars
    pbar_reading.close()
    pbar_writing.close()

    # Close bamfiles
    bam_in_obj.close()
    bam_out_obj.close()
    logger.info(f"Wrote {written} reads to output bam")


@deco.log_anndata
@beartype
def split_bam_clusters(adata: sc.AnnData,
                       bams: str | Iterable[str],
                       groupby: str,
                       barcode_col: Optional[str] = None,
                       read_tag: str = "CB",
                       output_prefix: str = "split_",
                       reader_threads: Optional[int] = 1,
                       writer_threads: Optional[int] = 1,
                       parallel: bool = False,
                       pysam_threads: Optional[int] = 4,
                       buffer_size: int = 10000,
                       max_queue_size: int = 1000,
                       individual_pbars: bool = False,
                       sort_bams: bool = False,
                       index_bams: bool = False) -> None:
    """
    Split BAM files into clusters based on 'groupby' from the anndata.obs table.

    Parameters
    ----------
    adata : sc.Anndata
        Annotated data matrix containing clustered cells in .obs.
    bams : str | Iterable[str]
        One or more BAM files to split into clusters
    groupby : str
        Name of a column in adata.obs to cluster the pseudobulks by.
    barcode_col : str, default None
        Name of a column in adata.obs to use as barcodes. If None, use the index of .obs.
    read_tag : str, default "CB"
        Tag to use to identify the reads to split. Must match the barcodes of the barcode_col.
    output_prefix : str, default `split_`
        Prefix to use for the output files.
    reader_threads : Optional[int], default 1
        Number of threads to use for reading. Set None to use settings.get_threads.
    writer_threads : Optional[int], default 1
        Number of threads to use for writing. Set None to use settings.get_threads.
    parallel : bool, default False
        Whether to enable parallel processsing.
    pysam_threads : Optional[int], default 4
        Number of threads for pysam. Set None to use settings.get_threads.
    buffer_size : int, default 10000
        The size of the buffer between readers and writers.
    max_queue_size : int, default 1000
        The maximum size of the queue between readers and writers.
    individual_pbars : bool, default False
        Whether to show a progress bar for each individual BAM file and output clusters. Default: False (overall progress bars).
    sort_bams : bool, default False
        Sort reads in each output bam
    index_bams : bool, default False
        Create an index file for each output bam. Will throw an error if `sort_bams` is False.

    Raises
    ------
    ValueError
        1. If groupby column is not in adata.obs
        2. If barcode column is not in adata.obs
        3. If index_bams is set and sort_bams is False
    """

    # then load modules
    utils.checker.check_module("tqdm")
    if utils.jupyter._is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    utils.checker.check_module("pysam")
    import pysam

    # check whether groupby and barcode_col are in adata.obs
    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs!")

    if barcode_col is not None and barcode_col not in adata.obs.columns:
        raise ValueError(f"Column '{barcode_col}' not found in adata.obs!")

    if index_bams and not sort_bams:
        raise ValueError("`sort_bams=True` must be set for indexing to be possible.")

    if isinstance(bams, str):
        bams = [bams]

    # create output folder if needed
    utils.io.create_dir(os.path.dirname(output_prefix))  # upper directory of prefix

    # Establish clusters from obs
    clusters = list(set(adata.obs[groupby]))
    logger.info(f"Found {len(clusters)} groups in .obs.{groupby}: {clusters}")

    if writer_threads > len(clusters):
        logger.info(f"The number of writers ({writer_threads}) is larger than the number of output clusters ({len(clusters)}). Limiting writer_threads to the number of clusters.")
        writer_threads = len(clusters)

    # setup barcode <-> cluster dict
    if barcode_col is None:
        barcode2cluster = dict(zip(adata.obs.index.tolist(), adata.obs[groupby]))
    else:
        barcode2cluster = dict(zip(adata.obs[barcode_col], adata.obs[groupby]))

    # create template used for bam header
    template = open_bam(bams[0], "rb", verbosity=0)

    # Get number in reads in input bam(s)
    logger.info("Reading total number of reads from bams...")
    n_reads = {}
    for path in bams:
        handle = open_bam(path, "rb", verbosity=0)
        n_reads[path] = get_bam_reads(handle)
        handle.close()

    # --------- Start splitting --------- #
    logger.info("Starting splitting of bams...")
    if parallel:
        # ---------- parallel splitting ---------- #

        # create path for output files
        out_paths = {}
        output_files = []
        for cluster in clusters:
            # replace special characters in filename with "_" https://stackoverflow.com/a/27647173
            save_cluster_name = re.sub(r'[\\/*?:"<>| ]', '_', cluster)
            out_paths[cluster] = f"{output_prefix}{save_cluster_name}.bam"
            output_files.append(f"{output_prefix}{save_cluster_name}.bam")

        # ---- Setup pools and queues ---- #
        # setup pools
        reader_pool = mp.Pool(reader_threads)
        writer_pool = mp.Pool(writer_threads)

        # setup queues to forward data between processes
        manager = mp.Manager()

        # Make chunks of writer_threads
        cluster_chunks = utils.general.split_list(clusters, writer_threads)
        cluster_queues = {}
        for chunk in cluster_chunks:
            m = manager.Queue(maxsize=max_queue_size)  # setup queue to get reads from readers
            for cluster in chunk:
                cluster_queues[cluster] = m  # manager is shared between different clusters if writer_threads < number of clusters

        # Queue for progress bar
        progress_queue = manager.Queue()

        # ---- Start process ---- #
        # start reading bams and add reads into respective cluster queue
        reader_jobs = []
        for i, bam in enumerate(bams):
            reader_jobs.append(reader_pool.apply_async(_buffered_reader, (bam, cluster_queues, barcode2cluster, read_tag, progress_queue, buffer_size)))
        reader_pool.close()  # no more tasks to add

        # write reads to files; one process per file
        writer_jobs = []
        for chunk in cluster_chunks:
            queue = cluster_queues[chunk[0]]  # same queue for the whole chunk
            path_dict = {cluster: out_paths[cluster] for cluster in chunk}  # subset of out_paths specific for this writer
            writer_jobs.append(writer_pool.apply_async(_writer, (queue, path_dict, str(template.header), progress_queue, pysam_threads)))
        writer_pool.close()

        # ---- End process + cleanup ---- #

        # wait for readers and writers to finish
        _monitor_progress(progress_queue, cluster_queues, reader_jobs, writer_jobs, n_reads, individual_pbars)  # does not return until all readers/writers are finished
        reader_pool.join()
        writer_pool.join()

    else:
        # ---------- sequential splitting ---------- #
        # open output bam files
        handles = {}
        output_files = []
        for cluster in clusters:
            # replace special characters in filename with "_" https://stackoverflow.com/a/27647173
            save_cluster_name = re.sub(r'[\\/*?:"<>| ]', '_', cluster)
            f_out = f"{output_prefix}{save_cluster_name}.bam"
            handles[cluster] = open_bam(f_out, "wb", template=template, threads=pysam_threads, verbosity=0)

            output_files.append(f_out)

        # Loop over bamfile(s)
        for i, bam in enumerate(bams):
            logger.info(f"Looping over reads from {bam} ({i+1}/{len(bams)})")

            bam_obj = open_bam(bam, "rb", verbosity=0)

            # Update progress based on total number of reads
            total = get_bam_reads(bam_obj)
            pbar = tqdm(total=total)
            step = int(total / 10000)  # 10000 total updates

            i = 0
            written = 0
            for read in bam_obj:
                i += 1
                if read.has_tag(read_tag):
                    bc = read.get_tag(read_tag)
                else:
                    bc = None

                # Update step manually - there is an overhead to update per read with hundreds of million reads
                if i == step:
                    pbar.update(step)
                    i = 0

                if bc in barcode2cluster:
                    cluster = barcode2cluster[bc]
                    handles[cluster].write(read)
                    written += 1

            # close progressbar
            pbar.close()

            logger.info(f"Wrote {written} reads to cluster files")

        # Close all files
        for handle in handles.values():
            handle.close()

    # ---------- post split functionality ---------- #
    # sort reads
    if sort_bams:
        logger.info("Sorting output bams...")
        for file in tqdm(output_files, desc="Sorting reads", unit="files"):
            temp_file = file + ".tmp"  # temporary sort file
            pysam.sort("-o", temp_file, file)
            os.rename(temp_file, file)

    # index files
    if index_bams:
        logger.info("Indexing output bams...")
        for file in tqdm(output_files, desc="Indexing", unit="files"):
            pysam.index(file, "-@", str(pysam_threads))

    logger.info("Finished splitting bams!")


# ------------------------------------------------------------------ #
# ---------------------- bam helper functions ---------------------- #
# ------------------------------------------------------------------ #

@beartype
def open_bam(file: str,
             mode: str,
             verbosity: Literal[0, 1, 2, 3] = 3, **kwargs: Any) -> "pysam.AlignmentFile":
    """
    Open bam file with pysam.AlignmentFile. On a specific verbosity level.

    Parameters
    ----------
    file : str
        Path to bam file.
    mode : str
        Mode to open the file in. See pysam.AlignmentFile
    verbosity : Literal[0, 1, 2, 3], default 3
        Set verbosity level. Verbosity level 0 for no messages.
    **kwargs : Any
        Forwarded to pysam.AlignmentFile

    Returns
    -------
    pysam.AlignmentFile
        Object to work on SAM/BAM files.
    """

    # check then load modules
    utils.checker.check_module("pysam")
    import pysam

    # save verbosity, then set temporary one
    former_verbosity = pysam.get_verbosity()
    pysam.set_verbosity(verbosity)

    # open file
    handle = pysam.AlignmentFile(file, mode, **kwargs)

    # return to former verbosity
    pysam.set_verbosity(former_verbosity)

    return handle


@beartype
def get_bam_reads(bam_obj: "pysam.AlignmentFile") -> int:
    """
    Get the number of reads from an open pysam.AlignmentFile.

    Parameters
    ----------
    bam_obj : pysam.AlignmentFile
        An open pysam.AlignmentFile object to get the number of reads from.

    Returns
    -------
    int
        number of reads in the bam file
    """

    # check then load modules
    utils.checker.check_module("pysam")
    import pysam

    # Get number of reads in bam
    try:
        total = bam_obj.mapped + bam_obj.unmapped
    except ValueError:  # fall back to "samtools view -c file" if bam_obj.mapped is not available
        path = bam_obj.filename
        total = int(pysam.view("-c", path))

    return total


# ------------------------------------------------------------------ #
# -------------------- multiprocessing functions ------------------- #
# ------------------------------------------------------------------ #


@beartype
def _monitor_progress(progress_queue: Any,
                      cluster_queues: Any,
                      reader_jobs: Any,
                      writer_jobs: Any,
                      total_reads: dict[str, int],
                      individual_pbars: bool = False) -> int:
    """
    Monitor read/write progress of split_bam_clusters.

    Parameters
    ----------
    progress_queue : multiprocessing.queues.Queue
        Queue to send progress updates to.
    cluster_queues : dict[str, multiprocessing.queues.Queue]
        Dictionary of queues which reads were sent to.
    reader_jobs : list[multiprocessing.pool.AsyncResult]
        List of reader jobs.
    writer_jobs : list[multiprocessing.pool.AsyncResult]
        List of writer jobs.
    total_reads : dict[str, int]
        Dictionary containing total reads for each input .bam-file. Keys are filenames and values are total reads.
    individual_pbars : bool, default False
        If True, show progress bar for each cluster. If False, show one progress bar for all clusters.

    Returns
    -------
    int
        Returns 0 on success.
    """

    # Check parameter that beartype cannot cover
    utils.checker.check_type(progress_queue, "progress_queue", BaseProxy)
    for value in cluster_queues.values():
        utils.checker.check_type(value, "cluster_queues value", BaseProxy)
    for jobs in reader_jobs:
        utils.checker.check_type(jobs, "reader_jobs", ApplyResult)
    for jobs in writer_jobs:
        utils.checker.check_type(jobs, "writer_jobs", ApplyResult)

    if utils.jupyter._is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    cluster_names = cluster_queues.keys()

    print(' ', end='', flush=True)  # hack for making progress bars work in notebooks; https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308

    # Initialize progress bars dependent on individual_pbars true/false
    pbars = {}
    if individual_pbars is True:  # individual progress per reader/writer

        # readers
        for i, name in enumerate(total_reads):
            pbars[name] = tqdm(total=total_reads[name], position=i, desc=f"Reading ({name})", unit="reads")

        # writers
        for j, cluster in enumerate(cluster_names):
            pbars[cluster] = tqdm(total=1, position=i + j, desc=f"Writing queued reads ({cluster})")  # total is 1 instead of 0 to trigger progress bar
            pbars[cluster].total = 0  # reset to 0 (bar is still shown)
            pbars[cluster].refresh()

    else:   # merged progress for reading and writing

        # readers
        sum_reads = sum(total_reads.values())
        pbar = tqdm(total=sum_reads, position=0, desc="Reading from bams", unit="reads")
        for name in total_reads:
            pbars[name] = pbar  # all names share the same pbar

        # writers
        pbar = tqdm(total=1, position=1, desc="Writing queued reads", unit="reads")
        pbar.total = 0  # reset to 0 (bar is still shown)
        pbar.refresh()
        for cluster in cluster_names:
            pbars[cluster] = pbar  # all clusters share the same pbar

    # Fetch progress from readers/writers
    writers_running = len(writer_jobs)
    reading_done = False
    while True:

        task, name, value = progress_queue.get()
        pbar = pbars[name]

        # update pbar depending on the task
        if task == "read":
            pbar.update(value)
            pbar.refresh()

        elif task == "sent":
            pbar.total += value
            pbar.refresh()

        elif task == "written":
            pbar.update(value)
            pbar.refresh()

        elif task == "done":
            writers_running -= 1

        # Check if all reads were read by comparing to pbar total
        reader_pbars = [pbars[name] for name in total_reads]
        if reading_done is False and (sum([pbar.total for pbar in reader_pbars]) >= sum([pbar.n for pbar in reader_pbars])):
            reading_done = True
            _ = [reader_job.get() for reader_job in reader_jobs]  # wait for all writers to finish

            for queue in cluster_queues.values():
                queue.put((None, None))  # Insert None into queue to signal end of reads

        # Check if all writers are done
        if writers_running == 0:
            _ = [writer_job.get() for writer_job in writer_jobs]  # wait for all readers to finish
            break

    return 0  # success


# @beartype beartype seems to not work with this function
def _buffered_reader(path: str,
                     out_queues: dict[str | int, Any],
                     bc2cluster: dict[str | int, str | int],
                     tag: str,
                     progress_queue: Any,
                     buffer_size: str = 10000) -> int:
    """
    Open bam file and add reads to respective output queue.

    Parameters
    ----------
    path : str
        Path to bam file.
    out_queues : dict[str | int, multiprocessing.queues.Queue]
        Dict of multiprocesssing.Queues with cluster as key.
    bc2cluster : dict[str | int, str | int]
        Dict of clusters with barcode as key.
    tag : str
        Read tag that should be used for queue assignment.
    progress_queue : multiprocessing.queues.Queue
        Queue to send progress updates to.
    buffer_size : int, default 10000
        Size of buffer (number of reads) for each queue to collect before writing.

    Returns
    -------
    int
        Returns 0 on success.

    Raises
    ------
    Exception
        If buffered reader failes.
    """

    # Test parameter types not covered by beartype
    for value in out_queues.values():
        utils.checker.check_type(value, "out_queues.values", BaseProxy)
    utils.checker.check_type(progress_queue, "progress_queue", BaseProxy)

    try:
        # open bam
        bam = open_bam(path, "rb", verbosity=0)

        # Setup read buffer per cluster
        read_buffer = {cluster: [] for cluster in set(bc2cluster.values())}

        # put each read into correct queue
        step = 100000  # progress every in hundred thousand reads
        n_reads_step = 0  # count of reads read from bam per step
        for read in bam:

            # get barcode
            if read.has_tag(tag):
                bc = read.get_tag(tag)
            else:
                bc = None

            # put read into matching cluster queue
            if bc in bc2cluster:
                cluster = bc2cluster[bc]
                read_buffer[cluster].append(read.to_string())

                # Send reads to buffer when buffer size is reached
                if len(read_buffer[cluster]) == buffer_size:
                    out_queues[cluster].put((cluster, read_buffer[cluster]))  # send the tuple of (clustername, read_buffer) to queue
                    progress_queue.put(("sent", cluster, len(read_buffer[cluster])))
                    read_buffer[cluster] = []

            # update progress bar for reading
            n_reads_step += 1
            if n_reads_step == step:
                progress_queue.put(("read", path, n_reads_step))
                n_reads_step = 0

        # all reads have been read; add remaining reads to progress
        progress_queue.put(("read", path, n_reads_step))

        # Send remaining reads to buffer
        for cluster in read_buffer:
            if len(read_buffer[cluster]) > 0:
                out_queues[cluster].put((cluster, read_buffer[cluster]))
                progress_queue.put(("sent", cluster, len(read_buffer[cluster])))

        return 0  # success

    except Exception as e:
        print(e.message)
        raise e


@beartype
def _writer(read_queue: Any,
            out_paths: dict[str | int, str],
            bam_header: str,
            progress_queue: Any,
            pysam_threads: int = 4) -> int:
    """
    Write reads to given file.

    Parameters
    ----------
    read_queue : multiprocessing.queues.Queue
        Queue of reads to be written into file.
    out_paths : dict
        Path to output files for this writer. In the format {cluster1: <path>, cluster2: <path>}
    bam_header : str
        pysam.AlignmentHeader used as template for output bam.
    progress_queue : multiprocessing.queues.Queue
        Queue to send progress updates to.
    pysam_threads : int, default 4
        Number of threads for pysam to use for writing. This is different from the threads used for the individual writers.

    Returns
    -------
    int
        Returns 0 on success.

    Raises
    ------
    Exception
        If buffered reader failes.
    """

    # Check parameter that are not covered by beartype
    utils.checker.check_type(read_queue, "read_queue", BaseProxy)
    utils.checker.check_type(progress_queue, "progress_queue", BaseProxy)

    try:
        import pysam  # install of pysam was checked in parent function

        cluster_names = list(out_paths.keys())

        # open bam files for writing cluster reads
        try:
            handles = {}
            for cluster in cluster_names:
                handles[cluster] = pysam.AlignmentFile(out_paths[cluster], "wb", text=bam_header, threads=pysam_threads)
        except Exception as e:
            print(e.message)
            raise e

        # fetch reads from queue and write to bam
        while True:
            cluster, read_lst = read_queue.get()

            # stop writing
            if cluster is None:
                break

            # write reads to file
            handle = handles[cluster]  # the handle for this cluster
            for read in read_lst:

                # create read object and write to handle
                read = pysam.AlignedSegment.fromstring(read, handle.header)
                handle.write(read)

            # Send written to progress
            progress_queue.put(("written", cluster, len(read_lst)))

        # Close files after use
        for handle in handles.values():
            handle.close()

        # Tell monitoring queue that this writing job is done
        progress_queue.put(("done", cluster_names[0], None))  # cluster is not used here, but is needed for the progress queue

        return 0  # success

    except Exception as e:
        print(e.message)
        raise e


@beartype
def bam_to_bigwig(bam: str,
                  output: Optional[str] = None,
                  scale: bool = True,
                  overwrite: bool = True,
                  tempdir: str = ".",
                  remove_temp: bool = True,
                  bedtools_path: Optional[str] = None,
                  bgtobw_path: Optional[str] = None) -> str:
    """
    Convert reads in a bam-file to bigwig format.

    Parameters
    ----------
    bam : str
        Path to bam file.
    output : Optional[str], default None
        Path to output file. If None, output is written to same directory as bam file with same name and .bw extension.
    scale : bool, default True
        Scale output depth to reads per million mapped reads.
    overwrite : bool, default True
        Overwrite output file if it already exists.
    tempdir : str, default "."
        Path to directory where temporary files are written.
    remove_temp : bool, default True
        Remove temporary files after conversion.
    bedtools_path : Optional[str], default None
        Path to bedtools binary. If None, the function will search for the binary in the path.
    bgtobw_path : Optional[str], default None
        Path to bedGraphToBigWig binary. If None, the function will search for the binary in the path.

    Returns
    -------
    str
        Path to output file.
    """

    # Set output name and check if bigwig already exists
    if output is None:
        output = bam.replace(".bam", ".bw")

    if os.path.exists(output) and overwrite is False:
        logger.warning("Output file already exists. Set overwrite=True to overwrite.")
        return output

    # Check required modules
    utils.checker.check_module("pysam")
    import pysam

    if bedtools_path is None:
        bedtools_path = utils.general.get_binary_path("bedtools")

    if bgtobw_path is None:
        bgtobw_path = utils.general.get_binary_path("bedGraphToBigWig")

    # Get size of genome and write a chromsizes file
    bamobj = pysam.AlignmentFile(bam, "rb")
    chromsizes = {chrom: bamobj.lengths[i] for i, chrom in enumerate(bamobj.references)}
    chromsizes_file = utils.io.get_temporary_filename(tempdir)

    with open(chromsizes_file, "w") as f:
        for chrom, size in chromsizes.items():
            f.write(f"{chrom}\t{size}\n")

    # Get number of mapped reads in file for normalization
    logger.info("Getting scaling factor")
    scaling_factor = 0
    if scale:
        n_reads = get_bam_reads(bamobj)
        scaling_factor = 1 / (n_reads / 1e6)
        scaling_factor = round(scaling_factor, 5)
    bamobj.close()

    # Convert bam to bedgraph
    bedgraph_out = utils.io.get_temporary_filename(tempdir)
    cmd = f"{bedtools_path} genomecov -bg -ibam {bam} > {bedgraph_out}"
    logger.info("Running: " + cmd)
    utils.general.run_cmd(cmd)

    # Sort and scale input
    bedgraph_out_sorted = utils.io.get_temporary_filename(tempdir)
    cmd = f"sort -k1,1 -k2,2n -T {tempdir} {bedgraph_out} |  awk '{{$4=$4*{scaling_factor}; print $0}}' > {bedgraph_out_sorted}"
    logger.info("Running: " + cmd)
    utils.general.run_cmd(cmd)

    # Convert bedgraph to bigwig
    cmd = f"{bgtobw_path} {bedgraph_out_sorted} {chromsizes_file} {output}"
    logger.info("Running: " + cmd)
    utils.general.run_cmd(cmd)

    # Remove all temp files
    if remove_temp is True:
        utils.io.remove_files([chromsizes_file, bedgraph_out, bedgraph_out_sorted])

    logger.info(f"Finished converting bam to bigwig! Output bigwig is found in: {output}")

    return output


# ---------------------------------------------------------------------------------- #
# ------------------------------ Bam to fragments file ----------------------------- #
# ---------------------------------------------------------------------------------- #

@beartype
def create_fragment_file(bam: str,
                         barcode_tag: Optional[str] = 'CB',
                         barcode_regex: Optional[str] = None,
                         outdir: Optional[str] = None,
                         nproc: Optional[int] = 1,
                         index: bool = False,
                         min_dist: int = 10,
                         max_dist: int = 5000,
                         include_clipped: bool = True,
                         shift_plus: int = 5,
                         shift_minus: int = -4,
                         keep_temp: bool = False) -> str:
    """
    Create fragments file out of a BAM file.

    This is an alternative to using the sinto package,
    which is slow and has issues with large bam-files.

    Parameters
    ----------
    bam : str
        Path to .bam file.
    barcode_tag : Optional[str], default 'CB'
        The tag where cell barcodes are saved in the bam file. Set to None if the barcodes are in read names.
    barcode_regex : Optional[str], default None
        Regex to extract barcodes from read names. Set to None if barcodes are stored in a tag.
    outdir : Optional[str], default None
        Path to save fragments file. If None, the file will be saved in the same folder as the .bam file. Temporary intermediate files are also written to this directory.
    nproc : Optional[int], default 1
        Number of threads for parallelization. Set None to use settings.get_threads.
    index : bool, default False
        If True, index fragments file. Requires bgzip and tabix.
    min_dist : int, default 10
        Minimum fragment length to consider.
    max_dist : int, default 5000
        Maximum fragment length to consider.
    include_clipped : bool, default True
        Whether to include soft-clipped bases in the fragment length. If True, the full length between reads is used. If False, the fragment length will be calculated from the aligned parts of the reads.
    shift_plus : int, default 5
        Shift the start position of the forward read by this value (standard for ATAC).
    shift_minus : int, default -4
        Shift the end position of the reverse read by this value (standard for ATAC).
    keep_temp : bool, default False
        If True, keep temporary files.

    Returns
    -------
    str
        Path to fragments file.

    Raises
    ------
    ValueError
        If both barcode_tag and barcode_regex (or neither) are set.
    FileNotFoundError
        If the input bam file does not exist.
    Exception
        On unkown error while sorting .bam
    """

    utils.checker.check_module("pysam")
    import pysam

    if nproc is None:
        nproc = settings.get_threads()

    # Establish output filename
    if not outdir:
        outdir = os.path.dirname(bam)
    os.makedirs(outdir, exist_ok=True)  # create output directory if it does not exist
    out_prefix = os.path.join(outdir, os.path.basename(bam).split('.')[0])
    outfile = out_prefix + '_fragments.tsv'  # final output file

    # Check input tag/regex
    if barcode_tag and barcode_regex:
        raise ValueError("Only one of 'barcode_tag' and 'barcode_regex' can be provided! Please set one of the parameters to None.")
    elif not barcode_tag and not barcode_regex:
        raise ValueError("Either 'barcode_tag' or 'barcode_regex' must be provided!")

    # Check if bam file exists
    if not os.path.exists(bam):
        raise FileNotFoundError(f"{bam} does not exist!")

    # Ensure that bam-file is indexed
    if not os.path.exists(bam + ".bai"):

        index_success = False
        sort_success = False
        while not index_success:
            if not sort_success:
                logger.warning(".bam-file has no index - trying to index .bam-file with pysam...")

            try:
                pysam.index(bam)
                index_success = True

            except Exception as e:  # try to sort the bam file before indexing again

                if sort_success:  # if we already sorted the file
                    logger.error("Unknown error when indexing bam file after sort - please see the error message.")
                    raise e

                try:
                    logger.warning(".bam-file could not be indexed with pysam - bam might be unsorted. Trying to sort .bam-file using pysam...")
                    bam_sorted = os.path.join(outdir, os.path.basename(bam).split(".")[0] + "_sorted.bam")
                    pysam.sort("-o", bam_sorted, bam)
                    bam = bam_sorted   # try to index again with this bam
                    sort_success = True
                except Exception as e:
                    logger.error("Unknown error when sorting .bam-file - please see the error message.")
                    raise e

        logger.info(".bam-file was successfully indexed.")

    # Chunk chromosomes
    bamfile = pysam.AlignmentFile(bam)
    chromosomes = bamfile.references
    bamfile.close()
    chromosome_chunks = utils.general.split_list(chromosomes, nproc)

    # Create fragments from bam
    logger.info("Creating fragments from chromosomes...")
    temp_files = []
    if nproc == 1:
        chunk = chromosome_chunks[0]   # only one chunk
        temp_file = out_prefix + f"_fragments_{chunk[0]}_{chunk[-1]}.tmp"
        temp_files = [temp_file]
        n_written = _write_fragments(bam, chromosomes, temp_file, barcode_tag=barcode_tag, barcode_regex=barcode_regex,
                                     min_dist=min_dist, max_dist=max_dist, include_clipped=include_clipped, shift_plus=shift_plus, shift_minus=shift_minus)

    else:

        pool = mp.Pool(nproc)
        jobs = []
        for chunk in chromosome_chunks:
            logger.debug("Starting job for chromosomes: " + ", ".join(chunk))
            temp_file = out_prefix + f"_fragments_{chunk[0]}_{chunk[-1]}.tmp"
            temp_files.append(temp_file)
            job = pool.apply_async(_write_fragments, args=(bam, chunk, temp_file, barcode_tag, barcode_regex, min_dist, max_dist, include_clipped, shift_plus, shift_minus))
            jobs.append(job)

        pool.close()

        # monitor progress
        utils.multiprocessing.monitor_jobs(jobs, description="Progress")  # waits for all jobs to finish
        pool.join()

        # get number of written fragments
        n_written = sum([job.get() for job in jobs])

    # check if any fragments were written
    if n_written == 0:
        logger.warning("No fragments were written to file - please check your input files and parameters.")
    else:
        logger.info(f"Found a total of {n_written} valid fragments in the .bam file.")

    # Merge temp files
    logger.info("Merging identical fragments...")
    tempdir = f"{outdir}"
    os.makedirs(tempdir, exist_ok=True)  # create temp directory if it does not exist
    cmd = f"cat {' '.join(temp_files)} | sort -T {tempdir} -k1,1 -k2,2n -k3,3n -k4 "
    cmd += "| uniq -c | awk -v OFS='\\t' '{$(NF+1)=$1;$1=\"\"}1' "  # count number of fragments at position, move count to last column
    cmd += "| awk '{gsub(/^[ \\t]+/, \"\");}1' "  # remove leading whitespace
    cmd += f"> {outfile}"

    logger.debug(f"Running command: \"{cmd}\"")
    os.system(cmd)
    logger.info(f"Successfully created fragments file: {outfile}")

    # Remove temp files
    if not keep_temp:
        utils.io.rm_tmp(temp_files=temp_files)

    # Index fragments file
    if index:
        cmd = f"bgzip --stdout {outfile} > {outfile}.gz"
        logger.info(f"Running command: \"{cmd}\"")
        os.system(cmd)

        cmd = f"tabix -p bed {outfile}.gz"
        logger.info(f"Running command: \"{cmd}\"")
        os.system(cmd)

    # return path to sorted fragments file
    return outfile


@beartype
def _get_barcode_from_readname(read: "pysam.AlignedSegment", regex: str) -> str:
    """Extract barcode from read name.

    Parameters
    ----------
    read : pysam.AlignedSegment
        Read from BAM file.
    regex : str
        Regex to extract barcode from read name.

    Returns
    -------
    str
        Barcode.
    """

    match = re.search(regex, read.query_name)
    if match:
        return match.group(0)
    else:
        return None


@beartype
def _get_barcode_from_tag(read: "pysam.AlignedSegment", tag: str) -> str:
    """Extract barcode from read tag.

    Parameters
    ----------
    read : pysam.AlignedSegment
        Read from BAM file.
    tag : str
        Tag where barcode is stored.

    Returns
    -------
    str
        Barcode.
    """

    if read.has_tag(tag):
        return read.get_tag(tag)
    else:
        return None


@beartype
def _write_fragments(bam: str,
                     chromosomes: Sequence[str],
                     outfile: str,
                     barcode_tag: Optional[str] = "CB",
                     barcode_regex: Optional[str] = None,
                     min_dist: int = 10,
                     max_dist: int = 5000,
                     include_clipped: bool = True,
                     shift_plus: int = 5,
                     shift_minus: int = -4) -> int:
    """Write fragments from a bam-file within a list of chromosomes to a text file.

    Parameters
    ----------
    bam : str
        Path to .bam file.
    chromosomes : Sequence[str]
        List of chromosomes to fetch from bam file.
    outfile : str
        Path to output file.
    barcode_tag : Optional[str], default 'CB'
        The tag where cell barcodes are saved in the bam file. Set to None if the barcodes are in read names.
    barcode_regex : Optional[str], default None
        Regex to extract barcodes from read names. Set to None if barcodes are stored in a tag.
    min_dist : int, default 10
        Minimum fragment length to consider.
    max_dist : int, default 5000
        Maximum fragment length to consider.
    include_clipped : boolean, default True
        Whether to include soft-clipped bases in the fragment length. If True, the full length between reads is used. If False, the fragment length will be calculated from the aligned parts of the reads.
    shift_plus : int, default 5
        Shift the start position of the forward read by this value (standard for ATAC).
    shift_minus : int, default -4
        Shift the end position of the reverse read by this value (standard for ATAC).

    Returns
    -------
    int
        Number of fragments written to file.

    Raises
    ------
    ValueError
        If None of barcode_tag and barcode_regex are set.
    """

    utils.checker.check_module("pysam")
    import pysam

    # Establish function to use for getting barcode
    if barcode_tag:
        get_barcode = partial(_get_barcode_from_tag, tag=barcode_tag)
    elif barcode_regex:
        get_barcode = partial(_get_barcode_from_readname, regex=barcode_regex)
    else:
        raise ValueError("Either barcode_tag or barcode_regex must be provided!")

    fragments = {}
    n_written = 0
    with open(outfile, "w") as out_txt:
        with pysam.AlignmentFile(bam, "rb") as bamfile:
            for chromosome in chromosomes:
                for read in bamfile.fetch(chromosome):  # fetch all reads on this chromosome

                    fragment_name = read.query_name
                    if read.is_mapped and read.is_paired:  # reads must be mapped and paired
                        if fragment_name not in fragments:
                            fragments[fragment_name] = read   # first mate found; store it in the dictionary

                        else:  # The mate was already found, so we can write the fragment

                            # Estimate fragment length
                            reads = [fragments[fragment_name], read] if read.is_reverse else [read, fragments[fragment_name]]  # put forward read first
                            n_reverse = sum([read.is_reverse for read in reads])  # one of the reads must be reverse for it to be a proper forward-reverse pair
                            fragment_start = reads[0].reference_start + shift_plus  # forward read; count whole read without soft-clipping
                            fragment_end = reads[1].reference_end + shift_minus  # reverse read; count whole read without soft-clipping; shift_minus is negative

                            if include_clipped is True:
                                fragment_start -= reads[0].query_alignment_start
                                fragment_end += reads[1].query_length - reads[1].query_alignment_end  # the difference between length and end position gives the number of clipped bases for reverse read
                            fragment_length = fragment_end - fragment_start

                            # If the fragment is valid; we can save it
                            if n_reverse == 1 and fragment_length >= min_dist and fragment_length <= max_dist:

                                # Get barcode
                                barcode = get_barcode(read)

                                if barcode:
                                    # Write fragment to file
                                    out_txt.write(f"{read.reference_name}\t{fragment_start}\t{fragment_end}\t{barcode}\n")
                                    n_written += 1

                            fragments.pop(fragment_name)  # Remove the written fragment from the dictionary

                # Remove all remaining fragments from the dictionary; new chromosome or end of file
                fragments = {}

    return n_written
