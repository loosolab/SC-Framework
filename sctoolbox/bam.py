""" Functionality to split bam files into smaller bam files based on clustering in adata.obs """

import re
import os
import multiprocessing
import sctoolbox.utilities as utils
import sctoolbox.annotation as anno
import pandas as pd
import gzip


def split_bam_clusters(adata,
                       bams,
                       groupby,
                       barcode_col=None,
                       read_tag="CB",
                       output_prefix="split_",
                       reader_threads=1,
                       writer_threads=1,
                       parallel=False,
                       pysam_threads=4,
                       buffer_size=10000,
                       max_queue_size=1000,
                       individual_pbars=False,
                       sort_bams=False,
                       index_bams=False):
    """
    Split BAM files into clusters based on 'groupby' from the anndata.obs table.

    Parameters
    ----------
    adata : anndata.Anndata
        Annotated data matrix containing clustered cells in .obs.
    bams : str or list of str
        One or more BAM files to split into clusters
    groupby : str
        Name of a column in adata.obs to cluster the pseudobulks by.
    barcode_col : str, default None
        Name of a column in adata.obs to use as barcodes. If None, use the index of .obs.
    read_tag : str, default "CB"
        Tag to use to identify the reads to split. Must match the barcodes of the barcode_col.
    output_prefix : str, default "split_"
        Prefix to use for the output files.
    reader_threads : int, default 1
        Number of threads to use for reading.
    writer_threads : int, default 1,
        Number of threads to use for writing.
    parallel : boolean, default False
        Whether to enable parallel processsing.
    buffer_size : int, default 10000
        The size of the buffer between readers and writers.
    max_queue_size : int, default 1000
        The maximum size of the queue between readers and writers.
    individual_pbars : boolean, default False
        Whether to show a progress bar for each individual BAM file and output clusters. Default: False (overall progress bars).
    sort_bams : boolean, default False
        Sort reads in each output bam
    index_bams : boolean, default False
        Create an index file for each output bam. Will throw an error if `sort_bams` is False.
    """
    # check then load modules
    utils.check_module("tqdm")
    if utils._is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    utils.check_module("pysam")
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
    utils.create_dir(output_prefix)

    # Establish clusters from obs
    clusters = list(set(adata.obs[groupby]))
    print(f"Found {len(clusters)} groups in .obs.{groupby}: {clusters}")

    if writer_threads > len(clusters):
        print(f"The number of writers ({writer_threads}) is larger than the number of output clusters ({len(clusters)}). Limiting writer_threads to the number of clusters.")
        writer_threads = len(clusters)

    # setup barcode <-> cluster dict
    if barcode_col is None:
        barcode2cluster = dict(zip(adata.obs.index.tolist(), adata.obs[groupby]))
    else:
        barcode2cluster = dict(zip(adata.obs[barcode_col], adata.obs[groupby]))

    # create template used for bam header
    template = open_bam(bams[0], "rb", verbosity=0)

    # Get number in reads in input bam(s)
    print("Reading total number of reads from bams...")
    n_reads = {}
    for path in bams:
        handle = open_bam(path, "rb", verbosity=0)
        n_reads[path] = get_bam_reads(handle)
        handle.close()

    # --------- Start splitting --------- #
    print("Starting splitting of bams...")
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
        reader_pool = multiprocessing.Pool(reader_threads)
        writer_pool = multiprocessing.Pool(writer_threads)

        # setup queues to forward data between processes
        manager = multiprocessing.Manager()

        # Make chunks of writer_threads
        cluster_chunks = utils.split_list(clusters, writer_threads)
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
            print(f"Looping over reads from {bam} ({i+1}/{len(bams)})")

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

            print(f"Wrote {written} reads to cluster files")

        # Close all files
        for handle in handles.values():
            handle.close()

    # ---------- post split functionality ---------- #
    # sort reads
    if sort_bams:
        print("Sorting output bams...")
        for file in tqdm(output_files, desc="Sorting reads", unit="files"):
            temp_file = file + ".tmp"  # temporary sort file
            pysam.sort("-o", temp_file, file)
            os.rename(temp_file, file)

    # index files
    if index_bams:
        print("Indexing output bams...")
        for file in tqdm(output_files, desc="Indexing", unit="files"):
            pysam.index(file, "-@", str(pysam_threads))

    print("Finished splitting bams!")


# ------------------------------------------------------------------ #
# ---------------------- bam helper functions ---------------------- #
# ------------------------------------------------------------------ #

def open_bam(file, mode, verbosity=3, **kwargs):
    """
    Open bam file with pysam.AlignmentFile. On a specific verbosity level.

    Parameters
    ----------
    file : str
        Path to bam file.
    mode : str
        Mode to open the file in. See pysam.AlignmentFile
    verbosity : int, default 3
        Set verbosity level. Verbosity level 0 for no messages.
    **kwargs :
        Forwarded to pysam.AlignmentFile

    Returns
    -------
    pysam.AlignmentFile :
        Object to work on SAM/BAM files.
    """
    # check then load modules
    utils.check_module("pysam")
    import pysam

    # save verbosity, then set temporary one
    former_verbosity = pysam.get_verbosity()
    pysam.set_verbosity(verbosity)

    # open file
    handle = pysam.AlignmentFile(file, mode, **kwargs)

    # return to former verbosity
    pysam.set_verbosity(former_verbosity)

    return handle


def get_bam_reads(bam_obj):
    """
    Get the number of reads from an open pysam.AlignmentFile

    Parameters
    ----------
    bam_obj : pysam.AlignmentFile
        An open pysam.AlignmentFile object to get the number of reads from.

    Returns
    -------
    int :
        number of reads in the bam file
    """
    # check then load modules
    utils.check_module("pysam")
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

def _monitor_progress(progress_queue,
                      cluster_queues,
                      reader_jobs,
                      writer_jobs,
                      total_reads,
                      individual_pbars=False):
    """
    Function for monitoring read/write progress of split_bam_clusters.

    Parameters
    ----------
    progress_queue : multiprocessing.Queue
        Queue to send progress updates to.
    cluster_queues : dict of multiprocessing.Queue
        Dictionary of queues which reads were sent to.
    reader_jobs : list of multiprocessing.pool.AsyncResult
        List of reader jobs.
    writer_jobs : list of multiprocessing.pool.AsyncResult
        List of writer jobs.
    total_reads : dict
        Dictionary containing total reads for each input .bam-file. Keys are filenames and values are total reads.
    individual_pbars : bool, default False
        If True, show progress bar for each cluster. If False, show one progress bar for all clusters.

    Returns
    -------
    int :
        Returns 0 on success.
    """
    if utils._is_notebook() is True:
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

        elif task == "sent":
            pbar.total += value
            pbar.refresh()

        elif task == "written":
            pbar.update(value)

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


def _buffered_reader(path, out_queues, bc2cluster, tag, progress_queue, buffer_size=10000):
    """
    Open bam file and add reads to respective output queue.

    Parameters
    ----------
    path : str
        Path to bam file.
    read_num : int
        Number of reads per chunk.
    out_queue : dict
        Dict of multiprocesssing.Queues with cluster as key
    bc2cluster : dict
        Dict of clusters with barcode as key.
    tag : str
        Read tag that should be used for queue assignment.
    progress_queue : multiprocessing.Queue
        Queue to send progress updates to.
    buffer_size : int, default 10000
        Size of buffer (number of reads) for each queue to collect before writing.

    Returns
    -------
    int :
        Returns 0 on success.
    """
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


def _writer(read_queue, out_paths, bam_header, progress_queue, pysam_threads=4):
    """
    Write reads to given file.

    Parameters
    ----------
    read_queue : multiprocessing.Queue
        Queue of reads to be written into file.
    out_paths : dict
        Path to output files for this writer. In the format {cluster1: <path>, cluster2: <path>}
    bam_header : str(pysam.AlignmentHeader)
        Used as template for output bam.
    progress_queue : multiprocessing.Queue
        Queue to send progress updates to.
    pysam_threads : int, default 4
        Number of threads for pysam to use for writing. This is different from the threads used for the individual writers.

    Returns
    -------
    int :
        Returns 0 on success.
    """
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


def add_insertsize(adata,
                   bam=None,
                   fragments=None,
                   barcode_tag="CB",
                   threads=1):
    """ 
    Add information on insertsize to the adata object using either a .bam-file or a fragments file.

    Parameters
    ----------
    adata : anndata.AnnData

    barcode_tag : str, default 'CB'
        Tag used for the cell barcode in the bam file.

    Returns
    -------
    None 


    """

    if bam is not None and fragments is not None:
        raise ValueError("Please provide either a bam file or a fragments file - not both.")

    elif bam is not None:
        pass

    elif fragments is not None:
        pass

    else:
        raise ValueError("Please provide either a bam file or a fragments file.")


def insertsize_from_bam(bam,
                        barcode_tag="CB",
                        barcode_list=None,
                        regions='chr1:1-2000000',
                        chunk_size=100000):
    """ Get fragment insertsize distributions per barcode from bam file.

    Parameters
    -----------
    bam : str
        Path to bam file
    barcode_tag : str, default "CB"
        The read tag representing the barcode.
    barcode_list : list, default None
        List of barcodes to include in the analysis. If None, all barcodes are included.
    regions : str or list of str, default 'chr1:1-2000000'
        Regions to include in the analysis. If None, all regions are included.
    chunk_size : int, default 500000
        Size of bp chunks to read from bam file.
    n_jobs : int, default 1000
        Number of jobs to create for multiprocessing. This controls the step size for the progress bar.
    threads : int, default 1
        Number of threads to use for multiprocessing.

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

    # Open bamfile
    print("Opening bam file...")
    if not os.path.exists(bam + ".bai"):
        print("Bamfile has no index - trying to index with pysam...")
        pysam.index(bam)

    bam_obj = open_bam(bam, "rb", require_index=True)
    chromosome_lengths = dict(zip(bam_obj.references, bam_obj.lengths))

    # Create chunked genome regions:
    print(f"Creating chunks of size {chunk_size}bp...")

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
    print(f"Counting insertsizes across {len(regions_split)} chunks...")
    count_dict = {}
    pbar = tqdm(total=len(regions_split), desc="Progress: ", unit="chunks")
    for region in regions_split:
        chrom, start, end = re.split("[:-]", region)
        for read in bam_obj.fetch(chrom, int(start), int(end)):
            try:
                barcode = read.get_tag(barcode_tag)
            except Exception:  # tag was not found
                barcode = "NA"

            if barcode_list is not None:
                if barcode not in barcode_list:
                    continue

            # Add read to dict
            size = abs(read.template_length) - 9  # length of insertion
            count_dict = add_fragment(count_dict, barcode, size)

        # Update progress
        pbar.update(1)

    bam_obj.close()

    # Convert dict to pandas dataframes
    print("Converting counts to dataframe")
    table = pd.DataFrame.from_dict(count_dict, orient="index")
    table = table[["count", "mean_length"] + sorted(table.columns[2:])]
    table["mean_length"] = table["mean_length"].round(2)

    print("Done!")

    return table


def insertsize_from_fragments(fragments):
    """ Get fragment insertsize distributions per barcode from fragments file. 

    Parameters
    -----------
    fragments : str
        Path to fragments.bed(.gz) file.

    Returns
    --------
    pandas.DataFrame
        DataFrame with insertsize distributions per barcode.
    """

    # Open fragments file
    if anno._is_gz_file(fragments):
        f = gzip.open(fragments, "rt")
    else:
        f = open(fragments, "r")

    # Read fragments file and add to dict
    count_dict = {}
    for line in f:
        columns = line.rstrip().split("\t")
        start = int(columns[1])
        end = int(columns[2])
        barcode = columns[3]
        count = int(columns[4])
        size = end - start

        count_dict = add_fragment(count_dict, barcode, size, count)

    f.close()

    # Convert dict to pandas dataframe
    print("Converting counts to dataframe...")
    table = pd.DataFrame.from_dict(count_dict, orient="index")
    table = table[["count", "mean_length"] + sorted(table.columns[2:])]
    table["mean_length"] = table["mean_length"].round(2)

    return table


def add_fragment(count_dict, barcode, size, count=1):
    """ Add fragment of size 'size' to count_dict.

    Parameters
    -----------
    count_dict : dict
        Dictionary containing the counts per insertsize.
    barcode : str
        Barcode of the read.
    size : int
        Insertsize to add to count_dict.
    insertsize_max : int
        Maximum insertsize to collect for distribution.
    count : int, default 1
        Number of reads to add to count_dict.

    Returns
    --------
    count_dict : dict
        Updated count_dict.
    """

    # Initialize if barcode is seen for the first time
    if barcode not in count_dict:
        count_dict[barcode] = {"mean_length": 0, "count": 0}

    # Add read to dict
    if size >= 0 and size <= 1000:  # do not save negative insertsize, and set a cap on the maximum insertsize to limit outlier effects

        count_dict[barcode]["count"] += count

        # Update mean
        mu = count_dict[barcode]["mean_length"]
        total_count = count_dict[barcode]["count"]
        diff = (size - mu) / total_count
        count_dict[barcode]["mean_length"] = mu + diff

        # Save to distribution
        if size not in count_dict[barcode]:  # first time size is seen
            count_dict[barcode][size] = 0
        count_dict[barcode][size] += count

    return count_dict


def plot_insertsize_dist(adata, cells=None):
    """ """

    pass
