from logging.handlers import HTTPHandler
import pandas as pd
import sys
import os
import scanpy as sc
import importlib
import re
import multiprocessing
import time

import sctoolbox.checker as ch
import sctoolbox.creators as cr

import matplotlib.pyplot as plt


def _is_notebook():
    """ Utility to check if function is being run from a notebook or a script """
    try:
        _ = get_ipython()
        return(True)
    except NameError:
        return(False)


def create_dir(path):
    """ Create a directory if it is not existing yet.
    
    Parameters
    -----------
    path : str
        Path to the directory to be created.
    """
    
    dirname = os.path.dirname(path) #the last dir of the path
    os.makedirs(dirname, exist_ok=True)


def is_str_numeric(ans):
    try:
        float(ans)
        return True
    except ValueError:
        return False


def save_figure(path):
    """ Save the current figure to a file.
    
    Parameters
    ----------
    path : str
        Path to the file to be saved.
        Add the extension (e.g. .tiff) you wanna save your figure in the end of path, e.g., /mnt/*/note2_violin.tiff
        The lack of extension indicates the figure will be saved as .png
    """

    if path is not None:

        create_dir(path) #recursively create parent dir if needed
        plt.savefig(path, dpi=600, bbox_inches="tight")


def vprint(verbose=True):
    """ Print the verbose message.
    
    Parameters
    -----------
    verbose : Boolean, optional
        Set to False to disable the verbose message. Default: True
    """

    f = lambda message: print(message) if verbose == True else None

    return f

#Requirement for installed tools
def check_module(module):
    """ Check if <module> can be imported without error.
    
    Parameters
    -----------
    module : str
        Name of the module to check.

    Raises
    ------
    ImportError
        If the module is not available for import.
    """

    error = 0
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        error = 1
    except:
        raise #unexpected error loading module
    
    #Write out error if module was not found
    if error == 1:
        s = f"ERROR: Could not find the '{module}' module on path, but the module is needed for this functionality. Please install this package to proceed."
        raise ImportError(s)


#Loading adata file and adding the information to be evaluated and color list
def load_anndata(is_from_previous_note=True, which_notebook=None, data_to_evaluate=None):
    '''
    Load anndata object
    ==========
    Parameters
    ==========
    is_from_previous_note : Boolean
        Set to False if you wanna load an anndata object from other source rather than scRNAseq autom workflow.
    which_notebook : Int.
        The number of the notebook that generated the anndata object you want to load
        If is_from_previous_note=False, this parameter will be ignored
    data_to_evaluate : String
        This is the anndata.obs[STRING] to be used for analysis, e.g. "condition"
    '''
    #Author : Guilherme Valente
    def loading_adata(NUM):
        pathway=ch.fetch_info_txt()
        files=os.listdir(''.join(pathway))
        loading="anndata_" + str(NUM)
        if any(loading in items for items in files):
            for file in files:
               if loading in file:
                    anndata_file=file
        else: #In case the user provided an inexistent anndata number
            sys.exit(loading + " was not found in " + pathway)
        return(''.join(pathway) + "/" + anndata_file)
    #Messages and others
    m1="You choose is_from_previous_note=True. Then, set an which_notebook=[INT], which INT is the number of the notebook that generated the anndata object you want to load."
    m2="Set the data_to_evaluate=[STRING], which STRING is anndata.obs[STRING] to be used for analysis, e.g. condition."
    m3="Paste the pathway and filename where your anndata object deposited."
    m4="Correct the pathway or filename or type q to quit."
    opt1=["q", "quit"]

    if isinstance(data_to_evaluate, str) == False: #Close if the anndata.obs is not correct
            sys.exit(m2)
    if is_from_previous_note == True: #Load anndata object from previous notebook
        try:
            ch.check_notebook(which_notebook)
        except TypeError:
            sys.exit(m1)
        file_path=loading_adata(which_notebook)
        data=sc.read_h5ad(filename=file_path) #Loading the anndata
        cr.build_infor(data, "data_to_evaluate", data_to_evaluate) #Annotating the anndata data to evaluate
        return(data)
    elif is_from_previous_note == False: #Load anndata object from other source
        answer=input(m3)
        while path.isfile(answer) == False: #False if pathway is wrong
            if answer.lower() in opt1:
                sys.exit("You quit and lost all modifications :(")
            print(m4)
            answer=input(m4)
        data=sc.read_h5ad(filename=answer) #Loading the anndata
        cr.build_infor(data, "data_to_evaluate", data_to_evaluate) #Annotating the anndata data to evaluate
        cr.build_infor(data, "Anndata_path", answer.rsplit('/', 1)[0]) #Annotating the anndata path
        return(data)

def saving_anndata(ANNDATA, current_notebook=None):
    '''
    Save your anndata object
    Parameters
    ===========
    ANNDATA : anndata object
        adata object
    current_notebook : int
        The number of the current notebook.
    '''
    #Author : Guilherme Valente
    #Messages and others
    m1="Set an current_notebook=[INT], which INT is the number of current notebook."
    m2="Your new anndata object is saved here: "

    try:
        ch.check_notebook(current_notebook)
    except TypeError:
        sys.exit(m1) #Close if the notebook number is not an integer
    adata_output=ANNDATA.uns["infoprocess"]["Anndata_path"] + "anndata_" + str(current_notebook) + "_" + ANNDATA.uns["infoprocess"]["Test_number"] + ".h5ad"
    ANNDATA.write(filename=adata_output)
    print(m2 + adata_output)

def pseudobulk_table(adata, groupby, how="mean"):
    """ Get a pseudobulk table of values per cluster. 
    
    Parameters
    -----------
    adata : anndata.AnnData
        An annotated data matrix containing counts in .X.
    groupby : str
        Name of a column in adata.obs to cluster the pseudobulks by.
    how : str, optional
        How to calculate the value per cluster. Can be one of "mean" or "sum". Default: "mean"
    """
    
    adata = adata.copy()
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    
    #Fetch the mean/sum counts across each category in cluster_by
    res = pd.DataFrame(columns=adata.var_names, index=adata.obs[groupby].cat.categories)                                                      
    for clust in adata.obs[groupby].cat.categories: 
        
        if how == "mean":
            res.loc[clust] = adata[adata.obs[groupby].isin([clust]),:].X.mean(0)
        elif how == "sum":
            res.loc[clust] = adata[adata.obs[groupby].isin([clust]),:].X.sum(0)
    
    res = res.T #transform to genes x clusters
    return(res)

def split_list(l, n):
    """ Split list into n chunks.
    
    Parameters
    -----------
    l : list
        List to be chunked
    n : int
        Number of chunks.

    """
    chunks = []
    for i in range(0, n):
        chunks.append(l[i::n])

    return chunks


def open_bam(file, mode, verbosity=3, **kwargs):
    """
    Open bam file with pysam.AlignmentFile. On a specific verbosity level.

    Parameters
    ----------
    file : str
        Path to bam file.
    mode : str
        Mode to open the file in. See pysam.AlignmentFile
    verbosity : int
        Verbosity level 0 for no messages.
    **kwargs :
        Forwarded to pysam.AlignmentFile
    """
    # check then load modules
    check_module("pysam")
    import pysam

    # save verbosity, then set temporary one
    former_verbosity = pysam.get_verbosity()
    pysam.set_verbosity(verbosity)

    # open file
    handle = pysam.AlignmentFile(file, mode, **kwargs)

    # return to former verbosity
    pysam.set_verbosity(former_verbosity)

    return handle


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
                       max_queue_size=1000):
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
    barcode_col : str, optional
        Name of a column in adata.obs to use as barcodes. If None, use the index of .obs.
    read_tag : str, optional
        Tag to use to identify the reads to split. Must match the barcodes of the barcode_col. Default: "CB".
    output_prefix : str, optional
        Prefix to use for the output files. Default: "split_".
    reader_threads : int, default 1
        Number of threads to use for reading.
    writer_threads : int, default 1,
        Number of threads to use for writing.
    parallel : boolean, default False
        Whether to enable parallel processsing.
    buffer_size : int, optional
        The size of the buffer between readers and writers. Default is 10.000 reads.
    max_queue_size : int, optional
        The maximum size of the queue between readers and writers. Default is 1000.
    """

    # check then load modules
    check_module("pysam")
    import pysam

    check_module("tqdm")
    from tqdm import tqdm

    # check whether groupby and barcode_col are in adata.obs
    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs!")

    if barcode_col is not None and barcode_col not in adata.obs.columns:
        raise ValueError(f"Column '{barcode_col}' not found in adata.obs!")

    if isinstance(bams, str):
        bams = [bams]

    # create output folder if needed
    create_dir(output_prefix)

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

    if parallel:

        # create path for output files
        out_paths = {}
        for cluster in clusters:
            # replace special characters in filename with "_" https://stackoverflow.com/a/27647173
            save_cluster_name = re.sub(r'[\\/*?:"<>|]', '_', cluster)
            out_paths[cluster] = f"{output_prefix}{save_cluster_name}.bam"

        # ---- Setup pools and queues ---- #
        # setup pools
        reader_pool = multiprocessing.Pool(reader_threads)
        writer_pool = multiprocessing.Pool(writer_threads)

        # setup queues to forward data between processes
        manager = multiprocessing.Manager()

        # Make chunks of writer_threads
        cluster_chunks = split_list(clusters, writer_threads)
        cluster_queues = {}
        for chunk in cluster_chunks:
            # print("Making queues for clusters:", chunk)
            m = manager.Queue(maxsize=max_queue_size)  # setup queue to get reads from readers
            for cluster in chunk:
                cluster_queues[cluster] = m  # manager is shared between different clusters if writer_threads < number of clusters

        print(' ', end='', flush=True)  # hack for making progress bars work in notebooks

        # ---- Start process ---- #
        # start reading bams and add reads into respective cluster queue
        reader_results = []
        for i, bam in enumerate(bams):
            pbar_text = "Reading " + os.path.basename(bam)
            reader_results.append(reader_pool.apply_async(_buffered_reader, (bam, cluster_queues, barcode2cluster, read_tag, i, pbar_text, buffer_size), callback=lambda x: print(x)))

        # write reads to files; one process per file
        writer_results = []
        for j, chunk in enumerate(cluster_chunks):
            queue = cluster_queues[chunk[0]]  # same queue for the whole chunk
            position = len(bams) + j
            path_dict = {cluster: out_paths[cluster] for cluster in chunk}  # subset of out_paths specific for this writer
            writer_results.append(writer_pool.apply_async(_writer, (queue, path_dict, str(template.header), pysam_threads, position), callback=lambda x: print(x)))

        # ---- End process + cleanup ----#
        # wait for readers to finish
        reader_pool.close()
        _ = [result.get() for result in reader_results]  # get any errors from threads
        reader_pool.join()

        # put None into queues as a sentinel to stop writers
        for q in cluster_queues.values():
            q.put((None, None))

        # wait for writers to finish
        writer_pool.close()
        _ = [result.get() for result in writer_results]  # get any errors from threads
        writer_pool.join()

    else:
        # open output bam files
        handles = {}
        for cluster in clusters:
            # replace special characters in filename with "_" https://stackoverflow.com/a/27647173
            save_cluster_name = re.sub(r'[\\/*?:"<>|]', '_', cluster)
            f_out = f"{output_prefix}{save_cluster_name}.bam"
            handles[cluster] = pysam.AlignmentFile(f_out, "wb", template=template, threads=pysam_threads)

        # Loop over bamfile(s)
        for i, bam in enumerate(bams):
            print(f"Looping over reads from {bam} ({i+1}/{len(bams)})")

            bam_obj = pysam.AlignmentFile(bam, "rb")

            # Update progress based on total number of reads
            # fall back to "samtools view -c file" if bam_obj.mapped is not available
            try:
                total = bam_obj.mapped
            except ValueError:
                total = int(pysam.view("-c", bam))
            pbar = tqdm(total=total)
            step = int(total / 10000)  # 10000 total updates

            i = 0
            written = 0
            for read in bam_obj:
                i += 1

                bc = read.get_tag(read_tag)

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


def _buffered_reader(path, out_queues, bc2cluster, tag, pbar_position, pbar_text, buffer_size=10000):
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
    pbar_position : int
        The position of the pbar for this job.
    pbar_text : str
        The text of the pbar for this job.
    buffer_size : int, optional
        Size of buffer (number of reads) for each queue to collect before writing. Default: 10000.
    """

    import pysam

    if _is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    print(' ', end='', flush=True)  # hack for making progress bars work in notebooks; https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308

    # open bam
    bam = open_bam(path, "rb", verbosity=0)

    # Get number of reads in bam
    try:
        total = bam.mapped
    except ValueError:
        print("Getting number of reads using pysam")
        total = int(pysam.view("-c", path))
    
    # setup progressbar
    pbar = tqdm(total=total, position=pbar_position, desc=pbar_text, unit="reads")
    step = int(total / 10000)  # 10000 total updates

    # Setup read buffer per cluster
    read_buffer = {cluster: [] for cluster in set(bc2cluster.values())}

    # put each read into correct queue
    written = 0
    i = 0  # count to update pbar
    for read in bam:
        bc = read.get_tag(tag)

        # put read into matching cluster queue
        if bc in bc2cluster:
            cluster = bc2cluster[bc]
            read_buffer[cluster].append(read.to_string())

            # Send reads to buffer when buffer size is reached
            if len(read_buffer[cluster]) == buffer_size:
                out_queues[cluster].put((cluster, read_buffer[cluster]))  # send the tuple of (clustername, read_buffer) to queue
                written += len(read_buffer[cluster])
                read_buffer[cluster] = []

        # update progress bar
        i += 1
        if i == step:
            pbar.update(step)
            i = 0

    # all reads have been read
    pbar.n = total
    pbar.refresh()
    bam.close()

    # Send remaining reads to buffer
    for cluster in read_buffer:
        if len(read_buffer[cluster]) > 0:
            out_queues[cluster].put((cluster, read_buffer[cluster]))
            written += len(read_buffer[cluster])

    return f"Done reading '{path}' - sent {written} reads to writer queues"


def _writer(read_queue, out_paths, bam_header, pysam_threads=4, position=0):
    """
    Write reads to given file.

    Parameters
    ----------
    read_queue : multiprocessing.Queue
        Queue of reads to be written into file.
    out_paths : dict
        Path to output files for this writer. In the format {cluster1: <path>, cluster2: <path>}
    pysam_threads : int, optional
        Number of threads for pysam to use for writing. This is different from the threads used for the individual writers. Default: 4.
    bam_header : str(pysam.AlignmentHeader)
        Used as template for output bam.
    """
    import pysam

    if _is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    print(' ', end='', flush=True)  # hack for making progress bars work in notebooks
    pbar = tqdm(position=position, unit=" items", desc="Size of writer queue for clusters: {0}".format(list(out_paths.keys())))

    # open bam files for writing cluster reads
    try:
        handles = {}
        for cluster in out_paths:
            handles[cluster] = pysam.AlignmentFile(out_paths[cluster], "wb", text=bam_header, threads=pysam_threads)
    except Exception as e:
        print(e.message)
        raise e

    # fetch reads from queue and write to bam
    n_written = {cluster: 0 for cluster in out_paths}
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
            n_written[cluster] += 1

        # Get size of writer queue
        pbar.n = read_queue.qsize()
        pbar.refresh()

    # Close files after use
    for handle in handles.values():
        handle.close()

    return "Done writing reads to .bam-files."
