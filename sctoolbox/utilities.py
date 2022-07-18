import pandas as pd
import sys
import os
import scanpy as sc
import importlib
import re
import multiprocessing

from sctoolbox.checker import *
from sctoolbox.creators import *

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

def save_figure(path):
    """ Save the current figure to a file.
    
    Parameters
    ----------
    path : str
        Path to the file to be saved.
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
def load_anndata(is_from_previous_note=True, notebook=None, data_to_evaluate=None):
    '''
    Load anndata object
    ==========
    Parameters
    ==========
    is_from_previous_note : Boolean
        Set to False if you wanna load an anndata object from other source rather than scRNAseq autom workflow.
    NOTEBOOK : Int.
        This is the number of current notebook under execution
        If is_from_previous_note=False, this parameter will be ignored
    data_to_evaluate : String
        This is the anndata.obs[STRING] to be used for analysis, e.g. "condition"
    '''
    #Author : Guilherme Valente
    def loading_adata(NUM):
        pathway=check_infoyml(TASK="give_path")
        files=os.listdir(''.join(pathway))
        for a in files:
            if "anndata_" + str(NUM-1) in a:
                anndata_file=a
        return(''.join(pathway) + "/" + anndata_file)
    #Messages and others
    m1="You choose is_from_previous_note=True. Then, set an notebook=[INT], which INT is the number of current notebook."
    m2="Set the data_to_evaluate=[STRING], which STRING is anndata.obs[STRING] to be used for analysis, e.g. condition."
    m3="Paste the pathway and filename where your anndata object deposited."
    m4="Correct the pathway or filename or type q to quit."
    opt1=["q", "quit"]

    if isinstance(data_to_evaluate, str) == False: #Close if the anndata.obs is not correct
            sys.exit(m2)
    if is_from_previous_note == True: #Load anndata object from previous notebook
        if isinstance(notebook, int) == False: #Close if the notebook number is not correct
            sys.exit(m1)
        else:
            file_path=loading_adata(notebook)
            data=sc.read_h5ad(filename=file_path) #Loading the anndata
            build_infor(data, "data_to_evaluate", data_to_evaluate) #Annotating the anndata data to evaluate
            return(data)
    elif is_from_previous_note == False: #Load anndata object from other source
        answer=input(m3)
        while path.isfile(answer) == False: #False if pathway is wrong
            if answer.lower() in opt1:
                sys.exit("You quit and lost all modifications :(")
            print(m4)
            answer=input(m4)
        data=sc.read_h5ad(filename=answer) #Loading the anndata
        build_infor(data, "data_to_evaluate", data_to_evaluate) #Annotating the anndata data to evaluate
        build_infor(data, "Anndata_path", answer.rsplit('/', 1)[0]) #Annotating the anndata path
        return(data)

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


def split_bam_clusters(adata, bams, groupby, barcode_col=None, read_tag="CB", output_prefix="split_", reader_threads=1, writer_threads=1, parallel=False):
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

    # TODO create folder if specified in prefix

    #Establish clusters from obs
    clusters = set(adata.obs[groupby])
    print(f"Found {len(clusters)} groups in .obs.{groupby}: {list(clusters)}")

    # setup barcode <-> cluster dict
    if barcode_col is None:
        barcode2cluster = dict(zip(adata.obs.index.tolist(), adata.obs[groupby]))
    else:
        barcode2cluster = dict(zip(adata.obs[barcode_col], adata.obs[groupby]))

    # create template used for bam header
    template = pysam.AlignmentFile(bams[0], "rb")

    if parallel:
        # create path for output files
        out_paths = {}
        for cluster in clusters:
            # replace special characters in filename with "_" https://stackoverflow.com/a/27647173
            save_cluster_name = re.sub(r'[\\/*?:"<>|]', '_', cluster)
            out_paths[cluster] = f"{output_prefix}{save_cluster_name}.bam"

        ### Setup
        # setup pools
        reader_pool = multiprocessing.Pool(reader_threads)
        writer_pool = multiprocessing.Pool(writer_threads)

        # setup queues to forward data between processes
        manager = multiprocessing.Manager()

        #read_chunk_queue = manager.Queue()
        cluster_queues = {cluster: manager.Queue() for cluster in clusters}

        ### Start process
        # start reading bams and add reads into respective cluster queue
        reader_results = []
        for i, bam in enumerate(bams):
            bam_name = os.path.basename(bam)
            reader_results.append(reader_pool.apply_async(_buffered_reader, (bam, cluster_queues, barcode2cluster, read_tag, i, bam_name), callback=lambda x: print(x)))

        # write reads to files; one process per file
        writer_results = []
        for cluster in clusters:
            writer_results.append(writer_pool.apply_async(_writer, (cluster_queues[cluster], out_paths[cluster], str(template.header)), callback=lambda x: print(x)))

        ### End process + cleanup
        # wait for readers to finish
        reader_pool.close()
        _ = [result.get() for result in reader_results]  # get any errors from threads
        reader_pool.join()

        # put None into queues as a sentinel to stop writers
        for q in cluster_queues.values():
            q.put(None)

        # wait for writers to finish
        writer_pool.close()
        writer_pool.join()

    else:
        # open output bam files
        handles = {}
        for cluster in clusters:
            # replace special characters in filename with "_" https://stackoverflow.com/a/27647173
            save_cluster_name = re.sub(r'[\\/*?:"<>|]', '_', cluster)
            f_out = f"{output_prefix}{save_cluster_name}.bam"
            handles[cluster] = pysam.AlignmentFile(f_out, "wb", template=template)

        #Loop over bamfile(s)
        for i, bam in enumerate(bams):
            print(f"Looping over reads from {bam} ({i+1}/{len(bams)})")
            
            bam_obj = pysam.AlignmentFile(bam, "rb")
            
            #Update progress based on total number of reads
            # fall back to "samtools view -c file" if bam_obj.mapped is not available
            try:
                total = bam_obj.mapped
            except ValueError:
                total = int(pysam.view("-c", bam))
            pbar = tqdm(total=total)
            step = int(total / 10000) #10000 total updates
            
            i = 0
            written = 0
            for read in bam_obj:
                i += 1
                
                bc = read.get_tag(read_tag)
                
                #Update step manually - there is an overhead to update per read with hundreds of million reads
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

        #Close all files
        for handle in handles.values():
            handle.close()

def _buffered_reader(path, out_queues, bc2cluster, tag, pbar_position, pbar_text):
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
    pbar_text 
        The text of the pbar for this job.
    """
   
    import pysam
    
    if _is_notebook() == True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    print(' ', end='', flush=True)  #hack for making progress bars work in notebooks; https://github.com/tqdm/tqdm/issues/485#issuecomment-473338308

    # open bam
    bam = pysam.AlignmentFile(path, "rb")

    #number of reads in bam 
    try:
        total = bam.mapped
    except ValueError:
        print("Getting number of reads using pysam")
        total = int(pysam.view("-c", bam))

    # setup progressbar
    pbar = tqdm(total=total, position=pbar_position, desc=pbar_text)
    step = int(total / 10000) #10000 total updates

    # Setup read buffer per cluster
    read_buffer = {cluster: [] for cluster in set(bc2cluster.values())}
    buffer_size = 10000

    # put each read into correct queue
    i = 0
    for read in bam:
        bc = read.get_tag(tag)

        # put read into matching cluster queue
        if bc in bc2cluster:
            cluster = bc2cluster[bc]
            read_buffer[cluster].append(read.to_string())

            # Send reads to buffer when buffer size is reached
            if len(read_buffer[cluster]) == buffer_size:
                out_queues[cluster].put(read_buffer[cluster])
                out_queues[cluster] = []
        
        #update progress bar
        i += 1
        if i == step:
            pbar.update(step)
            i = 0

    bam.close()

    return(f"Done reading {path} of {i} reads")

def _writer(read_queue, path, bam_header):
    """
    Write reads to given file.

    Parameters
    ----------
    read_queue : multiprocessing.Queue
        Queue of reads to be written into file.
    path : str
        Path to output file
    bam_header : str(pysam.AlignmentHeader)
        Used as template for output bam.
    """
    import pysam

    handle = pysam.AlignmentFile(path, "wb", text=bam_header)

    i = 0
    while True:
        read_lst = read_queue.get()

        # stop writing
        if read_lst is None:
            break
        
        for read in read_lst:
            # create read object
            read = pysam.AlignedSegment.fromstring(read, handle.header)

            handle.write(read)

            i += 1

    handle.close()

    return(f"Done. Wrote {i} reads to {path}.")
