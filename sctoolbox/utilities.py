import pandas as pd
import sys
import os
import scanpy as sc
import importlib

from sctoolbox.checker import *
from sctoolbox.creators import *

import matplotlib.pyplot as plt

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


def split_bam_clusters(adata, bams, groupby, barcode_col=None, read_tag="CB", output_prefix="split_"):
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
    """
    
    # check whether groupby and barcode_col are in adata.obs
    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs!")

    if barcode_col not in adata.obs.columns:
        raise ValueError(f"Column '{barcode_col}' not foun in adata.obs!")
    
    if isinstance(bams, str):
        bams = [bams]
        
    #Establish clusters from obs
    clusters = set(adata.obs[groupby])
    print(f"Found {len(clusters)} groups in .obs.{groupby}: {list(clusters)}")

    if barcode_col is None:
        barcode2cluster = dict(zip(adata.obs.index.tolist(), adata.obs[groupby]))
    else:
        barcode2cluster = dict(zip(adata.obs[barcode_col], adata.obs[groupby]))

    #Open output files for writing clusters
    template = pysam.AlignmentFile(bams[0], "rb")
    
    handles = {}
    for cluster in clusters:
        f_out = f"{output_prefix}{cluster}.bam"
        handles[cluster] = pysam.AlignmentFile(f_out, "wb", template=template)
        
    #Loop over bamfile(s)
    for i, bam in enumerate(bams):
        print(f"Looping over reads from {bam} ({i+1}/{len(bams)})")
        
        bam_obj = pysam.AlignmentFile(bam, "rb")
        
        #Update progress based on total number of reads
        total = bam_obj.mapped
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
        
        print(f"Wrote {written} reads to cluster files")
        
    #Close all files
    for handle in handles.values():
        handle.close()
