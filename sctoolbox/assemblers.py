"""
Module to assembling anndata objects
"""
import scanpy as sc
import pandas as pd

import os
import glob

from scipy import sparse
from scipy.io import mmread


#######################################################################################################################
#################################ASSEMBLING ANNDATA FOR THE VELOCITY ANALYSIS##########################################
#######################################################################################################################

def from_single_starsolo(path, dtype="filtered"):
    '''
    This will assemble an anndata object from the starsolo folder.

    Parameters
    ----------
    path : str
        Path to the "solo" folder from starsolo.
    dtype : str, optional
        The type of solo data to choose. Must be one of ["raw", "filtered"]. Default: "filtered".
    '''
    #Author : Guilherme Valente

    #Establish which directory to look for data in 
    genedir = os.path.join(path, "Gene", dtype)
    velodir = os.path.join(path, "Velocyto", dtype)
    for path in [genedir, velodir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path to the data does not exist: {path}")

    #File paths for different elements
    matrix_f = os.path.join(genedir, 'matrix.mtx')
    barcodes_f = os.path.join(genedir, "barcodes.tsv")
    genes_f = os.path.join(genedir, "genes.tsv")
    spliced_f = os.path.join(velodir, 'spliced.mtx')
    unspliced_f = os.path.join(velodir,  'unspliced.mtx')
    ambiguous_f = os.path.join(velodir, 'ambiguous.mtx')

    #TODO: #Check whether files are present

    #Setup main adata object from matrix/barcodes/genes
    print("Setting up adata from solo files")
    adata = from_single_mtx(matrix_f, barcodes_f, genes_f, is_10X=False)
    adata.var.columns = ["gene", "type"] #specific to the starsolo format
    for col in adata.obs.columns:
        adata.var[col] = adata.var[col].astype("category")

    #Add in velocity information
    print("Adding velocity information from spliced/unspliced/ambiguous")
    spliced = sparse.csr_matrix(mmread(spliced_f).transpose())
    unspliced = sparse.csr_matrix(mmread(unspliced_f).transpose())
    ambiguous = sparse.csr_matrix(mmread(ambiguous_f).transpose())

    adata.layers["spliced"] = spliced
    adata.layers["unspliced"] = unspliced
    adata.layers["ambiguous"] = ambiguous

    return adata

def from_quant(path, configuration=[], use_samples=None, dtype="filtered"):
    '''
    Assemble an adata object from data in the 'quant' folder of the snakemake pipeline.

    Parameters
    -----------
    path : str
        The directory where the quant folder from snakemake preprocessing is located.
    configuration : list
        Configurations to setup the samples for anndata assembling. It must containg the sample, the word used in snakemake to assign the condition, and the condition, e.g., sample1:condition:room_air
    use_samples : list or None
        List of samples to use. If None, all samples will be used.
    dtype : str, optional
        The type of Solo data to choose. The options are 'raw' or 'filtered'. Default: filtered.
    '''
    #Author : Guilherme Valente

    #TODO: test that quant folder is existing
    
    #Collect configuration into a dictionary
    config_dict = {}
    if configuration is not None:
        for string in configuration:
            sample, condition, condition_name = string.split(":")

            if sample not in config_dict:
                config_dict[sample] = {}
            config_dict[sample][condition] = condition_name

    #Establishing which samples to use for assembly
    sample_dirs = glob.glob(os.path.join(path, "*"))
    sample_names = [os.path.basename(x) for x in sample_dirs]
    print(f"Found samples: {sample_names}")

    #Subset to use_samples if they are provided
    if use_samples is not None:
        idx = [i for i, sample in enumerate(sample_names) if sample in use_samples]
        sample_names = [sample_names[i] for i in idx]
        sample_dirs = [sample_dirs[i] for i in idx]

        print(f"Using samples: {sample_names}")
        if len(sample_names) == 0:
            raise ValueError("None of the given 'use_samples' match the samples found in the directory.")

    #Assembling from different samples:
    adata_list = []
    for sample_name, sample_dir in zip(sample_names, sample_dirs):

        print(f"Assembling sample '{sample_name}'")
        solo_dir = os.path.join(sample_dir, "solo")
        adata = from_single_starsolo(solo_dir, dtype=dtype)

        #Make barcode index unique
        adata.obs.index = adata.obs.index + "-" + sample_name
        adata.obs["sample"] = sample_name
        
        #Add additional information from configuration
        if sample_name in config_dict:
            for key in config_dict[sample_name]:
                adata.obs[key] = config_dict[sample_name][key]
                adata.obs[key] = adata.obs[key].astype("category")

        adata_list.append(adata)

    #Concatenating the adata objects
    print("Concatenating anndata objects")
    adata = adata_list[0].concatenate(adata_list[1:], join="outer")

    return adata


#######################################################################################################################
####################################CONVERTING FROM MTX+TSV/CSV TO ANNDATA OBJECT######################################
#######################################################################################################################

def from_single_mtx(mtx, barcodes, genes, is_10X = True, transpose = True, barcode_index = 0, genes_index = 0, delimiter = "\t", **kwargs):
    ''' Building adata object from single mtx and two tsv/csv files
    
    Parameter:
    ----------
    mtx : string
        Path to the mtx file (.mtx)
    barcodes : string
        Path to cell label file (.obs)
    genes : string
        Path to gene label file (.var)
    is_10X : boolean
        Set True if mtx file contains 10X data
    transpose : boolean
        Set True to transpose mtx matrix
    barcode_index : int
        Column which contains the cell barcodes (Default: 0 -> Takes first column)
    genes_index : int
        Column h contains the gene IDs (Default: 0 -> Takes first column)
    delimiter : string
        delimiter of genes and barcodes table
    **kwargs : additional arguments
        Contains additional arguments for scanpy.read_10x_mtx method
        
    returns
    -------
    anndata object containing the mtx matrix, gene and cell labels
    '''
    ### Read mtx file ###
    if is_10X:
        adata = sc.read_10x_mtx(path=mtx, **kwargs)
    else:
        adata = sc.read_mtx(filename=mtx, dtype='float32')
    
    ### Transpose matrix if necessary ###
    if transpose:
        adata = adata.transpose()
    
    ### Read in gene and cell annotation ###
    barcode_csv = pd.read_csv(barcodes, header=None, index_col=barcode_index, delimiter=delimiter)
    barcode_csv.index.names = ['index']
    barcode_csv.columns = [str(c) for c in barcode_csv.columns] #convert to string
    genes_csv = pd.read_csv(genes, header=None, index_col=genes_index, delimiter=delimiter)
    genes_csv.index.names = ['index']
    genes_csv.columns = [str(c) for c in genes_csv.columns] #convert to string
    
    ### Test if they are unique ###
    if not barcode_csv.index.is_unique:
        raise ValueError("Barcode index column does not contain unique values")
    if not genes_csv.index.is_unique:
        raise ValueError("Genes index column does not contain unique values")
    
    ### Add tables to anndata object ###
    adata.obs = barcode_csv 
    adata.var = genes_csv
    
    return adata


def from_mtx(mtx, barcodes, genes, **kwargs):
    '''
    Building adata object from list of mtx, barcodes and genes files
    
    Parameter:
    ----------
    mtx : list
        List of paths to mtx files
    barcodes : list
        List of paths to cell barcode files
    genes : list
        List of paths to gene label files
    is_10X : boolean
        Set True if mtx file contains 10X data
    transpose : boolean
        Set True to transpose mtx matrix
    barcode_index : int
        Column which contains the cell barcodes (Default: 0 -> Takes first column)
    genes_index : int
        Column h contains the gene IDs (Default: 0 -> Takes first column)
    delimiter : string
        delimiter of genes and barcodes table
    **kwargs : additional arguments
        Contains additional arguments for scanpy.read_10x_mtx method

    returns:
    --------
    merged anndata object containing the mtx matrix, gene and cell labels
    '''
    
    adata_objects = [from_single_mtx(m, barcodes[i], genes[i], **kwargs) for i, m in enumerate(mtx)]
    
    if len(adata_objects) >1:
        adata = adata_objects[0].concatenate(*adata_objects[1:], join = "outer")
    else:
        adata = adata_objects[0]
    
    return adata
