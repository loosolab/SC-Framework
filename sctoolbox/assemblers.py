"""
Module to assembling anndata objects
"""
import sctoolbox
import sctoolbox.checker as ch
import sctoolbox.creators as cr

import scanpy as sc
import pandas as pd
import anndata
from anndata import AnnData

import sys

from scipy import sparse
from scipy.io import mmread

#######################################################################################################################
#################################ASSEMBLING ANNDATA FOR THE VELOCITY ANALYSIS##########################################
def assembler(SAMPLE, PATH, DTYPE, COND_NAME):
    '''
    This will perform the anndata object 10X assembling for velocyt analysis
    Parameters
    ==========
    SAMPLE : String
       The sample name defined by snakemake, e.g., sample1
    PATH : String
       The pathway where the /quant is deposited
    DTYPE : String
       The type of Solo data choose, which default is filtered. The options are raw or filtered.
    COND_NAME : String.
      The word used in the snakemake to define the name of condition of the sample, e.g, condition
    '''
    #Author : Guilherme Valente
    #Messages and others
    m1="\t\tLoading matrix to compose the .X object"
    m2="\t\tLoading matrix to compose the .obs"
    m3="\t\tLoading matrix to compose the .var"
    path1=PATH + SAMPLE + "/solo/Velocyto/" + DTYPE
    path_for_matrix=PATH + SAMPLE + "/solo/Gene/" + DTYPE
    path_for_X_spl_unspl_ambig, path_for_obs, path_for_var=path1, path1 + '/barcodes.tsv', path1 + '/genes.tsv'

    print(m1)
    X = sc.read_mtx(path_for_matrix + '/matrix.mtx')
    X = X.X.transpose()
    print(m2)
    obs = pd.read_csv(path_for_obs, header = None, index_col = 0)
    obs.index.name = None #Remove index column name to make it compliant with the anndata format
    obs = obs + '-' + str(''.join(COND_NAME))
    print(m3)
    var = pd.read_csv(path_for_var, sep='\t', names = ('gene_ids', 'feature_types'), index_col = 1)
    spliced, unspliced, ambiguous = sparse.csr_matrix(mmread(path_for_X_spl_unspl_ambig + '/spliced.mtx')).transpose(), sparse.csr_matrix(mmread(path_for_X_spl_unspl_ambig + '/unspliced.mtx')).transpose(), sparse.csr_matrix(mmread(path_for_X_spl_unspl_ambig + '/ambiguous.mtx')).transpose()
    adata = anndata.AnnData(X = X, obs = obs, var = var, layers = {'spliced': spliced, 'unspliced': unspliced, 'ambiguous': ambiguous})
    adata.var_names_make_unique()
    return adata.copy()

def assembling_velocity(path_QUANT, tenX, assembling_10_velocity, TEST, dtype="filtered"):
    '''
    This will check if all data to assembling the 10X for velocity is proper and also perform the assembling
    Parameters
    ==========
    path_QUANT : String.
        The directory where the quant folder from snakemake preprocessing is located.
    tenX : List.
        Configurations to setup the samples for anndata assembling. It must containg the sample, the word used in snakemake to assign the condition, and the condition, e.g., sample1:condition:room_air
    assembling_10_velocity : Boolean
        If True, the anndata 10X assembling for velocity will be executed.
    TEST : String
        The name to label de analysis of this scRNAseq workflow, e.g., Test1
    dtype : String.
        The type of Solo data choose, which default is filtered. The options are raw or filtered.
    '''
    #Author : Guilherme Valente
    #Message and others
    adata_list, conditions_name, dict_rename_samples=list(), [], {} #Stores the anndata objects assembled, conditions of each sample and the dict will be used to rename the samples
    m1="The " + TEST + " is not the same as described in info.txt."
    m2="Num samples: " + str(len(tenX))
    m3="Assembling sample "
    m4="Concatenating anndata objects, renaming batches and building anndata.uns[infoprocess]."
    m5="\t\tSaving and loading."
    ######
    path_QUANT2=ch.check_input_path_velocity(path_QUANT, tenX, assembling_10_velocity, dtype="filtered") #Checking if all files for assembling are proper
    result_path=ch.check_infoyml(TASK="give_path") #Loading the output path
    test2=result_path.split("results/")[1].replace("/", '').strip()
    if TEST != test2: #Check if the test description is different that the one in info.txt.
        sys.exit(m1)
    print(m2)
    timer=0
    #Assembling
    for a in tenX:
        print(a)
        print(m3 + str(timer+1))
        sample, condition, condition_description=a.split(":")[0], a.split(":")[1], a.split(":")[2]
        dict_rename_samples[str(timer)]=condition_description
        if condition not in conditions_name:
            conditions_name.append(condition)
        adata_list.append(assembler(sample, path_QUANT2 + "/", dtype, conditions_name)) #EXECUTING THE ASSEMBLER
        timer=timer+1
    #Creating the final anndata and saving
    print(m4)
    adata = adata_list[0].concatenate(adata_list[1:])
    adata.obs["batch"].replace(dict_rename_samples, inplace=True)
    adata.obs.rename(columns = {"batch": ''.join(conditions_name)}, inplace = True)
    #Building anndata.info["infoprocess"]
    cr.build_infor(adata, "Test_number", TEST) #Anndata, key and value for anndata.uns["infoprocess"]
    cr.build_infor(adata, "Input_for_assembling", path_QUANT)
    cr.build_infor(adata, "Strategy", "Assembling for velocity")
    cr.build_infor(adata, "Anndata_path", result_path)
    #Saving the data
    print(m5)
    adata_output=result_path + "/anndata_1_" + TEST +".h5ad"
    adata.write(filename=adata_output)
    return(adata)

#######################################################################################################################
####################################CONVERTING FROM MTX+TSV/CSV TO ANNDATA OBJECT######################################
def from_single_mtx(mtx, barcodes, genes, is_10X = True, transpose = True, barcode_index = 0, genes_index = 0, delimiter = "\t", **kwargs):
    ''' Building adata object from single mtx and two tsv/csv files
    
    Parameter:
    ----------
    mtx : string
        Path to mtx files
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
    genes_csv = pd.read_csv(genes, header=None, index_col=genes_index, delimiter=delimiter)
    genes_csv.index.names = ['index']
    
    ### Test if they are unique ###
    if not barcode_csv.index.is_unique:
        raise ValueError("Barcode index column does not contain unique values")
    if not genes_csv.index.is_unique:
        raise ValueError("Genes index column does not contain unique values")
    
    ### Add tables to anndata object ###
    adata.obs = barcode_csv 
    adata.var = genes_csv
    
    return(adata)


def from_mtx(mtx, barcodes, genes, **kwargs):
    ''' Building adata object from list of mtx, barcodes and genes files
    
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
    
    adata_objects = [from_single_mtx(m, barcodes[i], genes[i], **kwargs) for i,m in enumerate(mtx)]
    
    if len(adata_objects) >1:
        adata = adata_objects[0].concatenate(*adata_objects[1:], join = "inner")
    else:
        adata = adata_objects[0]
    
    return adata
