"""
Module to assembling anndata objects
"""
import sctoolbox.checker as ch
import sctoolbox.creators as cr

import scanpy as sc
import pandas as pd
import anndata
from anndata import AnnData

import yaml
import sys

from scipy import sparse
from scipy.io import mmread

#######################################################################################################################
#################################ASSEMBLING ANNDATA FOR THE VELOCITY ANALYSIS##########################################
def assembler_velocity(SAMPLE, PATH, DTYPE): #Author: Guilherme Valente
    global adata
    path1=PATH + "/quant/" + SAMPLE + "/solo/Velocyto/" + DTYPE
    path_for_matrix=PATH + "/quant/" + SAMPLE + "/solo/Gene/" + DTYPE
    path_for_X_spl_unspl_ambig=path1
    path_for_obs=path1 + '/barcodes.tsv'
    path_for_var=path1 + '/genes.tsv'
    print("\t\tLoading matrix to compose the X object: " + SAMPLE)
    X = sc.read_mtx(path_for_matrix + '/matrix.mtx')
    X = X.X.transpose()
    print("\t\tLoading genes and cells identifiers to make the obs object: " + SAMPLE)
    obs = pd.read_csv(path_for_obs, header = None, index_col = 0)
    obs.index.name = None #Remove index column name to make it compliant with the anndata format
    obs = obs + '-' + str(''.join(conditions_name))
    print("\t\tLoading the gene features to make the var object: " + SAMPLE)
    var = pd.read_csv(path_for_var, sep='\t', names = ('gene_ids', 'feature_types'), index_col = 1)
    print("\t\tLoading spliced, unspliced and ambigous matrix to compose the X object: " + SAMPLE)
    spliced = sparse.csr_matrix(mmread(path_for_X_spl_unspl_ambig + '/spliced.mtx')).transpose()
    unspliced  = sparse.csr_matrix(mmread(path_for_X_spl_unspl_ambig + '/unspliced.mtx')).transpose()
    ambiguous  = sparse.csr_matrix(mmread(path_for_X_spl_unspl_ambig + '/ambiguous.mtx')).transpose()
    print("\t\tCreating partial anndata object: " + SAMPLE)
    adata = anndata.AnnData(X = X, obs = obs, var = var, layers = {'spliced': spliced, 'unspliced': unspliced, 'ambiguous': ambiguous})
    adata.var_names_make_unique()
    return adata.copy()

def velocity(tenX, TEST, SOLO_PATH): #tenX is the configuration of samples in the 10X.yml. TEST is test number
    dtype="filtered" #the dtype is the type of Solo data choose (raw or filtered), which default is filtered
    if ch.check_infoyml("Output_path"): #Check existence of results path in yaml.
        result_path=''.join(ch.check_infoyml("Output_path"))
        test=result_path.split("results/")[1]
        if TEST != test: #If the test description is not the same as yml, close the program.
            sys.exit("The " + TEST + " is not the same as described in info.yml.")
        else:
            global conditions_name
            adata_list=list()
            conditions_name=[]
            dict_rename_samples={}
            timer=0
            for a in tenX:
                print("Runing sample number " + str(timer + 1) + " out of " + str(len(tenX)) + " samples.")
                sample=a.split(":")[0]
                condition=a.split(":")[1]
                condition_description=a.split(":")[2]
                dict_rename_samples[str(timer)]=condition_description
                if condition not in conditions_name:
                    conditions_name.append(condition)
                adata_list.append(assembler_velocity(sample, SOLO_PATH, dtype)) #EXECUTING THE ASSEMBLER
                timer=timer+1
            #Creating the final anndata and saving
            print("Creating the final anndata object.")
            print("\t\tConcatenating objects.")
            adata = adata_list[0].concatenate(adata_list[1:])
            print("\t\tRenaming batches.")
            adata.obs["batch"].replace(dict_rename_samples, inplace=True)
            adata.obs.rename(columns = {"batch": ''.join(conditions_name)}, inplace = True)
            print("\t\tInserting informations")
            cr.build_infor(adata, "Test_number", TEST) #Anndata, key and value for anndata.uns["infoprocess"]
            cr.build_infor(adata, "Input_for_assembling", SOLO_PATH)
            cr.build_infor(adata, "Strategy", "Assembling for velocity")
            cr.build_infor(adata, "Anndata_path", result_path)
            print("\t\tSaving and loading.")
            name="/anndata_1_" + TEST +".h5ad"
            adata_output= result_path + name
            adata.write(filename=adata_output) #SAVIND THE ADATA FILE WITH SPLICED UNSPLICED; AMBIGOUS COUNTINGS
            #Loading adata file and printing num cells and num genes
            print("Loading the anndata for velocity and storing as an adata variable.")
            adata = sc.read_h5ad(filename=adata_output)

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
