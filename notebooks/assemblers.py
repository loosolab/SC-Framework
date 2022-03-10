"""
Module to assembling anndata objects
"""
#Importing modules
#import os
#from os import path
import yaml
import checker
import sys
import scanpy as sc
import pandas as pd
from scipy import sparse
from scipy.io import mmread
import anndata
from anndata import AnnData

#######################################################################################################################
#################################ASSEMBLING ANNDATA FOR THE VELOCITY ANALYSIS##########################################
def assembler_velocity(SAMPLE, PATH, DTYPE):
    global adata
    path1=PATH + "/" + SAMPLE + "/solo/Velocyto/" + DTYPE
    path_for_matrix=PATH + "/" + SAMPLE + "/solo/Gene/" + DTYPE
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

def velocity(tenX, TEST, dtype): #tenX is the configuration of samples in the 10X.yml. TEST is test number
    if checker.check_infoyml("Input_solo_path"): #Check existence of solo path.
        solo_path=''.join(checker.check_infoyml("Input_solo_path"))
    if checker.check_infoyml("Output_path"): #Check existence of results path in yaml.
        result_path=''.join(checker.check_infoyml("Output_path"))
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
                adata_list.append(assembler_velocity(sample, solo_path, dtype)) #EXECUTING THE ASSEMBLER
                timer=timer+1
            #Creating the final anndata and saving
            print("Creating the final anndata object.")
            print("\t\tConcatenating objects.")
            adata = adata_list[0].concatenate(adata_list[1:])
            print("\t\tRenaming batches.")
            adata.obs["batch"].replace(dict_rename_samples, inplace=True)
            adata.obs.rename(columns = {"batch": ''.join(conditions_name)}, inplace = True)
            print("\t\tSaving and loading.")
            name="/anndata_1_" + TEST +".h5ad"
            adata_output= result_path + name
            adata.write(filename=adata_output) #SAVIND THE ADATA FILE WITH SPLICED UNSPLICED; AMBIGOUS COUNTINGS
            #Loading adata file and printing num cells and num genes
            print("Loading the anndata for velocity and storing as an adata variable.")
            adata = sc.read_h5ad(filename=adata_output)
            
            return(adata)
#######################################################################################################################
