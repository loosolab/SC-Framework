import pandas as pd
import sys
import os
import scanpy as sc
import importlib

from sctoolbox.checker import *
from sctoolbox.creators import *


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
	adata : anndata object
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
