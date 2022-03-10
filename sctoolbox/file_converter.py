import os
from os.path import join, dirname, exists
import sys
from pathlib import Path

def convertToAdata(file, r_home=None):
    '''
    Converts .rds files containing Seurat or SingleCellExperiment to scanpy anndata.
    
    In order to work an R installation with Seurat & SingleCellExperiment is required.
    
    Parameters:
    ----------
        path (str):
            Path to the .rds file.
        r_home (str):
            Path to the R home directory. If None will construct path based on location of python executable.
            E.g for ".conda/scanpy/bin/python" will look at ".conda/scanpy/lib/R"
    
    Returns:
    ----------
         anndata.AnnData:
            Converted anndata object.
    '''
    
    ##### Set R installation path #####
    if not r_home:       
        # https://stackoverflow.com/a/54845971
        r_home = join(dirname(dirname(Path(sys.executable).as_posix())), "lib","R")

    if not exists(r_home):
        raise Exception(f'Path to R installation does not exist! Make sure R is installed. {r_home}')

    
    os.environ['R_HOME'] = r_home
    
    # Initialize R <-> python interface
    import anndata2ri
    from rpy2.robjects import r
    anndata2ri.activate()
    
    ##### convert to adata #####
    adata = r(f"""
                library(Seurat)
                
                object <- readRDS("{path}")

                # check type and convert if needed
                if (class(object) == "Seurat") {{
                    object <- as.SingleCellExperiment(object)
                }} else if (class(object) == "SingleCellExperiment") {{
                    object <- object
                }} else {{
                    stop("Unknown object! Expected class 'Seurat' or 'SingleCellExperiment' got ", class(object))
                }}
               """)

    return adata
