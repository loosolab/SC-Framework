import os
from os.path import join, dirname, exists
import sys
from pathlib import Path


def convertRobj(file, out, r_home=None):
    """
    converts .robj files containing SingleCellExperiment to .rds.
    The .rds file is then converted to anndata.

    In order to work an R installation with Seurat & SingleCellExperiment is required.

    Args:
        file (str):
            Path to .robj file.
        r_home (str):
             Path to the R home directory. If None will construct path based on location of python executable.
            E.g for ".conda/scanpy/bin/python" will look at ".conda/scanpy/lib/R"
        out (str):
            Path to save the anndata .h5ad file.

    Returns:
        saves a .rds file.
    """

    ##### Set R installation path #####
    if not r_home:
        # https://stackoverflow.com/a/54845971
        r_home = join(dirname(dirname(Path(sys.executable).as_posix())), "lib", "R")

    if not exists(r_home):
        raise Exception(f'Path to R installation does not exist! Make sure R is installed. {r_home}')

    os.environ['R_HOME'] = r_home

    # Initialize R <-> python interface
    import anndata2ri
    from rpy2.robjects import r
    anndata2ri.activate()

    ##### convert to rds #####
    r(f"""
        file <- load("{file}")
        object <- get(file[1])
        saveRDS(object, file='out/tmp.rds')
       """)

    ##### convert to adata #####
    adata = r(f"""
                    library(Seurat)

                    object <- readRDS("out/tmp.rds")

                    # check type and convert if needed
                    if (class(object) == "Seurat") {{
                        object <- as.SingleCellExperiment(object)
                    }} else if (class(object) == "SingleCellExperiment") {{
                        object <- object
                    }} else {{
                        stop("Unknown object! Expected class 'Seurat' or 'SingleCellExperiment' got ", class(object))
                    }}
                   """)

    ##### Saving adata.h5ad #####
    h5ad_file = out+'/anndata_1.h5ad'
    adata.write(filename=h5ad_file, compression='gzip')

    ##### Removing tmp.rds #####
    os.remove(out+'/tmp.rds')

    return adata


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
                
                object <- readRDS("{file}")

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
