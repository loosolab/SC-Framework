import os
from os.path import join, dirname, exists
import sys
from pathlib import Path

import sctoolbox.utilities as utils


def convertToAdata(file, out, r_home=None):
    '''
    Converts .rds files containing Seurat or SingleCellExperiment to scanpy anndata.

    In order to work an R installation with Seurat & SingleCellExperiment is required.

    Parameters:
    ----------
        path (str):
            Path to the .rds or .robj file.
        out (str):
            path to save anndata.h5ad file.
        r_home (str):
            Path to the R home directory. If None will construct path based on location of python executable.
            E.g for ".conda/scanpy/bin/python" will look at ".conda/scanpy/lib/R"

    Returns:
    ----------
         anndata.AnnData:
            Converted anndata object.
    '''

    # Set R installation path
    if not r_home:
        # https://stackoverflow.com/a/54845971
        r_home = join(dirname(dirname(Path(sys.executable).as_posix())), "lib", "R")

    if not exists(r_home):
        raise Exception(f'Path to R installation does not exist! Make sure R is installed. {r_home}')

    os.environ['R_HOME'] = r_home

    # Initialize R <-> python interface
    utils.check_module("anndata2ri")
    import anndata2ri
    utils.check_module("rpy2")
    from rpy2.robjects import r
    anndata2ri.activate()

    # check if Seurat and SingleCellExperiment are installed
    r("""
        if (!require(Seurat)) {
            stop("R dependency Seurat not found.")
        }
        if (!require(SingleCellExperiment)) {
            stop("R dependecy SingleCellExperiment not found.)
        }
    """)

    # ----- convert to anndata ----- #
    adata = r(f"""
                library(Seurat)

                # ----- load object ----- #
                if (endswith(lower("{file}"), ".robj")) {{
                    # load file; returns vector of created variables
                    new_vars <- load("{file}")
                    # store new variable into another variable to work on
                    object <- get(file[1])
                }} else if(endswith(lower("{file}"), ".rds")) {{
                    # load object
                    object <- readRDS("{file}")
                }} else {{
                    stop("Unknown file extension. Expected '.robj' or '.rds' got ", {file})
                }}

                # ----- convert to SingleCellExperiment ----- #
                # SingleCellExperiment is needed for anndata conversion
                if (class(object) == "Seurat") {{
                    object <- as.SingleCellExperiment(object)
                }} else if (class(object) == "SingleCellExperiment") {{
                    object <- object
                }} else {{
                    stop("Unknown object! Expected class 'Seurat' or 'SingleCellExperiment' got ", class(object))
                }}
            """)

    # Saving adata.h5ad
    h5ad_file = out + '/anndata_1.h5ad'
    adata.write(filename=h5ad_file, compression='gzip')

    return adata
