import os
from os.path import join, dirname, exists
import sys
from pathlib import Path

import sctoolbox.utilities as utils


def convertToAdata(file, output=None, r_home=None, layer=None):
    """
    Converts .rds files containing Seurat or SingleCellExperiment to scanpy anndata.

    In order to work an R installation with Seurat & SingleCellExperiment is required.

    Parameters
    ----------
    path : str
        Path to the .rds or .robj file.
    output : str, default None
        Path to output .h5ad file. Won't save if None.
    r_home : str, default None
        Path to the R home directory. If None will construct path based on location of python executable.
        E.g for ".conda/scanpy/bin/python" will look at ".conda/scanpy/lib/R"
    layer : str, default None
        Provide name of layer to be stored in anndata. By default the main layer is stored.
        In case of multiome data multiple layers are present e.g. RNA and ATAC. But anndata can only store a single layer.

    Returns
    -------
    anndata.AnnData or None:
        Returns converted anndata object if out_prefix is None.
    """
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
    from rpy2.robjects import r, default_converter, conversion
    anndata2ri.activate()

    # create rpy2 None to NULL converter
    # https://stackoverflow.com/questions/65783033/how-to-convert-none-to-r-null
    none_converter = conversion.Converter("None converter")
    none_converter.py2rpy.register(type(None), _none2null)

    # check if Seurat and SingleCellExperiment are installed
    r("""
        if (!require(Seurat)) {
            stop("R dependency Seurat not found.")
        }
        if (!require(SingleCellExperiment)) {
            stop("R dependecy SingleCellExperiment not found.")
        }
    """)

    # ----- convert to anndata ----- #
    with conversion.localconverter(default_converter + none_converter):
        adata = r(f"""
                    library(Seurat)

                    # ----- load object ----- #
                    if (endswith(lower("{file}"), ".robj")) {{
                        # load file; returns vector of created variables
                        new_vars <- load("{file}")
                        # store new variable into another variable to work on
                        object <- get(new_vars[1])
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

                    # ----- change layer ----- #
                    # adata can only store a single layer
                    if (!is.null({layer})) {{
                        layers <- c(mainExpName(object), altExpNames(object))

                        # check if layer is valid
                        if (!{layer} %in% layers) {{
                            stop("Invalid layer! Expected one of ", layers, " got, {layer})
                        }}

                        # select layer
                        if ({layer} != mainExpName(object)) {{
                            mainExpName(object) <- {layer}
                        }}
                    }}

                    # return object for conversion
                    object
              """)

    if output:
        # Saving adata.h5ad
        adata.write(filename=output, compression='gzip')
    else:
        return adata

def _none2null(none_obj):
    """ rpy2 converter that translates python 'None' to R 'NULL' """
    # See https://stackoverflow.com/questions/65783033/how-to-convert-none-to-r-null
    return r("NULL")
