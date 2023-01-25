import sctoolbox.utilities as utils


def convertToAdata(file, output=None, r_home=None, layer=None):
    """
    Converts .rds files containing Seurat or SingleCellExperiment to scanpy anndata.

    In order to work an R installation with Seurat & SingleCellExperiment is required.

    Parameters
    ----------
    file : str
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
        Returns converted anndata object if output is None.
    """
    # Setup R
    utils.setup_R(r_home)

    # Initialize R <-> python interface
    utils.check_module("anndata2ri")
    import anndata2ri
    utils.check_module("rpy2")
    from rpy2.robjects import r, default_converter, conversion, globalenv
    anndata2ri.activate()

    # create rpy2 None to NULL converter
    # https://stackoverflow.com/questions/65783033/how-to-convert-none-to-r-null
    none_converter = conversion.Converter("None converter")
    none_converter.py2rpy.register(type(None), utils._none2null)

    # check if Seurat and SingleCellExperiment are installed
    r("""
        if (!suppressPackageStartupMessages(require(Seurat))) {
            stop("R dependency Seurat not found.")
        }
        if (!suppressPackageStartupMessages(require(SingleCellExperiment))) {
            stop("R dependecy SingleCellExperiment not found.")
        }
    """)

    # add variables into R
    with conversion.localconverter(default_converter + none_converter):
        globalenv["file"] = file
        globalenv["layer"] = layer

    # ----- convert to anndata ----- #
    r("""
        # ----- load object ----- #
        # try loading .robj
        object <- try({
            # load file; returns vector of created variables
            new_vars <- load(file)
            # store new variable into another variable to work on
            get(new_vars[1])
        }, silent = TRUE)

        # if .robj failed try .rds
        if (class(object) == "try-error") {
            # load object
            object <- try(readRDS(file), silent = TRUE)
        }

        # if both .robj and .rds failed throw error
        if (class(object) == "try-error") {
            stop("Unknown file extension. Expected '.robj' or '.rds' got", file)
        }

        # ----- convert to SingleCellExperiment ----- #
        # can only convert Seurat -> SingleCellExperiment -> anndata
        if (class(object) == "Seurat") {
            object <- as.SingleCellExperiment(object)
        } else if (class(object) == "SingleCellExperiment") {
            object <- object
        } else {
            stop("Unknown object! Expected class 'Seurat' or 'SingleCellExperiment' got ", class(object))
        }

        # ----- change layer ----- #
        # adata can only store a single layer
        if (!is.null(layer)) {
            layers <- c(mainExpName(object), altExpNames(object))

            # check if layer is valid
            if (!layer %in% layers) {
                stop("Invalid layer! Expected one of ", paste(layers, collapse = ", "), " got ", layer)
            }

            # select layer
            if (layer != mainExpName(object)) {
                object <- swapAltExp(object, layer, saved = mainExpName(object), withColData = TRUE)
            }
        }
    """)

    # pull SingleCellExperiment into python
    # this also converts to anndata
    adata = globalenv["object"]

    if output:
        # Saving adata.h5ad
        adata.write(filename=output, compression='gzip')
    else:
        return adata
