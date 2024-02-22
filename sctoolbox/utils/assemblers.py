"""Module to assemble anndata objects."""

import scanpy as sc
import pandas as pd
import os
import glob
from scipy import sparse
from scipy.io import mmread

from beartype.typing import Optional, Union, Literal, Any
from beartype import beartype

import sctoolbox.utils as utils
from sctoolbox._settings import settings
logger = settings.logger


#####################################################################
#                   Assemble from multiple h5ad files               #
#####################################################################

@beartype
def prepare_atac_anndata(adata: sc.AnnData,
                         set_index: bool = True,
                         index_from: Optional[str] = None,
                         coordinate_cols: Optional[list[str]] = None,
                         h5ad_path: Optional[str] = None) -> sc.AnnData:
    """
    Prepare AnnData object of ATAC-seq data to be in the correct format for the subsequent pipeline.

    This includes formatting the index, formatting the coordinate columns, and setting the barcode as the index.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to be prepared.
    set_index : bool, default True
        If True, index will be formatted and can be set by a given column.
    index_from : Optional[str], default None
        Column to build the index from.
    coordinate_cols : Optional[list[str]], default None
        Location information of the peaks.
    h5ad_path : Optional[str], default None
        Path to the h5ad file.

    Returns
    -------
    sc.AnnData
        The prepared AnnData object.

    """

    if set_index:
        logger.info("formatting index")
        utils.var_index_from(adata, index_from)

    # Establish columns for coordinates
    if coordinate_cols is None:
        coordinate_cols = adata.var.columns[:3]  # first three columns are coordinates
    else:
        utils.check_columns(adata.var,
                            coordinate_cols,
                            name="adata.var")  # Check that coordinate_cols are in adata.var)

    # Format coordinate columns
    logger.info("formatting coordinate columns")
    utils.format_adata_var(adata, coordinate_cols, coordinate_cols)

    # check if the barcode is the index otherwise set it
    utils.barcode_index(adata)

    if h5ad_path is not None:
        adata.obs = adata.obs.assign(file=h5ad_path)

    return adata


#####################################################################
#          ASSEMBLING ANNDATA FROM STARSOLO OUTPUT FOLDERS          #
#####################################################################

@beartype
def from_single_starsolo(path: str,
                         dtype: Literal['filtered', 'raw'] = "filtered",
                         header: Union[int, list[int], Literal['infer'], None] = 'infer') -> sc.AnnData:
    """
    Assembles an anndata object from the starsolo folder.

    Parameters
    ----------
    path : str
        Path to the "solo" folder from starsolo.
    dtype : Literal['filtered', 'raw'], default "filtered"
        The type of solo data to choose.
    header : Union[int, list[int], Literal['infer'], None], default "infer"
        Set header parameter for reading metadata tables using pandas.read_csv.

    Returns
    -------
    sc.AnnData
        An anndata object based on the provided starsolo folder.

    Raises
    ------
    FileNotFoundError
        If path does not exist or files are missing.
    """

    # Establish which directory to look for data in
    genedir = os.path.join(path, "Gene", dtype)
    velodir = os.path.join(path, "Velocyto", dtype)
    for path in [genedir, velodir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path to the data does not exist: {path}")

    # File paths for different elements
    matrix_f = os.path.join(genedir, 'matrix.mtx')
    barcodes_f = os.path.join(genedir, "barcodes.tsv")
    genes_f = os.path.join(genedir, "genes.tsv")
    spliced_f = os.path.join(velodir, 'spliced.mtx')
    unspliced_f = os.path.join(velodir, 'unspliced.mtx')
    ambiguous_f = os.path.join(velodir, 'ambiguous.mtx')

    # Check whether files are present
    for f in [matrix_f, barcodes_f, genes_f, spliced_f, unspliced_f, ambiguous_f]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File '{f}' was not found. Please check that path contains the full output of starsolo.")

    # Setup main adata object from matrix/barcodes/genes
    logger.info("Setting up adata from solo files")
    adata = from_single_mtx(matrix_f, barcodes_f, genes_f, header=header)
    adata.var.columns = ["gene", "type"]  # specific to the starsolo format

    # Add in velocity information
    logger.info("Adding velocity information from spliced/unspliced/ambiguous")
    spliced = sparse.csr_matrix(mmread(spliced_f).transpose())
    unspliced = sparse.csr_matrix(mmread(unspliced_f).transpose())
    ambiguous = sparse.csr_matrix(mmread(ambiguous_f).transpose())

    adata.layers["spliced"] = spliced
    adata.layers["unspliced"] = unspliced
    adata.layers["ambiguous"] = ambiguous

    return adata


@beartype
def from_quant(path: str,
               configuration: list = [],
               use_samples: Optional[list] = None,
               dtype: Literal["raw", "filtered"] = "filtered") -> sc.AnnData:
    """
    Assemble an adata object from data in the 'quant' folder of the snakemake pipeline.

    Parameters
    ----------
    path : str
        The directory where the quant folder from snakemake preprocessing is located.
    configuration : list
        Configurations to setup the samples for anndata assembling.
        It must containg the sample, the word used in snakemake to assign the condition,
        and the condition, e.g., sample1:condition:room_air
    use_samples : Optional[list], default None
        List of samples to use. If None, all samples will be used.
    dtype : Literal["raw", "filtered"], default 'filtered'
        The type of Solo data to choose.

    Returns
    -------
    sc.AnnData
        The assembled anndata object.

    Raises
    ------
    ValueError
        If `use_samples` contains not existing names.
    """

    # TODO: test that quant folder is existing

    # Collect configuration into a dictionary
    config_dict = {}
    if configuration is not None:
        for string in configuration:
            sample, condition, condition_name = string.split(":")

            if sample not in config_dict:
                config_dict[sample] = {}
            config_dict[sample][condition] = condition_name

    # Establishing which samples to use for assembly
    sample_dirs = glob.glob(os.path.join(path, "*"))
    sample_names = [os.path.basename(x) for x in sample_dirs]
    logger.info(f"Found samples: {sample_names}")

    # Subset to use_samples if they are provided
    if use_samples is not None:
        idx = [i for i, sample in enumerate(sample_names) if sample in use_samples]
        sample_names = [sample_names[i] for i in idx]
        sample_dirs = [sample_dirs[i] for i in idx]

        logger.info(f"Using samples: {sample_names}")
        if len(sample_names) == 0:
            raise ValueError("None of the given 'use_samples' match the samples found in the directory.")

    # Assembling from different samples:
    adata_list = []
    for sample_name, sample_dir in zip(sample_names, sample_dirs):

        logger.info(f"Assembling sample '{sample_name}'")
        solo_dir = os.path.join(sample_dir, "solo")
        adata = from_single_starsolo(solo_dir, dtype=dtype)

        # Make barcode index unique
        adata.obs.index = adata.obs.index + "-" + sample_name
        adata.obs["sample"] = sample_name

        # Add additional information from configuration
        if sample_name in config_dict:
            for key in config_dict[sample_name]:
                adata.obs[key] = config_dict[sample_name][key]
                adata.obs[key] = adata.obs[key].astype("category")

        adata_list.append(adata)

    # Concatenating the adata objects
    logger.info("Concatenating anndata objects")
    adata = adata_list[0].concatenate(adata_list[1:], join="outer")

    # Add information to uns
    utils.add_uns_info(adata, ["sctoolbox", "source"], os.path.abspath(path))

    return adata


#####################################################################
#          CONVERTING FROM MTX+TSV/CSV TO ANNDATA OBJECT            #
#####################################################################

@beartype
def from_single_mtx(mtx: str,
                    barcodes: str,
                    genes: str,
                    transpose: bool = True,
                    header: Union[int, list[int], Literal['infer'], None] = 'infer',
                    barcode_index: int = 0,
                    genes_index: int = 0,
                    delimiter: str = "\t",
                    **kwargs: Any) -> sc.AnnData:
    r"""
    Build an adata object from single mtx and two tsv/csv files.

    Parameters
    ----------
    mtx : str
        Path to the mtx file (.mtx)
    barcodes : str
        Path to cell label file (.obs)
    genes : str
        Path to gene label file (.var)
    transpose : bool, default True
        Set True to transpose mtx matrix.
    header : Union[int, list[int], Literal['infer'], None], default 'infer'
        Set header parameter for reading metadata tables using pandas.read_csv.
    barcode_index : int, default 0
        Column which contains the cell barcodes.
    genes_index : int, default 0
        Column which contains the gene IDs.
    delimiter : str, default '\t'
        delimiter of genes and barcodes table.
    **kwargs : Any
        Contains additional arguments for scanpy.read_mtx method

    Returns
    -------
    sc.AnnData
        Anndata object containing the mtx matrix, gene and cell labels

    Raises
    ------
    ValueError
        If barcode or gene files contain duplicates.
    """

    # Read mtx file
    adata = sc.read_mtx(filename=mtx, dtype='float32', **kwargs)

    # Transpose matrix if necessary
    if transpose:
        adata = adata.transpose()

    # Read in gene and cell annotation
    barcode_csv = pd.read_csv(barcodes, header=header, index_col=barcode_index, delimiter=delimiter)
    barcode_csv.index.names = ['index']
    barcode_csv.columns = [str(c) for c in barcode_csv.columns]  # convert to string
    genes_csv = pd.read_csv(genes, header=header, index_col=genes_index, delimiter=delimiter)
    genes_csv.index.names = ['index']
    genes_csv.columns = [str(c) for c in genes_csv.columns]  # convert to string

    # Test if they are unique
    if not barcode_csv.index.is_unique:
        raise ValueError("Barcode index column does not contain unique values")
    if not genes_csv.index.is_unique:
        raise ValueError("Genes index column does not contain unique values")

    # Add tables to anndata object
    adata.obs = barcode_csv
    adata.var = genes_csv

    # Add filename to .obs
    adata.obs["filename"] = os.path.basename(mtx)
    adata.obs["filename"] = adata.obs["filename"].astype("category")

    return adata


@beartype
def from_mtx(path: str,
             mtx: str = "*_matrix.mtx*",
             barcodes: str = "*_barcodes.tsv*",
             genes: str = "*_genes.tsv*",
             **kwargs: Any) -> sc.AnnData:
    """
    Build an adata object from list of mtx, barcodes and genes files.

    Parameters
    ----------
    path : str
        Path to data files
    mtx : str, default '*_matrix.mtx*'
        String for glob to find matrix files.
    barcodes : str, default '*_barcodes.tsv*'
        String for glob to find barcode files.
    genes : str, default '*_genes.tsv*'
        String for glob to find gene label files.
    **kwargs : Any
        Contains additional arguments for scanpy.read_mtx method

    Returns
    -------
    sc.AnnData
        Merged anndata object containing the mtx matrix, gene and cell labels

    Raises
    ------
    ValueError
        If files are not found.
    """

    mtx = glob.glob(os.path.join(path, mtx))
    barcodes = glob.glob(os.path.join(path, barcodes))
    genes = glob.glob(os.path.join(path, genes))

    # Check if lists are same length
    # https://stackoverflow.com/questions/35791051/better-way-to-check-if-all-lists-in-a-list-are-the-same-length
    it = iter([mtx, barcodes, genes])
    the_len = len(next(it))
    if not all(len(list_len) == the_len for list_len in it):
        raise ValueError("Found different quantitys of mtx, genes, barcode files.\n Please check given suffixes or filenames")

    if not mtx:
        raise ValueError('No files were found with the given directory and suffixes')

    adata_objects = []
    for i, m in enumerate(mtx):
        print(f"Reading files: {i+1} of {len(mtx)} ")
        adata_objects.append(from_single_mtx(m, barcodes[i], genes[i], **kwargs))

    if len(adata_objects) > 1:
        adata = adata_objects[0].concatenate(*adata_objects[1:], join="outer")
    else:
        adata = adata_objects[0]

    return adata


@beartype
def convertToAdata(file: str,
                   output: Optional[str] = None,
                   r_home: Optional[str] = None,
                   layer: Optional[str] = None) -> Optional[sc.AnnData]:
    """
    Convert .rds files containing Seurat or SingleCellExperiment to scanpy anndata.

    In order to work an R installation with Seurat & SingleCellExperiment is required.

    Parameters
    ----------
    file : str
        Path to the .rds or .robj file.
    output : Optional[str], default None
        Path to output .h5ad file. Won't save if None.
    r_home : Optional[str], default None
        Path to the R home directory. If None will construct path based on location of python executable.
        E.g for ".conda/scanpy/bin/python" will look at ".conda/scanpy/lib/R"
    layer : Optional[str], default None
        Provide name of layer to be stored in anndata. By default the main layer is stored.
        In case of multiome data multiple layers are present e.g. RNA and ATAC. But anndata can only store a single layer.

    Returns
    -------
    Optional[sc.AnnData]
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
            stop("Unknown file. Expected '.robj' or '.rds' got", file)
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

    # fixes https://gitlab.gwdg.de/loosolab/software/sc_framework/-/issues/205
    adata.obs.index = adata.obs.index.astype('object')
    adata.var.index = adata.var.index.astype('object')

    # Add information to uns
    utils.add_uns_info(adata, ["sctoolbox", "source"], os.path.abspath(file))

    if output:
        # Saving adata.h5ad
        adata.write(filename=output, compression='gzip')
    else:
        return adata
