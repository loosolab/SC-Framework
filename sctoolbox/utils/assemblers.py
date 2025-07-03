"""Module to assemble anndata objects."""

import scanpy as sc
import pandas as pd
import os
import glob
from pathlib import Path
from scipy import sparse
from scipy.io import mmread

from beartype.typing import Optional, Union, Literal, Any, Collection, Mapping
from beartype import beartype

import sctoolbox.utils as utils
from sctoolbox._settings import settings
logger = settings.logger


#####################################################################
#                   Assemble from multiple h5ad files               #
#####################################################################

@beartype
def prepare_atac_anndata(adata: sc.AnnData,
                         coordinate_cols: Optional[Union[list[str], str]] = None,
                         h5ad_path: Optional[str] = None,
                         remove_var_index_prefix: bool = True,
                         keep_original_index: Optional[str] = None,
                         coordinate_regex: str = r"chr[0-9XYM]+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+") -> sc.AnnData:
    r"""
    Prepare AnnData object of ATAC-seq data to be in the correct format for the subsequent pipeline.

    This includes formatting the index, formatting the coordinate columns, and setting the barcode as the index.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object to be prepared.
    coordinate_cols : Optional[list[str], str], default None
        Parameter ensures location info is in adata.var and adata.var.index and formatted correctly.
        1. A list of 3 adata.var column names e.g. ['chr', 'start', 'end'] that will be used to create the index.
        2. A string (adata.var column name) that contains all three coordinates to create the index.
        3. And if None, the coordinates will be created from the index.
    h5ad_path : Optional[str], default None
        Path to the h5ad file.
    remove_var_index_prefix : bool, default True
        If True, the prefix ("chr") of the index will be removed.
    keep_original_index : Optional[str], default None
        If not None, the original index will be kept in adata.obs by the name provided.
    coordinate_regex : str, default "chr[0-9XYM]+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+"
        Regular expression to check if the index is in the correct format.

    Returns
    -------
    sc.AnnData
        The prepared AnnData object.

    Raises
    ------
    ValueError
        If coordinate columns is a list that has not length 3.
    KeyError
        If coordinate columns are not found in adata.var.
    """
    if coordinate_cols:
        if isinstance(coordinate_cols, list):
            if len(coordinate_cols) != 3:
                logger.error("Coordinate columns must be a list of 3 elements. e.g. ['chr', 'start', 'end'] or a single string with all coordinates.")
                raise ValueError

        if not utils.checker.check_columns(adata.var,
                                           coordinate_cols,
                                           error=False,
                                           name="adata.var"):  # Check that coordinate_cols are in adata.var)
            logger.error('Coordinate columns not found in adata.var')
            raise KeyError

    # Format index
    logger.info("formatting index")
    # This checks if the index is available and valid, if not it creates it.
    utils.checker.var_column_to_index(adata,
                                      coordinate_cols=coordinate_cols,
                                      remove_var_index_prefix=remove_var_index_prefix,
                                      keep_original_index=keep_original_index,
                                      coordinate_regex=coordinate_regex)

    # Format coordinate columns
    logger.info("formatting coordinate columns")
    # This checks if the coordinate columns are available and valid, if not it creates them.

    # Establish columns for coordinates
    if coordinate_cols is None:
        coordinate_cols = ['chr', 'start', 'stop']

    # Format coordinate columns
    utils.checker.var_index_to_column(adata, coordinate_cols)

    # check if the barcode is the index otherwise set it
    utils.bioutils.barcode_index(adata)

    if h5ad_path is not None:
        adata.obs = adata.obs.assign(file=h5ad_path)

    return adata


@beartype
def from_h5ad(h5ad_file: Union[str, Collection[str], Mapping[str, str]], label: Optional[str] = "batch") -> sc.AnnData:
    """
    Load one or more .h5ad files.

    Multiple .h5ad files will be combined with a "batch" column added to adata.obs.

    Parameters
    ----------
    h5ad_file : Union[str, Collection[str], Mapping[str, str]]
        Path to one or more .h5ad files. Multiple .h5ad files will cause a "batch" column being added to adata.obs.
        In case of a mapping (dict) the function will populate the "batch" column using the dict-keys.
    label: Optional[str], default "batch"
        Name of the `adata.obs` column to place the batch information in. Forwarded to the `label` parameter of [scanpy.concat](https://anndata.readthedocs.io/en/stable/generated/anndata.concat.html#anndata.concat)

    Returns
    -------
    sc.AnnData
        The loaded anndata object. Multiple files will be combined into one object with a "batch" column in adata.obs.
    """
    if isinstance(h5ad_file, str):
        return sc.read_h5ad(filename=h5ad_file)
    elif isinstance(h5ad_file, Mapping):
        # load then combine anndata objects
        return utils.adata.concadata({k: sc.read_h5ad(f) for k, f in h5ad_file.items()}, label=label)
    else:
        # load then combine anndata objects
        return utils.adata.concadata([sc.read_h5ad(f) for f in h5ad_file], label=label)


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
               dtype: Literal["raw", "filtered"] = "filtered",
               **kwargs: Any) -> sc.AnnData:
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
    **kwargs : Any
        Contains additional arguments for the sctoolbox.utils.assemblers.from_single_starsolo method.

    Returns
    -------
    sc.AnnData
        The assembled anndata object.

    Raises
    ------
    ValueError
        If `use_samples` contains not existing names.
    FileNotFoundError
        If the path to the quant folder does not exist.
    """

    # Test that quant folder exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path to the quant folder does not exist: {path}")

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
        adata = from_single_starsolo(solo_dir, dtype=dtype, **kwargs)

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
    utils.adata.add_uns_info(adata, ["sctoolbox", "source"], os.path.abspath(path))

    return adata


#####################################################################
#          CONVERTING FROM MTX+TSV/CSV TO ANNDATA OBJECT            #
#####################################################################

@beartype
def from_single_mtx(mtx: Union[str, Path],
                    barcodes: Union[str, Path],
                    variables: Optional[Union[str, Path]] = None,
                    transpose: bool = True,
                    header: Union[int, list[int], Literal['infer'], None] = None,
                    barcode_index: int = 0,
                    var_index: Optional[int] = 0,
                    delimiter: str = "\t",
                    comment_flag: str = '#') -> sc.AnnData:
    r"""
    Build an adata object from single mtx and two tsv/csv files.

    Parameters
    ----------
    mtx : Union[str, Path]
        Path to the mtx file (.mtx)
    barcodes : Union[str, Path]
        Path to cell label file (.obs)
    variables : Optional[Union[str, Path]], default None
        Path to variable label file (.var). E.g. gene labels for RNA or peak labels for ATAC.
    transpose : bool, default True
        Set True to transpose mtx matrix.
    header : Union[int, list[int], Literal['infer'], None], default None
        Set header parameter for reading metadata tables using pandas.read_csv.
    barcode_index : int, default 0
        Column which contains the cell barcodes.
    var_index : Optional[int], default 0
        Column containing the variable IDs e.g. gene IDs or peak IDs.
    delimiter : str, default '\t'
        delimiter of the variable and barcode tables.
    comment_flag : str, default '#'
        Comment flag for the variable and barcode tables. Lines starting with this character will be ignored.

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
    adata = sc.read_mtx(filename=mtx, dtype='float32')

    # Transpose matrix if necessary
    if transpose:
        adata = adata.transpose()

    # Read in gene and cell annotation
    barcode_csv = pd.read_csv(barcodes, header=header, index_col=barcode_index, delimiter=delimiter, comment=comment_flag)
    barcode_csv.index.names = ['index']
    barcode_csv.columns = [str(c) for c in barcode_csv.columns]  # convert to string

    if variables:
        # Read in var table
        var_csv = pd.read_csv(variables, header=header, index_col=var_index, delimiter=delimiter, comment=comment_flag)
        var_csv.index.names = ['index']
        var_csv.columns = [str(c) for c in var_csv.columns]  # convert to string

    # Test if they are unique
    if not barcode_csv.index.is_unique:
        raise ValueError("Barcode index column does not contain unique values")
    if variables and not var_csv.index.is_unique:
        raise ValueError("Genes index column does not contain unique values")

    # Add tables to anndata object
    adata.obs = barcode_csv
    if variables:
        adata.var = var_csv

    # Add filename to .obs
    adata.obs["filename"] = os.path.basename(mtx)
    adata.obs["filename"] = adata.obs["filename"].astype("category")

    return adata


@beartype
def from_mtx(path: str,
             mtx: str = "*matrix.mtx*",
             barcodes: str = "*barcodes.tsv*",
             variables: str = "*genes.tsv*",
             var_error: bool = True,
             **kwargs: Any) -> sc.AnnData:
    """
    Build an adata object from list of mtx, barcodes and variables files.

    This function recursively scans trough all subdirectories of the given path for files matching the pattern of the `mtx` parameter.
    Mtx files accompanied by a file matching the `barcodes` or `variables` pattern will be added to the resulting AnnData as `.obs` and `.var` tables.

    Parameters
    ----------
    path : str
        Path to data files
    mtx : str, default '*matrix.mtx*'
        String for glob to find matrix files.
    barcodes : str, default '*barcodes.tsv*'
        String for glob to find barcode files.
    variables : str, default '*genes.tsv*'
        String for glob to find e.g. gene label files (RNA).
    var_error : bool, default True
        Will raise an error when there is no variables file found next to any .mtx file. Set the parameter to False will consider the variable file optional.
    **kwargs : Any
        Contains additional arguments for the sctoolbox.utils.assemblers.from_single_mtx method.


    Returns
    -------
    sc.AnnData
        Merged anndata object containing the mtx matrix, gene and cell labels

    Raises
    ------
    ValueError
        1. If mtx files are not found.
        2. If multiple barcode or variable files were found for one mtx file.
        3. If the barcode file is missing for mtx file.
        4. If the variable file is missing for mtx file and var_error is set to True
    """
    # initialize path as path object
    path = Path(path)

    # recursively find mtx files
    mtx_files = list(path.rglob(mtx))

    if not mtx_files:
        raise ValueError('No files were found with the given directory and suffixes.')

    adata_objects = []
    for i, m in enumerate(mtx_files):
        logger.info(f"Reading files: {i+1} of {len(mtx_files)} ")

        # find barcode and variable file in same folder
        barcode_file = list(m.parents[0].glob(barcodes))
        if len(barcode_file) > 1:
            raise ValueError(f'{str(m)} expected one barcode file but found multiple: {[str(bf) for bf in barcode_file]}')
        elif len(barcode_file) < 1:
            raise ValueError(f'Missing required barcode file for {str(m)}.')

        variable_file = list(m.parents[0].glob(variables))
        if len(variable_file) > 1:
            raise ValueError(f'{str(m)} expected one variable file but found multiple: {[str(vf) for vf in variable_file]}')
        elif var_error and len(variable_file) < 1:
            raise ValueError(f'Missing required variable file for {str(m)}.')

        # create adata object
        adata_objects.append(
            from_single_mtx(m,
                            barcode_file[0],
                            variable_file[0] if variable_file else None,
                            **kwargs)
        )

        # add relative path to obs as it could contain e.g. sample information
        adata_objects[-1].obs["rel_path"] = str(m.parents[0].relative_to(path))

    # create final adata
    if len(adata_objects) > 1:
        adata = utils.adata.concadata(adata_objects)
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
    utils.general.setup_R(r_home)

    # Initialize R <-> python interface
    utils.checker.check_module("anndata2ri")
    import anndata2ri
    utils.checker.check_module("rpy2")
    from rpy2.robjects import r, default_converter, conversion, globalenv
    anndata2ri.activate()

    # create rpy2 None to NULL converter
    # https://stackoverflow.com/questions/65783033/how-to-convert-none-to-r-null
    none_converter = conversion.Converter("None converter")
    none_converter.py2rpy.register(type(None), utils.general._none2null)

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

        # update/validate SeuratObject to match newer versions
        object = UpdateSeuratObject(object)

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
    utils.adata.add_uns_info(adata, ["sctoolbox", "source"], os.path.abspath(file))

    if output:
        # Saving adata.h5ad
        adata.write(filename=output, compression='gzip')
    else:
        return adata
