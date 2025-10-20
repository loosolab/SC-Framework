"""Module to assemble anndata objects."""

import scanpy as sc
import pandas as pd
import os
import glob
from pathlib import Path
from scipy import sparse
from scipy.io import mmread

from beartype.typing import Optional, Union, Literal, Any, Collection, Mapping, Callable
from beartype import beartype

import sctoolbox.utils as utils
from sctoolbox.plotting.general import plot_table
from sctoolbox._settings import settings
logger = settings.logger


#####################################################################
#                   Assemble from multiple h5ad files               #
#####################################################################

@beartype
def prepare_atac_anndata(adata: sc.AnnData,
                         coordinate_cols: Optional[Union[list[str], str]] = None,
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

    return adata


@beartype
def from_h5ad(h5ad_file: Union[str, Collection[str], Mapping[str, str]],
              report: Optional[str] = None,
              label: Optional[str] = "batch") -> sc.AnnData:
    """
    Load one or more .h5ad files.

    Multiple .h5ad files will be combined with a "batch" column added to adata.obs.

    Parameters
    ----------
    h5ad_file : Union[str, Collection[str], Mapping[str, str]]
        Path to one or more .h5ad files. Multiple .h5ad files will cause a "batch" column being added to adata.obs.
        In case of a mapping (dict) the function will populate the "batch" column using the dict-keys.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    label: Optional[str], default "batch"
        Name of the `adata.obs` column to place the batch information in. Forwarded to the `label` parameter of [scanpy.concat](https://anndata.readthedocs.io/en/stable/generated/anndata.concat.html#anndata.concat)

    Returns
    -------
    sc.AnnData
        The loaded anndata object. Multiple files will be combined into one object with a "batch" column in adata.obs.
    """

    return _read_and_merge(h5ad_file, sc.read_h5ad, label, report)


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
               report: Optional[str] = None,
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
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
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

    # generate and save report
    if settings.report_dir and report:
        info_table = {}

        for sample_name, sample_dir in zip(sample_names, sample_dirs):
            info_table.setdefault("Name", []).append(sample_name)
            info_table.setdefault("Source", []).append(sample_dir)

        # save table
        plot_table(table=pd.DataFrame(info_table), report=report, show_index=False)

    return adata


#####################################################################
#          CONVERTING FROM MTX+TSV/CSV TO ANNDATA OBJECT            #
#####################################################################

@beartype
def from_single_mtx(mtx: Union[str, Path],
                    barcodes: Union[str, Path],
                    variables: Optional[Union[str, Path]] = None,
                    transpose: bool = True,
                    header: Union[int, list[int], Literal['infer'], None] = 'infer',
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
        Automatically tries to enable/disable the header if the length of the mtx is one different to either variables or barcodes.
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
        1. If barcode or gene files contain duplicates.
        2. Variables or barcodes does not fit mtx size.
    """

    # Read mtx file
    adata = sc.read_mtx(filename=mtx, dtype='float32')

    # Transpose matrix if necessary
    if transpose:
        adata = adata.transpose()

    # statisfy the linter
    if False:
        raise ValueError()

    def load_meta(file, header, index_col, delimiter, comment, expected_size, name):
        """Load and prepare AnnData.var or AnnData.obs."""
        # load the file
        table = pd.read_csv(file, header=header, index_col=index_col, delimiter=delimiter, comment=comment)

        # check if the size is expected; try to fix when the size is one of
        if len(table) - 1 == expected_size:
            logger.warning(f"{name} file is one less than expected. Trying to fix by disabling the header.")
            table = pd.read_csv(file, header=0, index_col=index_col, delimiter=delimiter, comment=comment)
        elif len(table) + 1 == expected_size:
            logger.warning(f"{name} file is one more than expected. Trying to fix by enabling the header.")
            table = pd.read_csv(file, header=None, index_col=index_col, delimiter=delimiter, comment=comment)

        if len(table) != expected_size:
            raise ValueError(f"{name} file is of size {len(table)} but AnnData expects {expected_size}. Try to toggle the transpose argument.")

        table.index.names = ["index"]
        table.columns = [str(c) for c in table.columns]  # conver to string

        # Test if they are unique
        if not table.index.is_unique:
            raise ValueError(f"{name} index column does not contain unique values")

        return table

    # Add tables to anndata object
    adata.obs = load_meta(barcodes, header, barcode_index, delimiter, comment_flag, len(adata.obs), "Barcodes")
    if variables:
        adata.var = load_meta(variables, header, var_index, delimiter, comment_flag, len(adata.var), "Variables")

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
             report: Optional[str] = None,
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
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
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

    # generate and save report
    if settings.report_dir and report:
        info_table = {}

        for m in mtx_files:
            info_table.setdefault("Name", []).append("NA")
            info_table.setdefault("Source", []).append(str(m.parents[0]))

        # save table
        plot_table(table=pd.DataFrame(info_table), report=report, show_index=False)

    return adata


@beartype
def convertToAdata(file: str,
                   output: Optional[str] = None,
                   r_home: Optional[str] = None,
                   layer: Optional[str] = None,
                   report: Optional[str] = None) -> Optional[sc.AnnData]:
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
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.

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

    # create rpy2 None to NULL converter
    # https://stackoverflow.com/questions/65783033/how-to-convert-none-to-r-null
    none_converter = conversion.Converter("None converter")
    none_converter.py2rpy.register(type(None), utils.general._none2null)

    with conversion.localconverter(default_converter + none_converter):
        # check if Seurat and SingleCellExperiment are installed
        r("""
            if (!suppressPackageStartupMessages(require(Seurat))) {
                stop("R dependency Seurat not found.")
            }
            if (!suppressPackageStartupMessages(require(SingleCellExperiment))) {
                stop("R dependecy SingleCellExperiment not found.")
            }
        """)

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
            object <- UpdateSeuratObject(object)

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

    with conversion.localconverter(anndata2ri.converter):
        # pull SingleCellExperiment into python
        # this also converts to anndata
        adata = globalenv["object"]

    # fixes https://gitlab.gwdg.de/loosolab/software/sc_framework/-/issues/205
    adata.obs.index = adata.obs.index.astype('object')
    adata.var.index = adata.var.index.astype('object')

    # Add information to uns
    utils.adata.add_uns_info(adata, ["sctoolbox", "source"], os.path.abspath(file))

    # generate and save report
    if settings.report_dir and report:
        info_table = {
            "Name": ["NA"],
            "Source": [os.path.abspath(file)]
        }

        # save table
        plot_table(table=pd.DataFrame(info_table), report=report, show_index=False)

    if output:
        # Saving adata.h5ad
        adata.write(filename=output, compression='gzip')
    else:
        return adata


@beartype
def from_R(
        rds_file: Union[str, Collection[str], Mapping[str, str]],
        label: Optional[str] = "batch",
        output: Optional[str] = None,
        report: Optional[str] = None,
        layer: Optional[Union[str, Collection[str], Mapping[str, str]]] = None,
        r_home: Optional[str] = None) -> Optional[sc.AnnData]:
    """
    Convert one or more .rds/.robj file(s) containing Seurat or SingleCellExperiment to scanpy anndata.

    In order to work an R installation with Seurat & SingleCellExperiment is required.

    Parameters
    ----------
    rds_file : Union[str, Collection[str], Mapping[str, str]]
        Path or list of paths to the .rds or .robj file(s).
    label: Optional[str], default "batch"
        Name of the `adata.obs` column to place the batch information in.
        Forwarded to the `label` parameter of [scanpy.concat](https://anndata.readthedocs.io/en/stable/generated/anndata.concat.html#anndata.concat)
    output : Optional[str], default None
        Path to output .h5ad file. Won't save if None.
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    layer : Optional[Union[str, Collection[str], Mapping[str, str]]], default None
        Provide name of layer to be stored in anndata. By default the main layer is stored.
        In case of multiome data multiple layers are present e.g. RNA and ATAC. But anndata can only store a single layer.
        Must be string or match type of rds_file.
    r_home : Optional[str], default None
        Path to the R home directory. If None will construct path based on location of python executable.
        E.g for ".conda/scanpy/bin/python" will look at ".conda/scanpy/lib/R"

    Returns
    -------
    Optional[sc.AnnData]
        Returns converted (and merged) anndata object or None when output is set.
    """

    adata = _read_and_merge(rds_file, convertToAdata, label, report, layer=layer, r_home=r_home)

    if output:
        # Saving adata.h5ad
        adata.write(filename=output, compression='gzip')
    else:
        return adata


@beartype
def _read_and_merge(
        path: Union[str, Collection[str], Mapping[str, str]],
        method: Callable,
        label: Optional[str] = "batch",
        report: Optional[str] = None,
        **kwargs: Any) -> sc.AnnData:
    """
    Read in one or multiple files and convert/merge them to one anndata object.

    Parameters
    ----------
    path : Union[str, Collection[str], Mapping[str, str]]
        Path or list of paths to the input file(s).
    method : Callable
        Method for reading individual files. Set depending on file format e.g.
        scanpy.read_h5ad for h5ad files.
    label: Optional[str], default "batch"
        Name of the `adata.obs` column to place the batch information in.
        Forwarded to the `label` parameter of [scanpy.concat](https://anndata.readthedocs.io/en/stable/generated/anndata.concat.html#anndata.concat)
    report : Optional[str]
        Name of the output file used for report creation. Will be silently skipped if `sctoolbox.settings.report_dir` is None.
    **kwargs : Any
        Contains additional arguments for reading method.

    Raises
    ------
    ValueError
        1. layer datatype and path datatype do not match (if layer is not string).
        2. if path and layer are lists with different lengths.
        3. if not all keys in path dict match with keys in layer dict.

    Notes
    -----
    This function is designed to work with functions that return a h5ad file and does not require file specific input.
    Parameters with different values per file needs to be handled specifically.
    Current special cases:
        - layer (e.g. used by utils.assemblers.convertToAdata)

    Returns
    -------
    sc.AnnData
        Returns converted (and merged) anndata object.
    """

    # Checks for layer option used by convertToAdata method
    has_layer = False
    if "layer" in kwargs and kwargs["layer"] is not None:
        has_layer = True
        layer_is_not_string = not isinstance(kwargs["layer"], str)
        layer_has_matching_type = isinstance(kwargs["layer"], type(path))
        if not layer_has_matching_type and layer_is_not_string:
            raise ValueError("layer datatype must match input datatype or be of type str")
        elif layer_has_matching_type and layer_is_not_string:
            if isinstance(path, list):
                if len(path) != len(kwargs["layer"]):
                    raise ValueError("When giving a list path and layer must be of same length.")
            elif isinstance(path, dict):
                if not all(key in kwargs["layer"] for key in path):
                    raise ValueError("Missing keys in layer dict.")

    # source is stored in adata.uns["sctoolbox"]["source"]
    # TODO move for-loop/kwargs handling into seperate function
    if isinstance(path, str):
        adata = method(path, **kwargs)
        source = os.path.abspath(path)
    elif isinstance(path, Mapping):
        # load then combine anndata objects
        sub_kwargs = kwargs.copy()
        adatas, source = dict(), dict()
        for k, f in path.items():
            if has_layer and layer_is_not_string:
                sub_kwargs = {"layer": kwargs["layer"][k]}
            adatas[k] = method(f, **sub_kwargs)
            source[k] = os.path.abspath(f)
        adata = utils.adata.concadata(adatas, label=label)
    else:
        # load then combine anndata objects
        adatas, source = list(), list()
        sub_kwargs = kwargs.copy()
        for i, f in enumerate(path):
            if has_layer and layer_is_not_string:
                sub_kwargs = {"layer": kwargs["layer"][i]}
            adatas.append(method(f, **sub_kwargs))
            source.append(os.path.abspath(f))
        adata = utils.adata.concadata(adatas, label=label)

    # generate and save report
    if settings.report_dir and report:
        info_table = {}

        if isinstance(path, str):
            info_table.setdefault("Name", []).append("NA")
            info_table.setdefault("Source", []).append(path)
        elif isinstance(path, Mapping):
            for k, v in path.items():
                info_table.setdefault("Name", []).append(k)
                info_table.setdefault("Source", []).append(v)
        else:
            for v in path:
                info_table.setdefault("Name", []).append("NA")
                info_table.setdefault("Source", []).append(v)

        # save table
        plot_table(table=pd.DataFrame(info_table), report=report, show_index=False)

    # Add information to uns
    utils.adata.add_uns_info(adata, ["sctoolbox", "source"], source)

    return adata
