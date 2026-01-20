"""Module for type checking functions."""

import re
import os
import sys
import importlib
import matplotlib
import numpy as np
import gzip
import shutil
import scanpy as sc
import pandas as pd

from beartype.typing import Optional, Tuple, Any, Iterable, Union, Literal, Sequence
from beartype import beartype
from numpy.typing import ArrayLike, NDArray

import sctoolbox.utils as utils
from sctoolbox._settings import settings
logger = settings.logger


@beartype
def check_module(module: str, addon: Optional[str] = None) -> None:
    """
    Check if <module> can be imported without error.

    Parameters
    ----------
    module : str
        Name of the module to check.
    addon : Optional[str], default None
        Additional message added to the error.

    Raises
    ------
    ImportError
        If the module is not available for import.
    Exception
        If the module cannot be loaded due to an unknown exception.
    """

    error = 0
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        error = 1
    except Exception:
        raise  # unexpected error loading module

    # Write out error if module was not found
    if error == 1:
        s = f"ERROR: Could not find the '{module}' module on path, but the module is needed for this functionality. Please install this package to proceed."

        if addon:
            s += f"\n{addon}"

        raise ImportError(s)


def _is_interactive() -> bool:
    """
    Check if matplotlib backend is interactive.

    Returns
    -------
    bool
        True if interactive, False otherwise.
    """

    backend = matplotlib.get_backend()

    if backend == 'module://ipympl.backend_nbagg':
        return True
    else:
        return False


@beartype
def _add_path() -> str:
    """
    Add python executables path to environment variable PATH.

    Returns
    -------
    str
        Path to python executables.
    """

    python_exec_dir = os.path.dirname(sys.executable)  # get path to python executable

    if python_exec_dir not in os.environ['PATH']:  # check if path is already in environment variable
        os.environ['PATH'] += os.pathsep + python_exec_dir  # add python executable path to environment variable
        return python_exec_dir
    else:
        return python_exec_dir


#####################################################################
# ------------------------- Type checking ------------------------ #
#####################################################################

@beartype
def _is_gz_file(filepath: str) -> bool:
    """
    Check wheather file is a compressed .gz file.

    Parameters
    ----------
    filepath : str
        Path to file.

    Returns
    -------
    bool
        True if the file is a compressed .gz file.
    """

    with open(filepath, 'rb') as test_f:
        return test_f.read(2) == b'\x1f\x8b'


@beartype
def gunzip_file(f_in: str, f_out: str) -> None:
    """
    Decompress file.

    Parameters
    ----------
    f_in : str
        Path to compressed input file.
    f_out : str
        Destination to decompressed output file.
    """

    with gzip.open(f_in, 'rb') as h_in:
        with open(f_out, 'wb') as h_out:
            shutil.copyfileobj(h_in, h_out)


@beartype
def is_str_numeric(ans: str) -> bool:
    """
    Check if string can be converted to number.

    Parameters
    ----------
    ans : str
        String to check.

    Returns
    -------
    bool
        True if string can be converted to float.
    """

    try:
        float(ans)
        return True
    except ValueError:
        return False


@beartype
def var_column_to_index(adata: sc.AnnData,
                        coordinate_cols: Optional[Union[list[str], str]] = None,
                        remove_var_index_prefix: bool = True,
                        keep_original_index: Optional[str] = None,
                        coordinate_regex: str = r"chr[0-9XYM]+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+") -> None:
    r"""
    Format adata.var index from a specified column or multiple coordinate columns.

    This formats the index of adata.var according to the pattern ["chr", "start", "stop"].
    The adata is changed inplace.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object to reformat.
    coordinate_cols : Optional[str | list[str]], default None
        Column name(s) in adata.var to be set as index.
        If multiple columns are given they are pasted together in the format "chr:start-stop" (requires three columns).
        If None, the first column of adata.var is used.
    remove_var_index_prefix : bool, default True
        If True, the prefix 'chr' is removed from the index.
    keep_original_index : Optional[str], default None
        If not None, the original index is stored in a new column with the given name.
    coordinate_regex : str, default r'chr[0-9XYM]+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+'
        Regex pattern to match the coordinate format.

    Raises
    ------
    ValueError
        If the index cannot be formatted.
    KeyError
        If the column name is not found in adata.var.
    """
    # check if index is already in the correct format
    adata.var.index = adata.var.index.astype(str)
    if not adata.var.index.str.contains(coordinate_regex).all():
        # try to format the index from the given column
        if coordinate_cols is None:
            # get the first index
            entry = list(adata.var.index)[0]
            # check the type of the index
            index_type = _get_index_type(entry, coordinate_regex)
            # check if the index type is known
            if index_type is None:
                logger.error(f'Index type ({entry}) is unknown please provide either the index column name, '
                             'coordinate cols or format the index to chr:start-stop.')
                raise ValueError
            # format the index
            else:
                coordinate_cols = 'original_index'
                # add a column to store the original index
                adata.var[coordinate_cols] = adata.var.index

                try:
                    # format the index
                    _var_index_from_single_col(adata, index_type, coordinate_cols)
                except KeyError as e:
                    # throw the error
                    raise KeyError(f'Error while formatting the index: {e}')

                if keep_original_index:
                    adata.var[keep_original_index] = adata.var[coordinate_cols]

                adata.var.pop(coordinate_cols)

        else:
            # check if from_column is a single column or a list of columns
            if isinstance(coordinate_cols, str):
                logger.info("formatting index from single column.")
                # get the first entry of the column
                entry = list(adata.var[coordinate_cols])[0]
                # check the type of the index
                index_type = _get_index_type(entry, coordinate_regex)
                logger.info(f'Index type: {index_type}')

                # format the index
                _var_index_from_single_col(adata, index_type, coordinate_cols)

            elif isinstance(coordinate_cols, list):

                if len(coordinate_cols) != 3:
                    raise ValueError("coordinate cols must be a list of 3 strings, for single column use a string")
                # get the columns
                else:
                    logger.info('formatting adata.var index from coordinate columns.')
                    chr_list = adata.var[coordinate_cols[0]]
                    start_list = adata.var[coordinate_cols[1]]
                    stop_list = adata.var[coordinate_cols[2]]

                    # Combine into the format "chr:start-stop" per row
                    new_index = [f"{chrom}:{start}-{stop}" for chrom, start, stop in zip(chr_list, start_list, stop_list)]

                    # Set the new index
                    adata.var['new_index'] = new_index
                    adata.var.set_index('new_index', inplace=True)
            else:
                raise ValueError("coordinate cols must be a string or a list of strings")
    else:
        # check if the prefix should be removed from the var. index
        if remove_var_index_prefix:
            logger.info('check if prefix should be removed from the var. index.')
            coordinate_pattern = r"^" + coordinate_regex + r"$"
            if not bool(re.fullmatch(coordinate_pattern, adata.var.index[0])):
                logger.info('removing prefix from the var. index.')

                # get the type of the index
                index_type = _get_index_type(adata.var.index[0], coordinate_regex)

                if index_type is None:
                    logger.info(f'Index type ({adata.var.index[0]}) is unknown, please provide either the index column name, '
                                'coordinate cols or format the index to chr:start-stop.')
                    # format the index
                else:
                    if keep_original_index is None:
                        copy_idx = 'original_index'
                    # add a column to store the original index
                    adata.var[copy_idx] = adata.var.index

                    try:
                        # format the index
                        _var_index_from_single_col(adata, index_type, copy_idx)
                    except KeyError as e:
                        # throw the error
                        raise KeyError(f'Error while formatting the index: {e}')

                    if keep_original_index:
                        adata.var[keep_original_index] = adata.var[copy_idx]

                    adata.var.pop(copy_idx)

        else:
            logger.info('adata.var.index is already in the correct format.')


@beartype
def _var_index_from_single_col(adata: sc.AnnData,
                               index_type: Literal["prefix"],
                               from_column: str,
                               coordinate_pattern: str = r'chr[0-9XYM]+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+') -> None:
    r"""
    Format the index of adata.var from a single column.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object to reformat.
    index_type : str
        The type of the index.
    from_column : str
        Column name in adata.var to be set as index.
    coordinate_pattern : str, default r'chr[0-9XYM]+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+'
        Regex pattern to match the coordinate format.
    """
    # TODO index_type is restricted to be a single value -> redundant
    # index is in the format prefix
    if index_type == "prefix":

        # init empty list to store the new index
        new_index = []
        # loop over the column and extract the coordinates
        for line in adata.var[from_column]:
            new_index.append(re.search(coordinate_pattern, line).group(0))
        adata.var['new_index'] = new_index
        # set the new index
        adata.var.set_index('new_index', inplace=True)


@beartype
def _get_index_type(entry: str, regex: str) -> Optional[str]:
    """
    Check the format of the index by regex.

    Parameters
    ----------
    entry : str
        String to identify the format on.
    regex : str
        Regex pattern to match the coordinate format.

    Returns
    -------
    Optional[str]
        The index format.
    """
    # TODO we need to rethink this function. All it does is: Does pattern match? -> Yes, No
    regex_prefix = r"^.+" + regex  # matches: some_name-chr1:12343-76899

    if re.match(regex_prefix, entry):
        return 'prefix'
    return None


@beartype
def validate_regions(adata: sc.AnnData,
                     coordinate_columns: Iterable[str]) -> bool:
    """
    Check if the regions in adata.var are valid.

    Tests if start <= end.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing the regions to be checked.
    coordinate_columns : Iterable[str]
        List of length 3 for column names in adata.var containing chr, start, end coordinates (in this order).

    Returns
    -------
    bool
        True if all regions are valid.
    """

    # Test whether the three columns are in the right format
    chr, start, end = coordinate_columns

    valid = False
    # Test if coordinate columns are in adata.var
    if utils.checker.check_columns(adata.var, coordinate_columns, name="adata.var", error=False):

        # Test whether the three columns are in the right format
        for _, line in adata.var.to_dict(orient="index").items():
            valid = False

            if isinstance(line[chr], str) and isinstance(line[start], int) and isinstance(line[end], int):
                if line[start] <= line[end]:  # start must be smaller than end
                    valid = True  # if all tests passed, the line is valid

            if valid is False:
                logger.info("The region {0}:{1}-{2} is not a valid genome region. Please check the format of columns: {3}".format(line[chr], line[start], line[end], coordinate_columns))
                return valid

    return valid


@beartype
def var_index_to_column(adata: sc.AnnData,
                        coordinate_columns: NDArray[str] | Sequence[str] | pd.core.indexes.base.Index = ["chr", "start", "end"]) -> None:
    """
    Format the index of adata.var and adds peak location columns (chr, start, end) to adata.var if needed.

    If coordinate_columns are given, the function will check if these columns already contain the information needed. If the coordinate_columns are in the correct format, nothing will be done.
    If the coordinate_columns are invalid (or coordinate_columns is not given) the index is checked for the following format:
    "*[_:-]start[_:-]end"

    If the index can be formatted, the formatted columns (coordinate_columns) will be added.
    If the index cannot be formatted, an error will be raised.

    NOTE: adata object is changed inplace.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object containing features to annotate.
    coordinate_columns : NDArray[str] | Sequence[str] | pd.core.indexes.base.Index, default ['chr', 'start', 'end']
        List of length 3 for column names in adata.var containing chr, start, end coordinates.

    Raises
    ------
    ValueError
        If regions are of incorrect format.
    """

    # Test whether the three columns are in the right format
    format_index = True
    if not isinstance(coordinate_columns, list):
        coordinate_columns = ['chr', 'start', 'end']
        logger.info("No column names supplied falling back to default names ['chr', 'start', 'end']")
    elif isinstance(coordinate_columns, list) and len(coordinate_columns) != 3:
        raise ValueError("The coordinate_columns must be a list of length 3 containing the column names for chr, start, end.")
    else:
        logger.info(f"The coordinate columns are: {coordinate_columns}")

    # Check if the columns are already in the right format and if they are in the adata.var
    if validate_regions(adata, coordinate_columns):
        format_index = False
    # Format index if needed
    if format_index:
        logger.info("Formatting the adata.var index to coordinate columns:")
        regex = r'[^_:\-]+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+'  # matches chr_start_end / chr-start-end / chr:start-end and variations

        # Prepare lists to insert
        peak_chr_list = []
        peak_start_list = []
        peak_end_list = []

        for name in adata.var.index:
            if re.match(regex, name):  # test if name can be split by regex

                # split the name into chr, start, end
                split_name = re.split(r'[\_\:\-]', name)
                peak_chr_list.append(split_name[0])
                peak_start_list.append(int(split_name[1]))
                peak_end_list.append(int(split_name[2]))

            else:
                raise ValueError("Index does not match the format *_start_stop or *:start-stop. Please check your index.")

        adata.var.drop(coordinate_columns, axis=1,
                       errors='ignore', inplace=True)

        adata.var.insert(0, coordinate_columns[2], peak_end_list)
        adata.var.insert(0, coordinate_columns[1], peak_start_list)
        adata.var.insert(0, coordinate_columns[0], peak_chr_list)

        # Check whether the newly added columns are in the right format
        if validate_regions(adata, coordinate_columns):
            logger.info('The newly added coordinate columns are in the correct format.')


@beartype
def in_range(value: int | float, limits: Tuple[int | float, int | float],
             include_limits: bool = True) -> bool:
    """
    Check if a value is in a given range.

    Parameters
    ----------
    value : int | float
        Number to check if in range.
    limits : Tuple[int | float, int | float]
        Lower and upper limits. E.g. (0, 10)
    include_limits : bool, default True
        If True includes limits in accepted range.

    Returns
    -------
    bool
        Returns whether the value is between the set limits.

    Examples
    --------
    .. exec_code::

        # --- hide: start ---
        import sctoolbox.utils as utils
        # --- hide: stop ---

        limit = (0.5, 1)
        value = 0.5
        print(utils.checker.in_range(value=value, limits=limit, include_limits=True))

    This will return 'True'; the value is in between the limits including the limits.
    """

    if include_limits:
        return value >= limits[0] and value <= limits[1]
    else:
        return value > limits[0] and value < limits[1]


@beartype
def is_integer_array(arr: ArrayLike) -> bool:
    """
    Check if all values of arr are integers.

    Parameters
    ----------
    arr : ArrayLike
        Array of values to be checked.

    Returns
    -------
    bool
        True if all values are integers, False otherwise.
    """

    # https://stackoverflow.com/a/7236784
    boolean = np.equal(np.mod(arr, 1), 0)

    return bool(np.all(boolean))


@beartype
def check_columns(df: pd.DataFrame,
                  columns: Union[Iterable[str], str],
                  error: bool = True,
                  name: str = "dataframe") -> Optional[bool]:
    """
    Check whether columns are found within a pandas dataframe.

    TODO do we need this?

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe to check.
    columns : Union[Iterable[str], str]
        A list of column names or name to check for within `df`.
    error : bool, default True
        If True raise errror if not all columns are found.
        If False return true or false
    name : str, default dataframe
        Dataframe name displayed in the error message.

    Returns
    -------
    Optional[bool]
        True or False depending on if columns are in dataframe
        None if error is set to True

    Raises
    ------
    KeyError
        If any of the columns are not in 'df' and error is set to True.
    """
    # check if columns is a string
    if isinstance(columns, str):
        columns = [columns]

    # get the columns of the dataframe
    df_columns = df.columns

    # check if the columns are in the dataframe
    not_found = []
    for column in columns:  # for each column to be checked
        if column is not None:
            if column not in df_columns:
                not_found.append(column)

    if len(not_found) > 0:
        if error:
            error_str = f"Columns '{not_found}' are not found in {name}. Available columns are: {list(df_columns)}"
            raise KeyError(error_str)
        else:
            return False
    else:
        if not error:
            return True


@beartype
def check_file_ending(file: str,
                      pattern: str = "gtf") -> None:
    """
    Check if a file has a certain file ending.

    TODO do we need this?

    Parameters
    ----------
    file : str
        Path to the file.
    pattern : str, default 'gtf'
        File ending to be checked for.
        If regex, the regex must match the entire string.

    Raises
    ------
    ValueError
        If file does not have the expected file ending.
    """

    valid = False
    if is_regex:
        if re.match(pattern, file):
            valid = True

    else:
        if file.endswith(pattern):
            valid = True

    if not valid:
        raise ValueError(f"File '{file}' does not have the expected file ending '{pattern}'")


@beartype
def is_regex(regex: str) -> bool:
    """
    Check if a string is a valid regex.

    Parameters
    ----------
    regex : str
        String to be checked.

    Returns
    -------
    bool
        True if string is a valid regex, False otherwise.
    """

    try:
        re.compile(regex)
        return True

    except re.error:
        return False


@beartype
def check_marker_lists(adata: sc.AnnData,
                       marker_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Remove genes in custom marker genes lists which are not present in dataset.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object containing features to annotate.
    marker_dict : dict[str, list[str]]
        A dictionary containing a list of marker genes as values and corresponding cell types as keys.
        The marker genes given in the lists need to match the index of adata.var.

    Returns
    -------
    dict[str, list[str]]
        A dictionary containing a list of marker genes as values and corresponding cell types as keys.
    """

    marker_dict = marker_dict.copy()

    for key, genes in list(marker_dict.items()):
        found_in_var = list(set(adata.var.index) & set(genes))
        not_found_in_var = list(set(genes) - set(adata.var.index))
        if not found_in_var:
            logger.warning(f"No marker in {key} marker list can be found in the data. "
                           + "Please check your marker list. Removing empty marker list form dictionary.")
            marker_dict.pop(key)
        elif not_found_in_var:
            marker_dict[key] = found_in_var
            logger.info(f"Removed {not_found_in_var} from {key} marker gene list")
    return marker_dict


def check_type(obj: Any, obj_name: str, test_type: Any):
    """
    Check type of given object.

    Parameters
    ----------
    obj : Any
        Object for which the type should be checked
    obj_name : str
        Object name that would be shown in the error message.
    test_type : Any
        Type that obj is tested for.

    Raises
    ------
    TypeError
        If object type does not match test type.

    Notes
    -----
    This function is mostly replaced by beartype.
    Only used for types not supported by beartype.
    """
    if not isinstance(obj, test_type):
        raise TypeError(f"Paramter {obj_name} is required to be of type: "
                        + f"{test_type}, but is type: {type(obj)}")
