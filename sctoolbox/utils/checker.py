"""Module for type checking functions."""

import re
import importlib
import matplotlib
import numpy as np
import gzip
import shutil
import scanpy as sc
import pandas as pd

from typing import Optional, Tuple, Any, Sequence
from beartype import beartype
import numpy.typing as npt

import sctoolbox.utils as utils
from sctoolbox._settings import settings
logger = settings.logger


@beartype
def check_module(module: str) -> None:
    """
    Check if <module> can be imported without error.

    Parameters
    ----------
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
    except Exception:
        raise  # unexpected error loading module

    # Write out error if module was not found
    if error == 1:
        s = f"ERROR: Could not find the '{module}' module on path, but the module is needed for this functionality. Please install this package to proceed."
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
def format_index(adata: sc.AnnData,
                 from_column: Optional[str] = None) -> None:
    """
    Format adata.var index.

    This formats the index of adata.var according to the pattern ["chr", "start", "stop"].
    The adata is changed inplace.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object to reformat.
    from_column : Optional[str], default None
        Column name in adata.var to be set as index.
    """

    if from_column is None:
        entry = adata.var.index[0]
        index_type = get_index_type(entry)

        if index_type == 'snapatac':
            adata.var['name'] = adata.var['name'].str.replace("b'", "")
            adata.var['name'] = adata.var['name'].str.replace("'", "")

            # split the peak column into chromosome start and end
            adata.var[['peak_chr', 'start_end']] = adata.var['name'].str.split(':', expand=True)
            adata.var[['peak_start', 'peak_end']] = adata.var['start_end'].str.split('-', expand=True)
            # set types
            adata.var['peak_chr'] = adata.var['peak_chr'].astype(str)
            adata.var['peak_start'] = adata.var['peak_start'].astype(int)
            adata.var['peak_end'] = adata.var['peak_end'].astype(int)
            # remove start_end column
            adata.var.drop('start_end', axis=1, inplace=True)

            adata.var = adata.var.set_index('name')

        elif index_type == "start_name":
            coordinate_pattern = r"(chr[0-9XYM]+)+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+"
            new_index = []
            for line in adata.var.index:
                new_index.append(re.search(coordinate_pattern, line).group(0))
            adata.var['new_index'] = new_index
            adata.var.set_index('new_index', inplace=True)

    else:
        entry = list(adata.var[from_column])[0]
        index_type = get_index_type(entry)

        if index_type == 'snapatac':
            adata.var['name'] = adata.var['name'].str.replace("b'", "")
            adata.var['name'] = adata.var['name'].str.replace("'", "")

            # split the peak column into chromosome start and end
            adata.var[['peak_chr', 'start_end']] = adata.var['name'].str.split(':', expand=True)
            adata.var[['peak_start', 'peak_end']] = adata.var['start_end'].str.split('-', expand=True)
            # set types
            adata.var['peak_chr'] = adata.var['peak_chr'].astype(str)
            adata.var['peak_start'] = adata.var['peak_start'].astype(int)
            adata.var['peak_end'] = adata.var['peak_end'].astype(int)
            # remove start_end column
            adata.var.drop('start_end', axis=1, inplace=True)

            adata.var = adata.var.set_index('name')

        elif index_type == "start_name":
            coordinate_pattern = r"(chr[0-9XYM]+)+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+"
            new_index = []
            for line in adata.var[from_column]:
                new_index.append(re.search(coordinate_pattern, line).group(0))
            adata.var['new_index'] = new_index
            adata.var.set_index('new_index', inplace=True)


@beartype
def get_index_type(entry: str) -> Optional[str]:
    """
    Check the format of the index by regex.

    Parameters
    ----------
    entry : str
        String to identify the format on.

    Returns
    -------
    Optional[str]
        The index format. Either 'snapatac', 'start_name' or None for unknown format.
    """

    regex_snapatac = r"^b'(chr[0-9]+)+'[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+"  # matches: b'chr1':12324-56757
    regex_start_name = r"^.+(chr[0-9]+)+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+"  # matches: some_name-chr1:12343-76899

    if re.match(regex_snapatac, entry):
        return 'snapatac'
    if re.match(regex_start_name, entry):
        return 'start_name'


@beartype
def validate_regions(adata: sc.AnnData,
                     coordinate_columns: Sequence[str]) -> None:
    """
    Check if the regions in adata.var are valid.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing the regions to be checked.
    coordinate_columns : Sequence[str]
        List of length 3 for column names in adata.var containing chr, start, end coordinates.

    Raises
    ------
    ValueError
        If invalid regions are detected.
    """

    # Test whether the first three columns are in the right format
    chr, start, end = coordinate_columns

    # Test if coordinate columns are in adata.var
    utils.check_columns(adata.var, coordinate_columns, name="adata.var")

    # Test whether the first three columns are in the right format
    for _, line in adata.var.to_dict(orient="index").items():
        valid = False

        if isinstance(line[chr], str) and isinstance(line[start], int) and isinstance(line[end], int):
            if line[start] <= line[end]:  # start must be smaller than end
                valid = True  # if all tests passed, the line is valid

        if valid is False:
            raise ValueError("The region {0}:{1}-{2} is not a valid genome region. Please check the format of columns: {3}".format(line[chr], line[start], line[end], coordinate_columns))


@beartype
def format_adata_var(adata: sc.AnnData,
                     coordinate_columns: Optional[Sequence[str]] = None,
                     columns_added: Sequence[str] = ["chr", "start", "end"]) -> None:
    """
    Format the index of adata.var and adds peak_chr, peak_start, peak_end columns to adata.var if needed.

    If coordinate_columns are given, the function will check if these columns already contain the information needed. If the coordinate_columns are in the correct format, nothing will be done.
    If the coordinate_columns are invalid (or coordinate_columns is not given) the index is checked for the following format:
    "*[_:-]start[_:-]stop"

    If the index can be formatted, the formatted columns (columns_added) will be added.
    If the index cannot be formatted, an error will be raised.

    NOTE: adata object is changed inplace.

    Parameters
    ----------
    adata : sc.AnnData
        The anndata object containing features to annotate.
    coordinate_columns : Optional[Sequence[str]], default None
        List of length 3 for column names in adata.var containing chr, start, end coordinates to check.
        If None, the index will be formatted.
    columns_added : Sequence[str], default ['chr', 'start', 'end']
        List of length 3 for column names in adata.var containing chr, start, end coordinates to add.

    Raises
    ------
    KeyError
        If `coordinate_columns` are not available.
    ValueError
        If regions are of incorrect format.
    """

    # Test whether the first three columns are in the right format
    format_index = True
    if coordinate_columns is not None:
        try:
            validate_regions(adata, coordinate_columns)
            format_index = False
        except KeyError:
            print("The coordinate columns are not found in adata.var. Trying to format the index.")
        except ValueError:
            print("The regions in adata.var are not in the correct format. Trying to format the index.")

    # Format index if needed
    if format_index:
        print("formatting adata.var index to coordinate columns:")
        regex = r'[^_:\-]+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+'  # matches chr_start_end / chr-start-end / chr:start-end and variations

        # Prepare lists to insert
        peak_chr_list = []
        peak_start_list = []
        peak_end_list = []

        names = adata.var.index
        for name in names:
            if re.match(regex, name):  # test if name can be split by regex

                # split the name into chr, start, end
                split_name = re.split(r'[\_\:\-]', name)
                peak_chr_list.append(split_name[0])
                peak_start_list.append(int(split_name[1]))
                peak_end_list.append(int(split_name[2]))

            else:
                raise ValueError("Index does not match the format *_start_stop or *:start-stop. Please check your index.")

        adata.var.drop(columns_added, axis=1,
                       errors='ignore', inplace=True)

        adata.var.insert(0, columns_added[2], peak_end_list)
        adata.var.insert(0, columns_added[1], peak_start_list)
        adata.var.insert(0, columns_added[0], peak_chr_list)

        # Check whether the newly added columns are in the right format
        validate_regions(adata, columns_added)


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
    """

    if include_limits:
        return value >= limits[0] and value <= limits[1]
    else:
        return value > limits[0] and value < limits[1]


@beartype
def is_integer_array(arr: npt.ArrayLike) -> bool:
    """
    Check if all values of arr are integers.

    Parameters
    ----------
    arr : npt.ArrayLike
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
                  columns: Sequence[str],
                  error: bool = True,
                  name: str = "dataframe") -> Optional[bool]:
    """
    Check whether columns are found within a pandas dataframe.

    TODO do we need this?

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe to check.
    columns : Sequence[str]
        A list of column names to check for within `df`.
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

    df_columns = df.columns

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
    """Check type of given object."""
    if not isinstance(obj, test_type):
        raise ValueError(f"Paramter {obj_name} is required to be of type: "
                         + f"{test_type}, but is type: {type(obj)}")
