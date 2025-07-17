"""General utility functions."""

import os
import re
import sys
from os.path import exists
import subprocess
import shutil
import getpass
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import contextlib


# type hint imports
from beartype.typing import Any, TYPE_CHECKING, Optional, Union, Sequence
from beartype import beartype
import numpy.typing as npt

if TYPE_CHECKING:
    import rpy2.rinterface_lib.sexp


# ------------------ Logging about run ----------------- #

def get_user() -> str:
    """
    Get the name of the current user.

    Returns
    -------
    str
        The name of the current user.
    """

    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"

    return username


def get_datetime() -> str:
    """
    Get a string with the current date and time for logging.

    Returns
    -------
    str
        A string with the current date and time in the format dd/mm/YY H:M:S
    """

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")  # dd/mm/YY H:M:S

    return dt_string


# ------------------ Packages and tools ----------------- #

def get_package_versions() -> dict[str, str]:
    """
    Receive a dictionary of currently installed python packages and versions.

    Returns
    -------
    dict[str, str]
        A dict in the form:
        `{"package1": "1.2.1", "package2":"4.0.1", (...)}`
    """

    # Import freeze
    try:
        from pip._internal.operations import freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze

    # Get list of packages and versions with freeze
    package_list = freeze.freeze()
    package_dict = {}  # dict for collecting versions
    for s in package_list:
        try:
            name, version = re.split("==| @ ", s)
            package_dict[name] = version
        except Exception:
            print(f"Error reading version for package: {s}")

    return package_dict


@beartype
def get_binary_path(tool: str) -> str:
    """
    Get path to a binary commandline tool.

    Looks either in the local dir, on path or in the dir of the executing python binary.

    Parameters
    ----------
    tool : str
        Name of the commandline tool to be found.

    Returns
    -------
    str
        Full path to the tool.

    Raises
    ------
    ValueError
        If executable is not found.
    """

    python_dir = os.path.dirname(sys.executable)
    if os.path.exists(tool):
        tool_path = f"./{tool}"

    else:

        # Check if tool is available on path
        tool_path = shutil.which(tool)
        if tool_path is None:

            # Search for tool within same folder as python (e.g. in an environment)
            python_dir = os.path.dirname(sys.executable)
            tool_path = shutil.which(tool, path=python_dir)

    # Check that tool is executable
    if tool_path is None or shutil.which(tool_path) is None:
        raise ValueError(f"Could not find an executable for {tool} on path.")

    return tool_path


def run_cmd(cmd: str) -> None:
    """
    Run a commandline command.

    Parameters
    ----------
    cmd : str
        Command to be run.

    Raises
    ------
    subprocess.CalledProcessError
        If command has an error.
    """

    try:
        subprocess.check_call(cmd, shell=True)
        print(f"Command '{cmd}' ran successfully!")
    except subprocess.CalledProcessError as e:
        # print(f"Error running command '{cmd}': {e}")
        if e.output is not None:
            print(f"Command standard output: {e.output.decode('utf-8')}")
        if e.stderr is not None:
            print(f"Command standard error: {e.stderr.decode('utf-8')}")
        raise e


#####################################################################
#                           R setup                                 #
#####################################################################

def setup_R(r_home: Optional[str] = None) -> None:
    """
    Add R installation for rpy2 use.

    Parameters
    ----------
    r_home : Optional[str], default None
        Path to the R home directory. If None will construct path based on location of python executable.
        E.g for ".conda/scanpy/bin/python" will look at ".conda/scanpy/lib/R"

    Raises
    ------
    Exception
        If path to R is invalid.
    """

    # Set R installation path
    if not r_home:
        r_home = os.path.join(sys.executable.split('/bin/')[0], 'lib/R')

    if not exists(r_home):
        raise Exception(f'Path to R installation does not exist! Make sure R is installed. {r_home}')

    from rpy2.rinterface_lib import openrlib

    os.environ['R_HOME'] = r_home
    openrlib.R_HOME = r_home


@beartype
def _none2null(none_obj: None) -> "rpy2.rinterface_lib.sexp.NULLType":
    """
    rpy2 converter that translates python 'None' to R 'NULL'.

    Intended to be added as a rpy2 converter object.

    Parameters
    ----------
    none_obj : None
        None object to convert to r"NULL".

    Returns
    -------
    rpy2.rinterface_lib.sexp.NULLType
        R NULL object.
    """

    # See https://stackoverflow.com/questions/65783033/how-to-convert-none-to-r-null
    from rpy2.robjects import r

    return r("NULL")


# ----------------- List functions ---------------- #

@beartype
def split_list(lst: Sequence[Any], n: int) -> list[Sequence[Any]]:
    """
    Split list into n chunks.

    Parameters
    ----------
    lst : Sequence[Any]
        Sequence to be chunked
    n : int
        Number of chunks.

    Returns
    -------
    list[Sequence[Any]]
        List of Sequences (chunks).
    """

    chunks = []
    for i in range(0, n):
        chunks.append(lst[i::n])

    return chunks


@beartype
def split_list_size(lst: list[Any], max_size: int) -> list[list[Any]]:
    """
    Split list into chunks of max_size.

    Parameters
    ----------
    lst : list[Any]
        List to be chunked
    max_size : int
        Max size of chunks.

    Returns
    -------
    list[list[Any]]
        List of lists (chunks).
    """

    chunks = []
    for i in range(0, len(lst), max_size):
        chunks.append(lst[i:i + max_size])

    return chunks


@beartype
def write_list_file(lst: list[Any], path: str) -> None:
    """
    Write a list to a file with one element per line.

    Parameters
    ----------
    lst : list[Any]
        A list of values/strings to write to file
    path : str
        Path to output file.
    """

    lst = [str(s) for s in lst]
    s = "\n".join(lst)

    with open(path, "w") as f:
        f.write(s)


def read_list_file(path: str) -> list[str]:
    """
    Read a list from a file with one element per line.

    Parameters
    ----------
    path : str
        Path to read file from.

    Returns
    -------
    list[str]
        List of strings read from file.
    """

    f = open(path)
    lst = f.read().splitlines()  # get lines without "\n"
    f.close()

    return lst


# ----------------- String functions ---------------- #

@beartype
def clean_flanking_strings(list_of_strings: list[str]) -> list[str]:
    """
    Remove common suffix and prefix from a list of strings.

    E.g. running the function on ['path/a.txt', 'path/b.txt', 'path/c.txt'] would yield ['a', 'b', 'c'].

    Parameters
    ----------
    list_of_strings : list[str]
        List of strings.

    Returns
    -------
    list[str]
        List of strings without common suffix and prefix
    """

    suffix = longest_common_suffix(list_of_strings)
    prefix = os.path.commonprefix(list_of_strings)

    list_of_strings_clean = [remove_prefix(s, prefix) for s in list_of_strings]
    list_of_strings_clean = [remove_suffix(s, suffix) for s in list_of_strings_clean]

    return list_of_strings_clean


@beartype
def longest_common_suffix(list_of_strings: list[str]) -> str:
    """
    Find the longest common suffix of a list of strings.

    Parameters
    ----------
    list_of_strings : list[str]
        List of strings.

    Returns
    -------
    str
        Longest common suffix of the list of strings.
    """

    reversed_strings = [s[::-1] for s in list_of_strings]
    reversed_lcs = os.path.commonprefix(reversed_strings)
    lcs = reversed_lcs[::-1]

    return lcs


def remove_prefix(s: str, prefix: str) -> str:
    """
    Remove prefix from a string.

    Parameters
    ----------
    s : str
        String to be processed.
    prefix : str
        Prefix to be removed.

    Returns
    -------
    str
        String without prefix.
    """

    return s[len(prefix):] if s.startswith(prefix) else s


def remove_suffix(s: str, suffix: str) -> str:
    """
    Remove suffix from a string.

    Parameters
    ----------
    s : str
        String to be processed.
    suffix : str
        Suffix to be removed.

    Returns
    -------
    str
        String without suffix.
    """

    return s[:-len(suffix)] if s.endswith(suffix) else s


@beartype
def sanitize_string(s: str, char_list: list[str], replace: str = "_") -> str:
    """
    Replace every occurrence of given substrings.

    Parameters
    ----------
    s : str
        String to sanitize
    char_list : list[str]
        Strings that should be replaced.
    replace : str, default "_"
        Replacement of substrings.

    Returns
    -------
    str
        Sanitized string.
    """

    for char in char_list:
        s = s.replace(char, replace)

    return s


@beartype
def identify_columns(df: pd.DataFrame,
                     regex: Union[list[str], str]) -> list[str]:
    """
    Get columns from pd.DataFrame that match the given regex.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe to be checked.
    regex : Union(list[str], str)
        List of multiple regex or one regex as string.

    Returns
    -------
    list[str]
        List of column names that match one of the provided regex.
    """

    if isinstance(regex, list):
        regex = "(" + ")|(".join(regex) + ")"

    return df.filter(regex=(regex)).columns.to_list()


@beartype
def scale_values(array: npt.ArrayLike, mini: int | float, maxi: int | float) -> np.ndarray:
    """Small utility to scale values in array to a given range.

    Parameters
    ----------
    array : npt.ArrayLike
        Array to scale.
    mini : int | float
        Minimum value of the scale.
    maxi : int | float
        Maximum value of the scale.

    Returns
    -------
    np.ndarray
        Scaled array values.
    """
    val_range = array.max() - array.min()
    a = (array - array.min()) / val_range
    return a * (maxi - mini) + mini


# ---------------- Logging functions ---------------- #

@contextlib.contextmanager
@beartype
def suppress_logging(level: int = logging.CRITICAL):
    """
    Temporarily disable all logging.

    Note: AI supported implementation

    Parameter
    ---------
    level:
        Supress logging below this level. See https://docs.python.org/3/library/logging.html#logging-levels

    Examples
    --------
    .. code-block:: python

        with supress_logging(level=50):
            function_to_silence()
    """
    # disable logging for ALL loggers
    logging.disable(level=level)
    try:
        yield
    finally:
        # enable logging for ALL loggers
        logging.disable(logging.NOTSET)
