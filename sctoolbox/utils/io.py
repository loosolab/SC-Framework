"""File input/output utilities."""

import os
import tempfile
import warnings
import glob
import deprecation
import yaml
from pathlib import Path

from beartype import beartype
from beartype.typing import Optional, Literal

import sctoolbox
from sctoolbox._settings import settings
logger = settings.logger


@beartype
def create_dir(path: str) -> None:
    """
    Create a directory if it is not existing yet.

    'path' can be either a direct path of the directory, or a path to a file for which the upper directory should be created.

    Parameters
    ----------
    path : str
        Path to the directory to be created.
    """

    base = os.path.basename(path)
    if "." in base:  # path is expected to be a file
        dirname = os.path.dirname(path)  # the last dir of the path
        if dirname != "":  # if dirname is "", file is in current dir
            os.makedirs(dirname, exist_ok=True)

    else:
        if path != "":
            os.makedirs(path, exist_ok=True)


@beartype
def get_temporary_filename(tempdir: str = ".") -> str:
    """
    Get a writeable temporary filename by creating a temporary file and closing it again.

    Parameters
    ----------
    tempdir : str, default "."
        The path where the temp file will be created.

    Returns
    -------
    str
        Name of the temporary file.
    """

    filehandle = tempfile.NamedTemporaryFile(mode="w", dir=tempdir, delete=True)
    filename = filehandle.name
    filehandle.close()  # remove the file again

    return filename


@deprecation.deprecated(deprecated_in="0.4b", removed_in="0.6",
                        current_version=sctoolbox.__version__,
                        details="Use rm_tmp() with rm_dir=False.")
@beartype
def remove_files(file_list: list[str]) -> None:
    """
    Delete all files in a file list. Prints a warning if deletion was not possible.

    Parameters
    ----------
    file_list : list[str]
        List of files to delete.
    """

    for f in file_list:
        try:
            os.remove(f)
        except Exception as e:
            warnings.warn(f"Could not remove file {f}. Exception was: {e}")


@beartype
def rm_tmp(temp_dir: Optional[str] = None,
           temp_files: Optional[list[str]] = None,
           rm_dir: bool = False,
           all: bool = False) -> None:
    """
    Delete given directory.

    Removes all given `temp_files` from the directory. If `temp_files` is `None` and `all` is `True` all files are removed.


    Parameters
    ----------
    temp_dir : Optional[list[str]], default None
        Path to the temporary directory.
    temp_files : Optional[list[str]], default None
        Paths to files to be deleted before removing the temp directory.
    rm_dir : bool, default False
        If True, the temp directory is removed.
    all : bool, default False
        If True, all files in the temp directory are removed.
    """

    try:
        if temp_files is None and not all:
            logger.info('tempfiles is None, not deleting any files')
        else:
            temp_files = glob.glob(os.path.join(temp_dir, "*")) if all else temp_files
            logger.info('removing tempfiles')
            for f in temp_files:
                try:
                    os.remove(f)
                except Exception as e:
                    warnings.warn(f"Could not remove file {f}. Exception was: {e}")

        if rm_dir:
            logger.info('removing temp_dir')
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                warnings.warn(f"Could not remove directory {temp_dir}. Exception was: {e}")

    except OSError as error:
        print(error)


@beartype
def update_yaml(d: dict, yml: str, path_prefix: Optional[Literal["report", "table"]] = None):
    """
    Add/ update entries in the given yaml file.

    Parameters
    ----------
    d : dict
        The dict that will be added to the yaml file.
    yml : str
        Path to the yaml file. Will create a new file if necessary.
    path_prefix : str, default None
        Either add `sctoolbox.settings.report_dir`, `sctoolbox.settings.table_dir` or no path as prefix.
    """
    if path_prefix == "report":
        file = Path(settings.report_dir) / yml
    elif path_prefix == "table":
        file = Path(settings.table_dir) / yml

    update_dict = {}
    # read if yaml already exists
    if file.is_file():
        with open(file, "r") as f:
            update_dict = yaml.safe_load(f)
        update_dict = {} if update_dict is None else update_dict

    # write updated dict
    with open(file, "w") as f:
        update_dict.update(d)
        yaml.safe_dump(update_dict, stream=f, sort_keys=False)
