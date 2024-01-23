"""File input/output utilities."""

import os
import tempfile
import warnings
import glob
import deprecation

from beartype import beartype
from beartype.typing import Optional

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
           force: bool = False) -> None:
    """
    Delete given directory.

    First attempts to remove all given `tempfiles` from directory. If `tempfiles` is `None` all files with 'gtf' in the filename (or exstension) are removed.
    After the matching files are deleted the function tries to delete the directory. Possible OSErrors are caught and printed.

    TODO deletion or refactoring

    Parameters
    ----------
    temp_dir : Optional[list[str]], default None
        Path to the temporary directory.
    temp_files : Optional[list[str]], default None
        Paths to files to be deleted before removing the temp directory.
    rm_dir : bool, default False
        If True, the temp directory is removed.
    force : bool, default False
        If True, all files in the temp directory are removed.
    """

    try:
        if temp_files is None:
            if force:
                temp_files = glob.glob(os.path.join(temp_dir, "*"))
            else:
                logger.info('tempfiles is None, not deleting any files')

        if temp_files is not None:
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
