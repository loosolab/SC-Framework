"""File input/output utilities."""

import os
import tempfile
import warnings
import glob


def create_dir(path):
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


def get_temporary_filename(tempdir="."):
    """
    Get a writeable temporary filename by creating a temporary file and closing it again.

    Parameters
    ----------
    tempdir : str, default .

    Returns
    -------
    str
        Name of the temporary file.
    """

    filehandle = tempfile.NamedTemporaryFile(mode="w", dir=tempdir, delete=True)
    filename = filehandle.name
    filehandle.close()  # remove the file again

    return filename


def remove_files(file_list):
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


def rm_tmp(temp_dir, tempfiles=None):
    """
    Delete given directory.

    First attempts to remove all given `tempfiles` from directory. If `tempfiles` is `None` all files with 'gtf' in the filename (or exstension) are removed.
    After the matching files are deleted the function tries to delete the directory. Possible OSErrors are caught and printed.

    TODO deletion or refactoring

    Parameters
    ----------
    temp_dir : str
        Path to the temporary directory.
    tempfiles : list[str], default None
        Paths to files to be deleted before removing the temp directory.
    """

    try:
        if tempfiles is None:
            for f in glob.glob(temp_dir + "/*gtf*"):
                os.remove(f)
        else:
            for f in tempfiles:
                os.remove(f)

        os.rmdir(temp_dir)

    except OSError as error:
        print(error)
