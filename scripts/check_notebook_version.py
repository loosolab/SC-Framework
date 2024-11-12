"""
Script to check if every notebook in directory has the latest version.
"""

import argparse
from pathlib import Path
import nbformat
from packaging import version
import sys 
import os


def check_notebook(nb_path: str, ver: str) -> None:
    """
    Check notebook.

    Parameters
    ----------
    nb_path : str
        Path to jupyter notebook.
    ver : str
        Current version of sc_framework.
    """

    print(f"Checking {nb_path}")

    err_msg = "The notebooks are required to contain the notebook " + \
              "version in the metadata field: nb['metadata']['sc_framework']['version']"

    # Read notebook
    nb = nbformat.read(nb_path, as_version=4)

    # Check for version number
    if "sc_framework" not in nb["metadata"]:
        raise KeyError(f"'sc_framework' key not found in metadata of notebook {nb_path}. {err_msg}")

    if "version" not in nb["metadata"]["sc_framework"]:
        raise KeyError(f"'version' key not found in metadata.sc_framework of notebook {nb_path}. {err_msg}")

    # Compare versions
    notebook_ver = version.parse(nb["metadata"]["sc_framework"]["version"])
    sc_ver = version.parse(ver)

    if notebook_ver > sc_ver:
        raise ValueError(f"Notebook version is higher than sc_framework version. ({notebook_ver} > {sc_ver})")
    if notebook_ver < sc_ver:
        raise ValueError(f"Notebook version is lower than sc_framework version. ({notebook_ver} < {sc_ver})")
    print("Notebook and sc_framework versions are matching!")


def check_versions(path_list: str, ver: str) -> None:
    """
    Loop over all directories and check notebook versions.

    Parameters
    ----------
    path_list : str
        List of paths for directories containing jupyter notebooks.
    version : str
        Current version of sc_framework.
    """

    for p in path_list:
        path = Path(p)
        for nb_path in path.glob("**/*.ipynb"):
            check_notebook(nb_path, ver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="check_notebook_version",
        description="Loops recursivly through given directories and checks \
            if the sc_framework version is up to date fore very notebook.")
    parser.add_argument('-n', '--notebook_paths', nargs='+', default=[], help="Directories containing .ipynb files")
    parser.add_argument('-v', '--version', help="Path to sctoolbox directory")
    args = parser.parse_args()

    # Only import _version.py file to get current sc_framework version
    sys.path.append(os.path.abspath(args.version))
    import _version as v
    ver = v.__version__

    check_versions(args.notebook_paths, ver)
