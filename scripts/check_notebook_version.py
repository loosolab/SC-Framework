"""
Script to check if every notebook in directory has the latest version.
"""

import argparse
from pathlib import Path
import nbformat
from packaging import version


def check_notebook(nb_path, ver):
    """Check notebook"""
    
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


def check_versions(path_list, ver):
    """Loop over all directories and check notebook versions."""

    for p in path_list:
        path = Path(p)
        for nb_path in path.glob("**/*.ipynb"):
            check_notebook(nb_path, ver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="check_notebook_version",
        description="Loops recursivly through given directories and checks \
            if the sc_framework version is up to date fore very notebook.")
    parser.add_argument('-n', '--notebook_paths', nargs='+', default=[])
    parser.add_argument('-v', '--version', type=str, help="New version.")
    args = parser.parse_args()
    check_versions(args.notebook_paths, args.version)