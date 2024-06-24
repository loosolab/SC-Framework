"""
Script to add/update notebook versions.
"""
import nbformat
import warnings
import argparse
from pathlib import Path
from packaging import version


def update_version(notebook: str, ver: str, force: bool):
    """
    Update sc_framework version of notebook.
    
    Parameters
    ----------
    notebook : str
        Path of notebook.
    ver : str
        New version of notebook.
    force : bool
        Ignore warnings and overwrite notebook version.
    """
    nb = nbformat.read(notebook, as_version=4)
    if "sc_framework" not in nb["metadata"]:
        nb["metadata"]["sc_framework"] = dict()
        nb["metadata"]["sc_framework"]["version"] = ver
    else:
        old_ver = nb["metadata"]["sc_framework"]["version"]
        if not force:
            if version.parse(old_ver) > version.parse(ver):
                warnings.warn(f"New version ({ver}) is older than notebook version ({old_ver}). Skipped {notebook} (set 'force' to overwrite)")
                return
            if version.parse(old_ver) == version.parse(ver):
                warnings.warn(f"New version is identical to notebook version. ({ver}). Skipped {notebook} (set 'force' to overwrite)")
                return
        nb["metadata"]["sc_framework"]["version"] = ver
    nbformat.write(nb, notebook)


def update_notebooks(repo_path: str, ver: str, force: bool):
    """
    Loops recursivly through given directory and updates the sc_framework version for every notebook.
    
    Parameters
    ----------
    repo_path : str
        Path of repository/directory containing notebooks.
    ver : str
        New version of notebook.
    force : bool
        Ignore warnings and overwrite notebook version.
    """
    path = Path(repo_path)
    for p in path.glob("**/*.ipynb"):
        print(p)
        update_version(p, ver, force)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="change_notebook_version",
        description="Loops recursivly through given directory and updates the sc_framework version for every notebook."
    )
    parser.add_argument("path", type=str, help="Directory containing notebooks.")
    parser.add_argument("version", type=str, help="New version.")
    parser.add_argument("-f", "--force", action='store_true', help="If set force new version")
    args = parser.parse_args()

    update_notebooks(args.path, args.version, args.force)
