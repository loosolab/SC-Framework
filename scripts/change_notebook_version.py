"""
Script to add/update notebook versions.
"""
import nbformat
import warnings
import argparse
from pathlib import Path
from packaging import version


def update_version(notebook, ver):
    nb = nbformat.read(notebook, as_version=4)
    if "sc_framework" not in nb["metadata"]:
        nb["metadata"]["sc_framework"] = dict()
        nb["metadata"]["sc_framework"]["version"] = ver
    else:
        old_ver = nb["metadata"]["sc_framework"]["version"]
        if version.parse(old_ver) > version.parse(ver):
            warnings.warn(f"New version ({ver}) is older than notebook version ({old_ver}). Skipped {notebook}")
            return
        if version.parse(old_ver) == version.parse(ver):
            warnings.warn(f"New version is identical to notebook version. ({ver}). Skipped {notebook}")
            return
        nb["metadata"]["sc_framework"]["version"] = ver
    nbformat.write(nb, notebook)


def update_notebooks(repo_path, ver):
    path = Path(repo_path)
    for p in path.glob("**/*.ipynb"):
        print(p)
        update_version(p, ver)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="change_notebook_version",
        description="Loops recursivly through given directory and updates the sc_framework version for every notebook."
    )
    parser.add_argument("path", type=str, help="Directory containing notebooks.")
    parser.add_argument("version", type=str, help="New version.")
    args = parser.parse_args()

    update_notebooks(args.path, args.version)