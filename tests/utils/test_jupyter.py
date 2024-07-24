"""Test jupyter.py functions."""

import pytest
import os
import nbformat
from sctoolbox.utils import jupyter
from sctoolbox import __version__ as sc_version


def test_compare_versions():
    """Test _compare_version function."""

    def change_version(nb_name, version):
        """Change notebook version inplace."""
        nb = nbformat.read(nb_name, as_version=4)
        nb["metadata"]["sc_framework"] = dict()
        nb["metadata"]["sc_framework"]["version"] = version
        nbformat.write(nb, nb_name)

    nb_path = "./test_notebook.ipynb"
    # Build new notebook
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, nb_path)

    # Test without sc_framework in metadata
    with pytest.warns(match="The Notebook seems to be outdated"):
        jupyter._compare_version(nb_path)

    # Test with fitting version
    change_version(nb_path, sc_version)
    jupyter._compare_version(nb_path)

    # Test with newer version
    change_version(nb_path, "1" + sc_version)
    with pytest.warns(match="The notebook has a newer"):
        jupyter._compare_version(nb_path)

    # Test with older version
    change_version(nb_path, "0.0.0")
    with pytest.warns(match="The notebook has an older"):
        jupyter._compare_version(nb_path)

    # Delete test notebook
    os.remove(nb_path)


def test_is_notebook():
    """Test if the function is run in a notebook."""

    boolean = jupyter._is_notebook()
    assert boolean is False  # testing environment is not notebook
