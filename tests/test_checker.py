"""Test checker functions."""

import pytest
import sctoolbox.utils.checker as ch
import scanpy as sc
import os
import re
import sys


@pytest.fixture
def snapatac_adata():
    """Return a adata object from SnapATAC."""

    f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'snapatac.h5ad')

    return sc.read(f)


def test_in_range():
    """Test if int is in given range."""
    assert ch.in_range(value=100, limits=(1, 1000))


def test_var_index_from(snapatac_adata):
    """Test if var_index_from works correctly."""
    # execute the function
    ch.var_index_from(snapatac_adata, 'name')

    # regex pattern to match the var coordinate
    coordinate_pattern = re.compile(r"(chr[0-9XYM]+)+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+")
    # match the first var index
    match = coordinate_pattern.match(snapatac_adata.var.index[0])
    # check if the match is not None
    assert match is not None

    # check if coordinate cols are added
    assert 'peak_chr' in snapatac_adata.var.columns
    assert 'peak_start' in snapatac_adata.var.columns
    assert 'peak_end' in snapatac_adata.var.columns


def test_get_index_type(snapatac_adata):
    """Test if get_index_type works correctly."""
    snapatac_index = "b'chr1':12324-56757"
    start_with_name_index = "some_name-chr1:12343-76899 "

    assert ch.get_index_type(snapatac_index) == 'snapatac'
    assert ch.get_index_type(start_with_name_index) == 'start_name'

def test_add_path():
    """Test _add_path function."""
    # test if the path is added correctly
    python_exec_dir = os.path.dirname(sys.executable)  # get path to python executable

    assert python_exec_dir == ch._add_path()
    assert python_exec_dir in os.environ['PATH']