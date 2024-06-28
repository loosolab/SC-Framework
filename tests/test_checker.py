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


@pytest.fixture
def atac_adata():
    """Return a adata object from ATAC-seq."""

    f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')

    return sc.read(f)


def test_in_range():
    """Test if int is in given range."""
    assert ch.in_range(value=100, limits=(1, 1000))


def test_var_index_from_single_col(snapatac_adata):
    """Test if var_index_from_single_col works correctly."""
    ch.var_index_from_single_col(snapatac_adata,
                                 index_type='binary',
                                 from_column='name')

    # regex pattern to match the var coordinate
    coordinate_pattern = re.compile(r"(chr[0-9XYM]+)+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+")

    # match the first var index
    match = coordinate_pattern.match(snapatac_adata.var.index[0])

    # check if the match is not None
    assert match is not None


def test_var_index_from(snapatac_adata, atac_adata):
    """Test if var_index_from works correctly."""
    adata = atac_adata.copy()
    # add string to var index
    adata.var.index = 'name_' + adata.var.index

    # test if the function formats the var index correctly
    ch.var_index_from(adata)

    # regex pattern to match the var coordinate
    coordinate_pattern = r"^(chr[0-9XYM]+)[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+$"

    # check if the first var index is in the correct format
    assert bool(re.fullmatch(coordinate_pattern, adata.var.index[0])) is True

    # prepare adata for the next test, add a column to var
    adata.var['index_copy'] = 'name_' + adata.var.index
    adata.var = adata.var.reset_index(drop=True)

    # test if the function formats the var index correctly
    ch.var_index_from(adata, coordinate_cols='index_copy')

    # check if the first var index is in the correct format
    assert bool(re.fullmatch(coordinate_pattern, adata.var.index[0])) is True

    # prepare adata for the next test
    adata.var = adata.var.reset_index(drop=True)
    adata.var.pop('index_copy')

    # test if the function formats the var index correctly from the coordinate columns
    ch.var_index_from(adata, coordinate_cols=['chr', 'start', 'stop'])

    # check if the first var index is in the correct format
    assert bool(re.fullmatch(coordinate_pattern, adata.var.index[0])) is True

    # test if the function formats binary index correctly
    ch.var_index_from(snapatac_adata, coordinate_cols='name')

    # check if the first var index is in the correct format
    assert bool(re.fullmatch(coordinate_pattern, snapatac_adata.var.index[0])) is True


def test_validate_regions(atac_adata):
    """Test if validate_regions works correctly."""

    assert ch.validate_regions(atac_adata, coordinate_columns=['chr', 'start', 'stop'])
    assert not ch.validate_regions(atac_adata, coordinate_columns=['chr', 'stop', 'start'], verbose=False)


def test_get_index_type(snapatac_adata):
    """Test if get_index_type works correctly."""
    binary_index = "b'chr1':12324-56757"
    start_with_name_index = "some_name-chr1:12343-76899 "

    assert ch.get_index_type(binary_index) == 'binary'
    assert ch.get_index_type(start_with_name_index) == 'named'


def test_check_columns(atac_adata):
    """Test if check_columns works correctly."""
    assert ch.check_columns(atac_adata.var, ['chr', 'start', 'stop'], error=False)
    assert ch.check_columns(atac_adata.var, 'chr', error=False)

    with pytest.raises(KeyError):
        ch.check_columns(atac_adata.var, ['chr', 'start', 'stop', 'name'], error=True)

    with pytest.raises(KeyError):
        ch.check_columns(atac_adata.var, 'name', error=True)


def test_add_path():
    """Test if _add_path adds the path correctly."""
    python_exec_dir = os.path.dirname(sys.executable)  # get path to python executable

    assert python_exec_dir == ch._add_path()
    assert python_exec_dir in os.environ['PATH']

    ori_PATH = os.environ['PATH']  # save the original PATH
    os.environ['PATH'] = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'  # mock notebook like path
    assert python_exec_dir == ch._add_path()
    os.environ['PATH'] = ori_PATH  # restore the original PATH
    assert python_exec_dir in os.environ['PATH']
