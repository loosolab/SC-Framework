"""Test checker functions."""

import pytest
import sctoolbox.utils.checker as ch
import scanpy as sc
import numpy as np
import os
import re
import sys


@pytest.fixture
def named_var_adata():
    """Return a adata object with a prefix attached to the .var index."""

    f = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_atac_named_var.h5ad')

    return sc.read(f)


@pytest.fixture
def atac_adata():
    """Return a adata object from ATAC-seq."""

    f = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_atac.h5ad')

    return sc.read(f)


@pytest.fixture
def adata_atac_emptyvar(atac_adata):
    """Create adata with empty adata.var."""
    adata = atac_adata.copy()
    adata.var = adata.var.drop(columns=adata.var.columns)
    return adata


@pytest.fixture
def adata_atac_invalid(atac_adata):
    """Create adata with invalid index."""
    adata = atac_adata.copy()
    adata.var.iloc[0, 1] = 500  # start
    adata.var.iloc[0, 2] = 100  # end
    adata.var.reset_index(inplace=True, drop=True)  # remove chromosome-start-stop index
    return adata


@pytest.fixture
def adata_rna():
    """Load rna adata."""
    adata_f = os.path.join(os.path.dirname(__file__), '../data', 'adata.h5ad')
    return sc.read_h5ad(adata_f)


@pytest.fixture
def adata2():
    """Load and return an anndata object."""
    f = os.path.join(os.path.dirname(__file__), '../data', "adata.h5ad")

    return sc.read_h5ad(f)


@pytest.fixture
def marker_dict():
    """Return a dict of cell type markers."""
    return {"Celltype A": ['ENSMUSG00000103377', 'ENSMUSG00000104428'],
            "Celltype B": ['ENSMUSG00000102272', 'invalid_gene'],
            "Celltype C": ['invalid_gene_1', 'invalid_gene_2']}


def test_check_module():
    """Test if check_moduel raises an error for a non-existing module."""

    with pytest.raises(Exception):
        ch.check_module("nonexisting_module")


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


@pytest.mark.parametrize("string,expected", [("1.3", True), ("astring", False)])
def test_is_str_numeric(string, expected):
    """Test if a string can be converted to numeric."""

    result = ch.is_str_numeric(string)

    assert result == expected


def test_in_range():
    """Test if int is in given range."""
    assert ch.in_range(value=100, limits=(1, 1000))


arr_ints = np.random.randint(10, size=(10, 10))
arr_ints2 = arr_ints.astype(float)
arr_floats = np.random.rand(10, 10)


@pytest.mark.parametrize("arr,boolean", [(arr_ints, True), (arr_ints2, True), (arr_floats, False)])
def test_is_integer_array(arr, boolean):
    """Get boolean of whether an array is an integer array."""

    result = ch.is_integer_array(arr)
    assert result == boolean


def test_check_marker_lists(adata2, marker_dict):
    """Test that check_marker_lists intersects lists correctly."""

    filtered_marker = ch.check_marker_lists(adata2, marker_dict)

    assert filtered_marker == {"Celltype A": ['ENSMUSG00000103377', 'ENSMUSG00000104428'],
                               "Celltype B": ['ENSMUSG00000102272']}


def test_in_range():
    """Test if int is in given range."""
    assert ch.in_range(value=100, limits=(1, 1000))


def test_var_index_from_single_col(named_var_adata):
    """Test if var_index_from_single_col works correctly."""
    ch._var_index_from_single_col(named_var_adata,
                                  index_type='prefix',
                                  from_column='coordinate_col')

    # regex pattern to match the var coordinate
    coordinate_pattern = re.compile(r"(chr[0-9XYM]+)+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+")

    # match the first var index
    match = coordinate_pattern.match(named_var_adata.var.index[0])

    # check if the match is not None
    assert match


def test_var_column_to_index_coordinate_cols(atac_adata):
    """Test if var_column_to_index works correctly with coordinate columns."""
    # regex pattern to match the var coordinate
    coordinate_pattern = r"^(chr[0-9XYM]+)[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+$"

    adata = atac_adata.copy()
    adata.var = adata.var.reset_index(drop=True)

    # test if the function formats the var index correctly from the coordinate columns
    ch.var_column_to_index(adata, coordinate_cols=['chr', 'start', 'stop'])

    # check if the first var index is in the correct format
    assert bool(re.fullmatch(coordinate_pattern, adata.var.index[0]))


def test_var_column_to_index(atac_adata):
    """Test if var_column_to_index works correctly."""
    adata = atac_adata.copy()
    # add string to var index
    adata.var.index = 'name_' + adata.var.index
    # test if the function formats the var index correctly
    ch.var_column_to_index(adata)

    # regex pattern to match the var coordinate
    coordinate_pattern = r"^(chr[0-9XYM]+)[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+$"

    # check if the first var index is in the correct format
    assert bool(re.fullmatch(coordinate_pattern, adata.var.index[0]))

    # prepare adata for the next test, add a column to var
    adata.var['index_copy'] = 'name_' + adata.var.index
    adata.var = adata.var.reset_index(drop=True)

    # test if the function formats the var index correctly
    ch.var_column_to_index(adata, coordinate_cols='index_copy')

    # check if the first var index is in the correct format
    assert bool(re.fullmatch(coordinate_pattern, adata.var.index[0]))


@pytest.mark.parametrize("coordinate_columns, expected", [(['chr', 'start', 'stop'], True),  # expects var tables to be unchanged
                                                          (['chr', 'stop', 'start'], False)])  # expects a valueerror due to format of columns
def test_validate_regions(atac_adata, coordinate_columns, expected):
    """Test if validate_regions works correctly."""

    assert ch.validate_regions(atac_adata, coordinate_columns=coordinate_columns) == expected


def test_get_index_type():
    """Test if get_index_type works correctly."""
    start_with_name_index = "some_name-chr1:12343-76899 "
    regex = r"(chr[0-9XYM])+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+"

    assert ch._get_index_type(start_with_name_index, regex) == 'prefix'


def test_check_columns(atac_adata, adata_atac_invalid):
    """Test if check_columns works correctly."""
    assert ch.check_columns(atac_adata.var, ['chr', 'start', 'stop'], error=False)
    assert ch.check_columns(atac_adata.var, 'chr', error=False)

    assert ch.validate_regions(adata_atac_invalid, ['chr', 'start', 'stop']) is False

    with pytest.raises(KeyError):
        ch.check_columns(atac_adata.var, ['chr', 'start', 'stop', 'name'], error=True)

    with pytest.raises(KeyError):
        ch.check_columns(atac_adata.var, 'name', error=True)


@pytest.mark.parametrize("fixture, expected", [("atac_adata", True),  # expects var tables to be unchanged
                                               ("adata_atac_emptyvar", False),
                                               # expects var tables to be changed
                                               ("adata_rna", ValueError),
                                               # expects a valueerror due to missing columns
                                               ("adata_atac_invalid",
                                                ValueError)])  # expects a valueerror due to format of columns
def test_var_index_to_column(fixture, expected, request):
    """Test whether adata regions can be formatted (or raise an error if not)."""

    adata_orig = request.getfixturevalue(fixture)  # fix for using fixtures in parametrize
    adata_cp = adata_orig.copy()  # make a copy to avoid changing the fixture

    if isinstance(expected, type):
        with pytest.raises(expected):
            ch.var_index_to_column(adata_cp, coordinate_columns=["chr", "start", "stop"])

    else:
        ch.var_index_to_column(adata_cp, coordinate_columns=["chr", "start", "stop"])

        assert np.array_equal(adata_orig.var.values,
                              adata_cp.var.values) == expected  # check if the original adata was changed or not


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
