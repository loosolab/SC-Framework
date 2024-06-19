"""Test checker functions."""

import pytest
import os
import sys
import re
import scanpy as sc
import numpy as np
import sctoolbox.utils.checker as ch
import sctoolbox.utilities as utils


@pytest.fixture
def snapatac_adata():
    """Return a adata object from SnapATAC."""

    f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'snapatac.h5ad')

    return sc.read(f)


@pytest.fixture
def adata_atac():
    """Load atac adata."""
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')
    return sc.read_h5ad(adata_f)


@pytest.fixture
def adata_atac_emptyvar(adata_atac):
    """Create adata with empty adata.var."""
    adata = adata_atac.copy()
    adata.var = adata.var.drop(columns=adata.var.columns)
    return adata


@pytest.fixture
def adata_atac_invalid(adata_atac):
    """Create adata with invalid index."""
    adata = adata_atac.copy()
    adata.var.iloc[0, 1] = 500  # start
    adata.var.iloc[0, 2] = 100  # end
    adata.var.reset_index(inplace=True, drop=True)  # remove chromosome-start-stop index
    return adata


@pytest.fixture
def adata_rna():
    """Load rna adata."""
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    return sc.read_h5ad(adata_f)


@pytest.fixture
def adata2():
    """Load and return an anndata object."""
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")

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
        utils.check_module("nonexisting_module")


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

    result = utils.is_str_numeric(string)

    assert result == expected


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


@pytest.mark.parametrize("fixture, expected", [("adata_atac", True),  # expects var tables to be unchanged
                                               ("adata_atac_emptyvar", False),  # expects var tables to be changed
                                               ("adata_rna", ValueError),  # expects a valueerror due to missing columns
                                               ("adata_atac_invalid", ValueError)])  # expects a valueerror due to format of columns
def test_format_adata_var(fixture, expected, request):
    """Test whether adata regions can be formatted (or raise an error if not)."""

    adata_orig = request.getfixturevalue(fixture)  # fix for using fixtures in parametrize
    adata_cp = adata_orig.copy()  # make a copy to avoid changing the fixture
    if isinstance(expected, type):
        with pytest.raises(expected):
            utils.format_adata_var(adata_cp, coordinate_columns=["chr", "start", "stop"])

    else:
        utils.format_adata_var(adata_cp, coordinate_columns=["chr", "start", "stop"], columns_added=["chr", "start", "end"])

        assert np.array_equal(adata_orig.var.values, adata_cp.var.values) == expected  # check if the original adata was changed or not


def test_in_range():
    """Test if int is in given range."""
    assert ch.in_range(value=100, limits=(1, 1000))


arr_ints = np.random.randint(10, size=(10, 10))
arr_ints2 = arr_ints.astype(float)
arr_floats = np.random.rand(10, 10)


@pytest.mark.parametrize("arr,boolean", [(arr_ints, True), (arr_ints2, True), (arr_floats, False)])
def test_is_integer_array(arr, boolean):
    """Get boolean of whether an array is an integer array."""

    result = utils.is_integer_array(arr)
    assert result == boolean


def test_check_marker_lists(adata2, marker_dict):
    """Test that check_marker_lists intersects lists correctly."""

    filtered_marker = utils.check_marker_lists(adata2, marker_dict)

    assert filtered_marker == {"Celltype A": ['ENSMUSG00000103377', 'ENSMUSG00000104428'],
                               "Celltype B": ['ENSMUSG00000102272']}
