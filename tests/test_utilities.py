"""Test utility functions."""

import pytest
import os
import numpy as np
import shutil
import pandas as pd
import sctoolbox.utilities as utils
import scanpy as sc


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """Return adata object with 3 groups."""

    adata = sc.AnnData(np.random.randint(0, 100, (100, 100)))
    adata.obs["group"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])

    return adata


@pytest.fixture
def adata2():
    """Load and return an anndata object."""
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")

    return sc.read_h5ad(f)


@pytest.fixture
def unsorted_fragments():
    """Return adata object with 3 groups."""

    fragments = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac_fragments.bed')
    return fragments


@pytest.fixture
def sorted_fragments():
    """Return adata object with 3 groups."""

    fragments = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_sorted_fragments.bed')
    return fragments


@pytest.fixture
def berries():
    """Return list of berries."""
    return ["blueberry", "strawberry", "blackberry"]


@pytest.fixture
def na_dataframe():
    """Return DataFrame with columns of multiple types containing NA."""
    data = {'int': [3, 2, 1, np.nan],
            'float': [1.2, 3.4, 5.6, np.nan],
            'string': ['a', 'b', 'c', np.nan],
            'boolean': [True, False, True, np.nan],
            'category_str': ['cat1', 'cat2', 'cat3', np.nan],
            'category_num': [10, 20, 30, np.nan]}
    df = pd.DataFrame.from_dict(data)

    df['category_str'] = df['category_str'].astype('category')
    df['category_num'] = df['category_num'].astype('category')
    return df


@pytest.fixture
def marker_dict():
    """Return a dict of cell type markers."""
    return {"Celltype A": ['ENSMUSG00000103377', 'ENSMUSG00000104428'],
            "Celltype B": ['ENSMUSG00000102272', 'invalid_gene'],
            "Celltype C": ['invalid_gene_1', 'invalid_gene_2']}


arr_ints = np.random.randint(10, size=(10, 10))
arr_ints2 = arr_ints.astype(float)
arr_floats = np.random.rand(10, 10)


@pytest.mark.parametrize("arr,boolean", [(arr_ints, True), (arr_ints2, True), (arr_floats, False)])
def test_is_integer_array(arr, boolean):
    """Get boolean of whether an array is an integer array."""

    result = utils.is_integer_array(arr)
    assert result == boolean


def test_create_dir():
    """Test if the directory is created."""

    # Ensure that testdir is not already existing
    if os.path.isdir("testdir"):
        shutil.rmtree("testdir")

    # create the dir with the utils function
    utils.create_dir("testdir")
    assert os.path.isdir("testdir")

    shutil.rmtree("testdir")  # clean up after tests


def test_is_notebook():
    """Test if the function is run in a notebook."""

    boolean = utils._is_notebook()
    assert boolean is False  # testing environment is not notebook


@pytest.mark.parametrize("string,expected", [("1.3", True), ("astring", False)])
def test_is_str_numeric(string, expected):
    """Test if a string can be converted to numeric."""

    result = utils.is_str_numeric(string)

    assert result == expected


def test_check_module():
    """Test if check_moduel raises an error for a non-existing module."""

    with pytest.raises(Exception):
        utils.check_module("nonexisting_module")


def test_get_adata_subsets(adata):
    """Test if adata subsets are returned correctly."""

    subsets = utils.get_adata_subsets(adata, "group")

    for group, sub_adata in subsets.items():
        assert sub_adata.obs["group"][0] == group
        assert sub_adata.obs["group"].nunique() == 1


def test_remove_files():
    """Remove files from list."""

    if not os.path.isfile("afile.txt"):
        os.mknod("afile.txt")

    files = ["afile.txt", "notfound.txt"]
    utils.remove_files(files)

    assert os.path.isfile("afile.txt") is False


def test_pseudobulk_table(adata):
    """Test if pseudobulk table is returned correctly."""

    pseudobulk = utils.pseudobulk_table(adata, "group")

    assert pseudobulk.shape[0] == adata.shape[0]
    assert pseudobulk.shape[1] == 3  # number of groups


def test_save_h5ad(adata):
    """Test if h5ad file is saved correctly."""

    path = "test.h5ad"
    utils.save_h5ad(adata, path)

    assert os.path.isfile(path)
    os.remove(path)  # clean up after tests


def test_get_organism():
    """Test function get_organism()."""

    # invalid host
    with pytest.raises(ConnectionError):
        utils.get_organism("ENSE00000000361", host="http://www.ensembl.org/invalid/")

    # invalid id
    with pytest.raises(ValueError):
        utils.get_organism("invalid_id")

    # valid call
    assert utils.get_organism("ENSG00000164690") == "Homo_sapiens"


def test_bed_is_sorted(unsorted_fragments, sorted_fragments):
    """Test if the _bed_is_sorted() function works as expected."""

    assert utils._bed_is_sorted(sorted_fragments)
    assert ~utils._bed_is_sorted(unsorted_fragments)


def test_sort_bed(unsorted_fragments):
    """Test if the sort bedfile functio works."""
    sorted_bedfile = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'sorted_bedfile.bed')
    utils._sort_bed(unsorted_fragments, sorted_bedfile)

    assert utils._bed_is_sorted(sorted_bedfile)

    # Clean up
    os.remove(sorted_bedfile)


def test_check_marker_lists(adata2, marker_dict):
    """Test that check_marker_lists intersects lists correctly."""

    filtered_marker = utils.check_marker_lists(adata2, marker_dict)

    assert filtered_marker == {"Celltype A": ['ENSMUSG00000103377', 'ENSMUSG00000104428'],
                               "Celltype B": ['ENSMUSG00000102272']}


# TODO
# following tests are skipped due to occasional "No internet connection" error.
# This may be related to too many requests in a short period of time.
# Anyway, we should fix this by mocking apybiomart calls.

# def test_gene_id_to_name(adata2):
#     """ Test function gene_id_to_name(). """

#     # invalid species
#     with pytest.raises(ValueError):
#         utils.gene_id_to_name(ids=[], species=None)

#     # invalid id
#     with pytest.raises(ValueError):
#         ids = list(adata2.var.index)
#         ids.append("invalid")
#         utils.gene_id_to_name(ids=ids, species="mmusculus")

#     # valid call
#     id_name_table = utils.gene_id_to_name(ids=list(adata2.var.index), species="mmusculus")

#     assert isinstance(id_name_table, pd.DataFrame)
#     assert len(id_name_table) == len(adata2.var)  # assert all genes kept
#     assert all(c in ["Gene stable ID", "Gene name"] for c in id_name_table.columns)  # assert correct column names


# def test_convert_id(adata2):
#     """ Test convert_id() function. """

#     new_adata = adata2.copy()
#     name_col = "Ensembl name"
#     inv_name_col = "invalid"

#     # invalid parameter combination
#     with pytest.raises(ValueError):
#         utils.convert_id(adata=new_adata, id_col_name=None, index=False, name_col=name_col, species="mmusculus", inplace=False)

#     # invalid column name
#     with pytest.raises(ValueError):
#         utils.convert_id(adata=new_adata, id_col_name=inv_name_col, name_col=name_col, species="mmusculus", inplace=False)

#     # ids as index
#     out_adata = utils.convert_id(adata=new_adata, index=True, name_col=name_col, species="mmusculus", inplace=False)
#     assert name_col in out_adata.var.columns

#     # ids as column
#     new_adata.var.reset_index(inplace=True)
#     out_adata = utils.convert_id(adata=new_adata, id_col_name="index", name_col=name_col, species="mmusculus", inplace=False)
#     assert name_col in out_adata.var.columns

#     # not inplace
#     assert name_col not in new_adata.var.columns

#     # inplace
#     assert utils.convert_id(adata=new_adata, id_col_name="index", name_col=name_col, species="mmusculus", inplace=True) is None
#     assert name_col in new_adata.var


# def test_unify_genes_column(adata2):
#     """ Test the unify_genes_column() function. """

#     mixed_name = "mixed"
#     name_col = "Gene name"
#     new_col = "unified_names"

#     # create mixed column
#     utils.convert_id(adata2, index=True, name_col=name_col)
#     mixed = [id if i % 2 == 0 else row[name_col] for i, (id, row) in enumerate(adata2.var.iterrows())]
#     adata2.var[mixed_name] = mixed

#     # invalid column
#     with pytest.raises(ValueError):
#         utils.unify_genes_column(adata2, column="invalid")

#     # new column created
#     assert new_col not in adata2.var.columns
#     assert new_col in utils.unify_genes_column(adata2, column=mixed_name, unified_column=new_col, inplace=False).var.columns

#     # column overwrite is working
#     utils.unify_genes_column(adata2, column=mixed_name, unified_column=mixed_name, inplace=True)
#     assert any(adata2.var[mixed_name] != mixed)

#     # no Ensembl IDs in output column
#     assert not any(adata2.var[mixed_name].str.startswith("ENS"))
