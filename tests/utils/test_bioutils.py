"""Test bioutils.py functions."""

import pytest
import scanpy as sc
import numpy as np
import os
import sctoolbox.utils as utils
import re
import shutil


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata_mock():
    """Return adata object with 3 groups."""

    adata = sc.AnnData(np.random.randint(0, 100, (100, 100)))
    adata.obs["group"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])

    return adata


@pytest.fixture
def adata():
    """Return a adata object from SnapATAC."""

    f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')

    return sc.read(f)


@pytest.fixture
def bedfile():
    """Return a bedfile."""

    f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_sorted_fragments.bed')

    return f


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


def test_pseudobulk_table(adata_mock):
    """Test if pseudobulk table is returned correctly."""

    pseudobulk = utils.bioutils.pseudobulk_table(adata_mock, "group")

    assert pseudobulk.shape[0] == adata_mock.shape[0]
    assert pseudobulk.shape[1] == 3  # number of groups


def test_barcode_index(adata):
    """Test barcode index."""

    regex = re.compile(r'([ATCG]{8,16})')
    # remove barcode from index and add it to a column
    adata.obs['barcode'] = adata.obs.index
    adata.obs = adata.obs.reset_index(drop=True)
    # get first index element
    first_index = str(adata.obs.index[0])
    # check if the first index element is a barcode
    match = regex.match(first_index)
    # assert match is None
    assert match is None

    utils.bioutils.barcode_index(adata)

    # get first index element
    first_index = adata.obs.index[0]
    # check if the first index element is a barcode
    match = regex.match(first_index)
    # assert match is None
    assert match is not None

    # execute barcode_index again to check if it will raise an error
    utils.bioutils.barcode_index(adata)


def test_get_organism():
    """Test function get_organism()."""

    # invalid host
    with pytest.raises(ConnectionError):
        utils.bioutils.get_organism("ENSE00000000361", host="http://www.ensembl.org/invalid/")

    # invalid id
    with pytest.raises(ValueError):
        utils.bioutils.get_organism("invalid_id")

    # valid call
    assert utils.bioutils.get_organism("ENSG00000164690") == "Homo_sapiens"


def test_overlap_two_bedfiles(bedfile):
    """Test overlap_two_bedfiles."""

    # Copy a file from source to destination
    test_data_dir = os.path.split(bedfile)[0]
    bedfile_copy = os.path.join(test_data_dir, 'copied.bed')
    shutil.copy(bedfile, bedfile_copy)

    # overlap bedfiles
    overlap_file = 'overlap.bed'
    utils.bioutils._overlap_two_bedfiles(bedfile, bedfile_copy, overlap_file)

    # Test for successful overlap
    assert os.path.exists(overlap_file) and os.path.getsize(overlap_file) > 0

    # clean up
    try:
        os.remove(bedfile_copy)
        os.remove(overlap_file)
    except FileNotFoundError:
        raise FileNotFoundError("The file does not exist")


def test_bed_is_sorted(unsorted_fragments, sorted_fragments):
    """Test if the _bed_is_sorted() function works as expected."""

    assert utils.bioutils._bed_is_sorted(sorted_fragments)
    assert ~utils.bioutils._bed_is_sorted(unsorted_fragments)


def test_sort_bed(unsorted_fragments):
    """Test if the sort bedfile functio works."""
    sorted_bedfile = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'sorted_bedfile.bed')
    utils.bioutils._sort_bed(unsorted_fragments, sorted_bedfile)

    assert utils.bioutils._bed_is_sorted(sorted_bedfile)

    # Clean up
    os.remove(sorted_bedfile)


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
