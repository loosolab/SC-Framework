"""Test bioutils functions."""

import pytest
import scanpy as sc
import sctoolbox.utils.bioutils as bioutils
import re
import os
import shutil


@pytest.fixture
def bedfile():
    """Return a bedfile."""

    f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_sorted_fragments.bed')

    return f


@pytest.fixture
def snapatac_adata():
    """Return a adata object from SnapATAC."""

    f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'snapatac.h5ad')

    return sc.read(f)


def test_barcode_index(snapatac_adata):
    """Test barcode index."""

    regex = re.compile(r'([ATCG]{8,16})')
    # get first index element
    first_index = snapatac_adata.obs.index[0]
    # check if the first index element is a barcode
    match = regex.match(first_index)
    # assert match is None
    assert match is None

    bioutils.barcode_index(snapatac_adata)

    # get first index element
    first_index = snapatac_adata.obs.index[0]
    # check if the first index element is a barcode
    match = regex.match(first_index)
    # assert match is None
    assert match is not None

    # execute barcode_index again to check if it will raise an error
    bioutils.barcode_index(snapatac_adata)


def test_overlap_two_bedfiles(bedfile):
    """Test overlap_two_bedfiles."""

    # Copy a file from source to destination
    test_data_dir = os.path.split(bedfile)[0]
    bedfile_copy = os.path.join(test_data_dir, 'copied.bed')
    shutil.copy(bedfile, bedfile_copy)

    # overlap bedfiles
    overlap_file = 'overlap.bed'
    bioutils._overlap_two_bedfiles(bedfile, bedfile_copy, overlap_file)

    # Test for successful overlap
    assert os.path.exists(overlap_file) and os.path.getsize(overlap_file) > 0

    # clean up
    try:
        os.remove(bedfile_copy)
        os.remove(overlap_file)
    except FileNotFoundError:
        print("The file does not exist")
    except Exception as e:
        print(f"An error occurred: {e}")
