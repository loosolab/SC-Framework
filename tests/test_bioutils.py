"""Test bioutils functions."""

import pytest
import scanpy as sc
import sctoolbox.utils.bioutils as bioutils
import re
import os


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
