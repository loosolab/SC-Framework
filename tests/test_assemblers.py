"""Test functions to assemble adata objects."""

import os
import re
import anndata
import pytest
import sctoolbox.utils.assemblers as assemblers
import scanpy as sc


@pytest.fixture
def snapatac_adata():
    """Return a adata object from SnapATAC."""

    f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'snapatac.h5ad')

    return sc.read(f)


def test_prepare_atac_anndata(snapatac_adata):
    """Test prepare_atac_anndata success."""
    f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'snapatac.h5ad')
    adata = assemblers.prepare_atac_anndata(snapatac_adata, index_from='name', h5ad_path=f)

    regex = re.compile(r'([ATCG]{8,16})')
    # get first index element
    first_index = adata.obs.index[0]
    # check if the first index element is a barcode
    match = regex.match(first_index)
    # assert match is None
    assert match is not None

    # regex pattern to match the var coordinate
    coordinate_pattern = re.compile(r"(chr[0-9XYM]+)+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+")
    # match the first var index
    match = coordinate_pattern.match(adata.var.index[0])
    # check if the match is not None
    assert match is not None

    # check if coordinate cols are added
    assert 'peak_chr' in adata.var.columns
    assert 'peak_start' in adata.var.columns
    assert 'peak_end' in adata.var.columns

    # check if the h5ad file path is saved
    assert adata.obs['file'][0] == f


def test_from_single_starsolo():
    """Test from_single_starsolo success."""

    SOLO_DIR = os.path.join(os.path.dirname(__file__), 'data', 'solo')
    adata = assemblers.from_single_starsolo(SOLO_DIR, dtype="filtered", header=None)

    assert isinstance(adata, anndata.AnnData)


def test_from_single_mtx():
    """Test from_single_mtx success."""

    MTX_FILENAME = os.path.join(os.path.dirname(__file__), 'data', 'solo', 'Gene', 'filtered', 'matrix.mtx')
    BARCODES_FILENAME = os.path.join(os.path.dirname(__file__), 'data', 'solo', 'Gene', 'filtered', 'barcodes.tsv')
    GENES_FILENAME = os.path.join(os.path.dirname(__file__), 'data', 'solo', 'Gene', 'filtered', 'genes.tsv')

    adata = assemblers.from_single_mtx(MTX_FILENAME, BARCODES_FILENAME, GENES_FILENAME, header=None)

    assert isinstance(adata, anndata.AnnData)
