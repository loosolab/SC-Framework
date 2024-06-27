"""Test add_insertsize function."""

import pytest
import sctoolbox.atac
import os
import scanpy as sc


@pytest.fixture
def adata():
    """Fixture for an AnnData object."""
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_atac.h5ad'))
    return adata


def test_add_insertsize_fragments(adata):
    """Test if add_insertsize adds information from a fragmentsfile."""

    adata = adata.copy()
    fragments = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_atac_fragments.bed')
    sctoolbox.atac.add_insertsize(adata, fragments=fragments)

    assert "insertsize_distribution" in adata.uns
    assert "mean_insertsize" in adata.obs.columns


def test_add_insertsize_bam(adata):
    """Test if add_insertsize adds information from a bamfile."""

    adata = adata.copy()
    bam = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_atac.bam')
    sctoolbox.atac.add_insertsize(adata, bam=bam)

    assert "insertsize_distribution" in adata.uns
    assert "mean_insertsize" in adata.obs.columns
