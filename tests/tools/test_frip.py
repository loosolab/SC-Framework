"""Test the frip score calculation function."""
import pytest
import scanpy as sc
import os
import sctoolbox.tools as tools


# ----------------------------- FIXTURES ------------------------------- #


@pytest.fixture
def adata():
    """Fixture for an AnnData object."""
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_atac.h5ad'))
    return adata


@pytest.fixture
def fragments():
    """Fixture for a fragments object."""
    fragments = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_sorted_fragments.bed')
    return fragments


def test_calc_frip_scores(adata, fragments):
    """Test the calc_frip_scores function."""
    adata, total_frip = tools.frip.calc_frip_scores(adata, fragments, temp_dir='')

    assert 'frip' in adata.obs.columns
