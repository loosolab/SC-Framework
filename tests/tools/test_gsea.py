"""Test tools/gsea.py functions."""

import pytest
import scanpy as sc
import sctoolbox.tools as tools
import pandas as pd


# ---------------------------- FIXTURES -------------------------------- #


@pytest.fixture
def adata():
    """Return clustered dataset with ranked genes."""
    obj = sc.datasets.pbmc3k_processed()
    tools.marker_genes.run_rank_genes(obj, "louvain")

    return obj


# ------------------------------ TESTS --------------------------------- #


def test_enrichr_marker_genes(adata):
    """Test enrichr_marker_genes."""
    result = tools.gsea.enrichr_marker_genes(adata,
                                             marker_key="rank_genes_louvain_filtered",
                                             organism="human")
    cols = ['Gene_set', 'Term', 'Overlap', 'P-value', 'Adjusted P-value', 'Odds Ratio', 'Combined Score', 'Genes']
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) > 0
    assert set(cols).issubset(set(result.columns))


def test_fail_enrichr_marker_genes(adata):
    """Test if invalid marker key is caught by enrichr_marker_genes."""

    with pytest.raises(KeyError):
        tools.gsea.enrichr_marker_genes(adata,
                                        marker_key="invalid",
                                        organism="human")
