"""Test tools/gsea.py functions."""

import pytest
import os
import scanpy as sc
import sctoolbox.tools as tools
import gseapy as gp
import pandas as pd


@pytest.fixture
def adata():
    """Return clustered dataset with ranked genes."""
    obj = sc.datasets.pbmc3k_processed()
    tools.marker_genes.run_rank_genes(obj, "louvain")
    return obj


def test_enrichr_marker_genes(adata):
    """Test enrichr_marker_genes."""
    result = tools.gsea.enrichr_marker_genes(adata,
                                             marker_key="rank_genes_cluster_filtered",
                                             organism="human")

    assert isinstance(result, pd.DataFrame)


def test_fail_enrichr_marker_genes(adata):
    """Test if invalid marker key is caught by enrichr_marker_genes."""

    with pytest.raises(KeyError):
        tools.gsea.enrichr_marker_genes(adata,
                                        marker_key="invalid",
                                        organism="human")
