"""Test gsea plotting functions."""

import pytest
import scanpy as sc
import numpy as np
import sctoolbox.tools as tools
from sctoolbox.plotting import gsea


# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """Minimal adata file for testing."""
    adata = sc.datasets.pbmc68k_reduced()
    tools.marker_genes.run_rank_genes(adata, "louvain")
    tools.gsea.gene_set_enrichment(adata,
                                   marker_key="rank_genes_louvain_filtered",
                                   organism="human",
                                   method="prerank",
                                   inplace=True)
    return adata

# ------------------------------ TESTS --------------------------------- #


def test_term_dotplot(adata):
    """Test term_dotplot success."""
    axes = gsea.term_dotplot(term="Actin Filament Organization (GO:0007015)",
                             adata=adata,
                             groupby="louvain")

    assert isinstance(axes, np.ndarray)
    ax_type = type(axes[0]).__name__
    assert ax_type.startswith("Axes")


def test_gsea_cluster_dotplot(adata):
    """Test tsea_cluster_dotplot success."""
    axes_dict = gsea.cluster_dotplot(adata)
    assert isinstance(axes_dict, dict)


def test_gsea_network(adata):
    """Test tsea_network success."""
    gsea.gsea_network(adata, cutoff=0.5)


def test_gsea_network_fail(adata):
    """Test tsea_network success."""
    with pytest.raises(ValueError):
        gsea.gsea_network(adata, cutoff=0.0000005)
    with pytest.raises(ValueError, match="Could not find gsea results."):
        gsea.gsea_network(sc.datasets.pbmc68k_reduced())
