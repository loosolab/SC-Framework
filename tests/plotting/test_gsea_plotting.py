"""Test gsea plotting functions."""

import pytest
import scanpy as sc
import numpy as np
from sctoolbox.plotting import gsea


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
