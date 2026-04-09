"""Test gsea plotting functions."""

import pytest
from sctoolbox.plotting import gsea


# ------------------------------ TESTS --------------------------------- #


def test_term_dotplot(adata_gsea, assert_axes_array):
    """Test term_dotplot success."""
    axes = gsea.term_dotplot(term="Actin Filament Organization (GO:0007015)",
                             adata=adata_gsea,
                             groupby="louvain")

    assert_axes_array(axes)


def test_gsea_cluster_dotplot(adata_gsea):
    """Test tsea_cluster_dotplot success."""
    axes_dict = gsea.cluster_dotplot(adata_gsea, save_figs=False)
    assert isinstance(axes_dict, dict)


def test_gsea_network(adata_gsea):
    """Test tsea_network success."""
    gsea.gsea_network(adata_gsea, cutoff=0.5)


def test_gsea_network_cutoff_too_low(adata_gsea):
    """Test gsea_network fails when no terms survive the cutoff."""
    with pytest.raises(ValueError):
        gsea.gsea_network(adata_gsea, cutoff=0.0000005)


def test_gsea_network_no_results(adata):
    """Test gsea_network fails when adata has no GSEA results."""
    with pytest.raises(ValueError, match="Could not find gsea results."):
        gsea.gsea_network(adata)
