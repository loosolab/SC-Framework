"""Test gsea plotting functions."""

import pytest
import matplotlib
import numpy as np
from sctoolbox.plotting import gsea


# ------------------------------ TESTS --------------------------------- #


def test_term_dotplot(adata_gsea):
    """Test term_dotplot success."""
    axes = gsea.term_dotplot(term="Actin Filament Organization (GO:0007015)",
                             adata=adata_gsea,
                             groupby="louvain")

    assert isinstance(axes, np.ndarray)
    assert isinstance(axes[0], matplotlib.axes.Axes)


def test_gsea_cluster_dotplot(adata_gsea):
    """Test tsea_cluster_dotplot success."""
    axes_dict = gsea.cluster_dotplot(adata_gsea, save_figs=False)
    assert isinstance(axes_dict, dict)


def test_gsea_network(adata_gsea):
    """Test tsea_network success."""
    gsea.gsea_network(adata_gsea, cutoff=0.5)


@pytest.mark.parametrize("adata_name,kwargs,match", [
    ("adata_gsea", {"cutoff": 0.0000005}, None),
    ("adata", {}, "Could not find gsea results."),
])
def test_gsea_network_fail(request, adata_name, kwargs, match):
    """Test gsea_network failure."""
    adata = request.getfixturevalue(adata_name)
    with pytest.raises(ValueError, match=match):
        gsea.gsea_network(adata, **kwargs)
