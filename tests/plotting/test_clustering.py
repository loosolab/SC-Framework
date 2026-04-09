"""Tests for clustering plotting functions."""

import pytest
import sctoolbox.plotting.clustering as pl
import numpy as np

from beartype.roar import BeartypeCallHintParamViolation


# ------------------------------ TESTS --------------------------------- #


@pytest.mark.parametrize("method", ["leiden"])
def test_search_clustering_parameters(adata, method):
    """Test if search_clustering_parameters returns an array of axes."""

    axarr = pl.search_clustering_parameters(adata, method=method,
                                            resolution_range=(0.1, 0.31, 0.1),
                                            ncols=2)
    assert isinstance(axarr, np.ndarray)
    assert axarr.shape == (2, 2)


def test_wrong_embedding_search_clustering_parameters(adata):
    """Test if search_cluster_parameters raises error."""
    with pytest.raises(KeyError):
        pl.search_clustering_parameters(adata, embedding="Invalid")


def test_search_clustering_parameters_errors(adata):
    """Test if search_clustering_parameters raises error."""

    with pytest.raises(ValueError):
        pl.search_clustering_parameters(adata, resolution_range=(0.1, 0.2, 0.3),
                                        method="leiden")


@pytest.mark.parametrize("resolution_range,method", [
    ((0.1, 0.3, 0.1, 0.3), "leiden"),   # invalid tuple length
    ((0.1, 0.3, 0.1), "unknown"),        # invalid method
])
def test_search_clustering_parameters_beartype(adata, resolution_range, method):
    """Test if beartype checks for tuple length."""
    with pytest.raises(BeartypeCallHintParamViolation):
        pl.search_clustering_parameters(adata, resolution_range=resolution_range, method=method)


@pytest.mark.parametrize("show_umap", [True, False])
def test_marker_gene_clustering(adata, show_umap, assert_axes_array):
    """Test marker_gene_clustering."""

    marker_dict = {"Celltype A": list(adata.var.index[:3]),
                   "Celltype B": list(adata.var.index[-2:])}

    axes_list = pl.marker_gene_clustering(adata, "condition",
                                          marker_dict, show_umap=show_umap)
    assert_axes_array(axes_list)
