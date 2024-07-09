"""Test plotting functions."""

import pytest
import sctoolbox.plotting as pl
import scanpy as sc
import os
import numpy as np
import matplotlib.pyplot as plt

from beartype.roar import BeartypeCallHintParamViolation

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")


# ------------------------------ FIXTURES --------------------------------- #

quant_folder = os.path.join(os.path.dirname(__file__), 'data', 'quant')


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """Load and returns an anndata object."""

    np.random.seed(1)  # set seed for reproducibility

    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")
    adata = sc.read_h5ad(f)

    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
    adata.obs["clustering"] = np.random.choice(["1", "2", "3", "4"], size=adata.shape[0])
    adata.obs["cat"] = adata.obs["condition"].astype("category")

    adata.obs["LISI_score_pca"] = np.random.normal(size=adata.shape[0])
    adata.obs["qc_float"] = np.random.uniform(0, 1, size=adata.shape[0])
    adata.var["qc_float_var"] = np.random.uniform(0, 1, size=adata.shape[1])

    adata.obs["qcvar1"] = np.random.normal(size=adata.shape[0])
    adata.obs["qcvar2"] = np.random.normal(size=adata.shape[0])

    sc.pp.normalize_total(adata, target_sum=None)
    sc.pp.log1p(adata)

    sc.tl.umap(adata, n_components=3)
    sc.tl.tsne(adata)
    # sc.tl.pca(adata)
    sc.tl.rank_genes_groups(adata, groupby='clustering', method='t-test_overestim_var', n_genes=250)
    # sc.tl.dendrogram(adata, groupby='clustering')

    return adata


# ------------------------------ TESTS --------------------------------- #

@pytest.mark.parametrize("method", ["leiden", "louvain"])
def test_search_clustering_parameters(adata, method):
    """Test if search_clustering_parameters returns an array of axes."""

    axarr = pl.clustering.search_clustering_parameters(adata, method=method,
                                                       resolution_range=(0.1, 0.31, 0.1),
                                                       ncols=2)
    assert type(axarr).__name__ == "ndarray"
    assert axarr.shape == (2, 2)


def test_wrong_embeding_search_clustering_parameters(adata):
    """Test if search_cluster_parameters raises error."""
    with pytest.raises(KeyError):
        pl.clustering.search_clustering_parameters(adata, embedding="Invalid")


def test_search_clustering_parameters_errors(adata):
    """Test if search_clustering_parameters raises error."""

    with pytest.raises(ValueError):
        pl.clustering.search_clustering_parameters(adata, resolution_range=(0.1, 0.2, 0.3),
                                                   method="leiden")


def test_search_clustering_parameters_beartype(adata):
    """Test if beartype checks for tuple length."""

    with pytest.raises(BeartypeCallHintParamViolation):
        pl.clustering.search_clustering_parameters(adata, resolution_range=(0.1, 0.3, 0.1, 0.3),
                                                   method="leiden")

    with pytest.raises(BeartypeCallHintParamViolation):
        pl.clustering.search_clustering_parameters(adata, resolution_range=(0.1, 0.3, 0.1),
                                                   method="unknown")


def test_group_expression_boxplot(adata):
    """Test if group_expression_boxplot returns a plot."""
    gene_list = adata.var_names.tolist()[:10]
    ax = pl.marker_genes.group_expression_boxplot(adata, gene_list, groupby="condition")
    ax_type = type(ax).__name__

    # depending on matplotlib version, it can be either AxesSubplot or Axes
    assert ax_type.startswith("Axes")


def test_group_correlation(adata):
    """Test if plot is written to pdf."""

    # Run group correlation
    pl.qc_filter.group_correlation(adata, groupby="condition",
                                   save="group_correlation.pdf")

    # Assert creation of file
    assert os.path.isfile("group_correlation.pdf")
    os.remove("group_correlation.pdf")


@pytest.mark.parametrize("groupby", [None, "condition"])
@pytest.mark.parametrize("add_labels", [True, False])
def test_n_cells_barplot(adata, groupby, add_labels):
    """Test n_cells_barplot success."""

    axarr = pl.qc_filter.n_cells_barplot(adata, "clustering", groupby=groupby,
                                         add_labels=add_labels)

    if groupby is None:
        assert len(axarr) == 1
    else:
        assert len(axarr) == 2


@pytest.mark.parametrize("show_umap", [True, False])
def test_marker_gene_clustering(adata, show_umap):
    """Test marker_gene_clustering."""

    marker_dict = {"Celltype A": ['ENSMUSG00000103377', 'ENSMUSG00000104428'],
                   "Celltype B": ['ENSMUSG00000102272']}

    axes_list = pl.clustering.marker_gene_clustering(adata, "condition",
                                                     marker_dict, show_umap=show_umap)
    assert isinstance(axes_list, list)
    ax_type = type(axes_list[0]).__name__
    assert ax_type.startswith("Axes")
