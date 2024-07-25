"""Tests for Velocity plotting functions."""

import pytest
import sctoolbox.plotting.velocity as pl
import scanpy as sc
import os
import numpy as np


# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """Load and returns an anndata object."""

    np.random.seed(1)  # set seed for reproducibility

    f = os.path.join(os.path.dirname(__file__), '../data', "adata.h5ad")
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


@pytest.mark.parametrize("sortby, title, figsize, layer",
                         [("condition", "condition", None, "spliced"),
                          (None, None, (4, 4), None)],
                         )
def test_pseudotime_heatmap(adata, sortby, title, figsize, layer):
    """Test pseudotime_heatmap success."""
    ax = pl.pseudotime_heatmap(adata, ['ENSMUSG00000103377',
                                       'ENSMUSG00000102851'],
                               sortby=sortby, title=title,
                               figsize=figsize, layer=layer)
    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")
