"""Test clustering functions."""

import numpy as np
import sctoolbox.tools as tl
import pytest
import scanpy as sc


@pytest.fixture
def equal_adata():
    """Build a mock anndata object with equally distributed features."""

    mtx = equal_mtx()
    adata = build_adata(mtx)

    return adata
    

@pytest.fixture
def clust_adata():
    """Return a clustered adata."""
    return sc.datasets.pbmc3k_processed()


def test_recluster(clust_adata):
    """Test recluster success."""
    # join monocytes clusters
    tl.recluster(adata=clust_adata,
                 column="louvain",
                 clusters=["CD14+ Monocytes", "FCGR3A+ Monocytes"],
                 task="join",
                 method="leiden",
                 resolution=1,
                 key_added="joined_louvain",
                 plot=True,
                 embedding="X_umap")

    assert "joined_louvain" in clust_adata.obs.columns
    assert len(set(clust_adata.obs["louvain"])) - 1 == len(set(clust_adata.obs["joined_louvain"]))

    # split cluster
    tl.recluster(adata=clust_adata,
                 column="louvain",
                 clusters=["CD4 T cells"],
                 task="split",
                 method="leiden",
                 resolution=1,
                 key_added="split_louvain",
                 plot=True,
                 embedding="X_umap")

    assert "split_louvain" in clust_adata.obs.columns
    assert len(set(clust_adata.obs["louvain"])) < len(set(clust_adata.obs["split_louvain"]))


def test_gini():
    """Test gini function."""

    test_array_equal = np.full(10, 5)

    test_array_unequal = np.zeros(100)
    test_array_unequal[-1] = 100

    assert tl.gini(test_array_equal) == 0
    assert tl.gini(test_array_unequal) == 0.99


def test_calc_ragi(equal_adata, unequal_adata):
    """Test calc_ragi function."""

    _, ragi_equal = tl.calc_ragi(equal_adata, "cluster", None)
    _, ragi_unequal = tl.calc_ragi(unequal_adata, "cluster", None)

    assert ragi_equal == 0
    assert round(ragi_unequal, 2) == 0.9


def build_adata(mtx):
    """Build mock anndata object."""

    # define the number of observations (obs) and var regions
    n_obs = 30
    n_vars = 30

    obs_df = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    obs_df['cluster'] = [i for i in range(1, 11) for _ in range(3)]
    var_df = pd.DataFrame(index=[f"var_{i}" for i in range(n_vars)])
    var_df['feature'] = np.arange(1, 31, 1)

    adata = anndata.AnnData(X=mtx, obs=obs_df, var=var_df)
    adata.var["total_counts"] = adata.X.sum(axis=1)

    return adata


def unequal_mtx():
    """Build a mock mtx with unequal distribution."""

    index = np.arange(0, 31, 3)
    zero_arr = np.zeros((30, 30))

    for i in range(0, len(index) - 1):
        zero_arr[index[i]:index[i + 1], index[i]:index[i + 1]] = 1

    mtx = sparse.csr_matrix(zero_arr)

    return mtx


def equal_mtx():
    """Build a mock mtx with equal distribution."""

    ones_arr = np.ones((30, 30))
    mtx = sparse.csr_matrix(ones_arr)

    return mtx
