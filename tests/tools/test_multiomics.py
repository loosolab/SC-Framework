"""Test multiomics functions."""

import pytest
import os
import anndata
import scanpy as sc
import numpy as np

import sctoolbox.multiomics as multi


@pytest.fixture
def adata():
    """Load and returns an anndata object."""
    f = os.path.join(os.path.dirname(__file__), '../data', "adata.h5ad")

    return sc.read_h5ad(f)


@pytest.fixture
def adata2(adata):
    """Build second adata from first."""
    adata2 = adata.copy()
    adata2.obsm['X_pca'] = np.random.uniform(low=-3, high=3, size=(200, 50))
    adata2.obsm['X_umap'] = np.random.uniform(low=-30, high=70, size=(200, 3))
    return adata2


def test_merge_anndata(adata, adata2):
    """Test if anndata are merged correctly."""
    adata_to_merge = {"1": adata, "2": adata2}
    merged_adata = multi.merge_anndata(adata_to_merge)

    new_obs_index = list(set(adata.obs.index) & set(adata2.obs.index))
    new_obs_cols, new_obsm_entries, new_var_index = list(), list(), list()
    new_var_cols = ["source"]
    for key, value in adata_to_merge.items():
        new_var_cols += [f"{key}_{i}" for i in value.var.columns]
        new_obs_cols += [f"{key}_{i}" for i in value.obs.columns]
        new_obsm_entries += [f"X_{key}_{i.split('_')[-1]}" for i in list(value.obsm)]
        new_var_index += [f"{key}_{i}" for i in value.var.index]

    # Check if merged object is an AnnData object
    assert isinstance(merged_adata, anndata.AnnData)
    # Check if merged obsm entries are merged properly
    assert all(elem in new_obsm_entries for elem in list(merged_adata.obsm))
    # Check if merged var columns are merged properly
    assert all(elem in new_var_cols for elem in merged_adata.var.columns)
    # Check if merged obs columns are merged properly
    assert all(elem in new_obs_cols for elem in merged_adata.obs.columns)
    # Check if merged var index is merged properly
    assert all(elem in new_var_index for elem in merged_adata.var.index)
    # Check if merged obs index is merged properly
    assert all(elem in new_obs_index for elem in merged_adata.obs.index)


def test_deep_merge_anndata(adata, adata2):
    """Test if anndata are merged correctly by checking obsm coordinates and matrix values for each cell."""
    adata_to_merge = {"1": adata, "2": adata2}
    merged_adata = multi.merge_anndata(adata_to_merge)

    for cell_id in merged_adata.obs.index:
        m_index = list(merged_adata.obs.index).index(cell_id)
        r_index = list(adata.obs.index).index(cell_id)
        c_index = list(adata2.obs.index).index(cell_id)
        m = merged_adata.X.tocsr()[m_index, :].todense().tolist()[0]
        r = adata.X.tocsr()[r_index, :].todense().tolist()[0]
        c = adata2.X.tocsr()[c_index, :].todense().tolist()[0]

        adata2.obsm["X_umap"][c_index]

        assert (list(merged_adata.obsm["X_1_umap"][m_index]) == list(adata.obsm["X_umap"][r_index]))
        assert (list(merged_adata.obsm["X_2_umap"][m_index]) == list(adata2.obsm["X_umap"][c_index]))
        assert merged_adata.obs.index[m_index] == adata.obs.index[r_index]
        assert merged_adata.obs.index[m_index] == adata2.obs.index[c_index]
        assert (m[:len(r)] == r) and (m[-len(c):] == c)
