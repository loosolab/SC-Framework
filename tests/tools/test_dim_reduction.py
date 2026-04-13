"""Test tools/dim_reduction.py functions."""

import pytest
import sctoolbox.tools.dim_reduction as std

import scanpy as sc
import numpy as np
import os


# ----------------------------- FIXTURES ------------------------------- #


@pytest.fixture
def adata_hv():
    """Load ATAC-seq AnnData with QC metrics and highly_variable annotation.

    Uses anndata_2.h5ad instead of the shared adata_atac fixture because
    test_lsi requires 'highly_variable' in var, which mm10_atac.h5ad lacks.

    Returns
    -------
    anndata.AnnData
        ATAC-seq AnnData object with highly_variable annotation.
    """
    return sc.read_h5ad(os.path.join(os.path.dirname(__file__), '../data', 'atac', 'anndata_2.h5ad'))


# ------------------------------ TESTS --------------------------------- #


# ------------------------------------ lsi ------------------------------------

@pytest.mark.parametrize("use_highly_variable", [True, False])
def test_lsi(adata_hv, use_highly_variable):
    """Test lsi success."""
    assert "X_lsi" not in adata_hv.obsm and "lsi" not in adata_hv.uns and "LSI" not in adata_hv.varm

    std.lsi(adata_hv, use_highly_variable=use_highly_variable)

    assert "X_lsi" in adata_hv.obsm and "lsi" in adata_hv.uns and "LSI" in adata_hv.varm

    if use_highly_variable:
        assert np.sum(adata_hv.varm['LSI'][~adata_hv.var['highly_variable']]) == 0
    else:
        assert np.sum(adata_hv.varm['LSI'][~adata_hv.var['highly_variable']]) != 0

    assert np.sum(adata_hv.varm['LSI'][adata_hv.var['highly_variable']]) != 0


# -------------------------------- propose_pcs --------------------------------


def test_propose_pcs_failure(adata_raw):
    """Test the propose_pcs function fails without precomputed PCA."""
    with pytest.raises(ValueError):
        std.propose_pcs(anndata=adata_raw)


@pytest.mark.parametrize("var_method, kwargs", [("knee", {}), ("percent", {"perc_thresh": 10})])
def test_propose_pcs_succsess(adata, var_method, kwargs):
    """Test propose_pcs success."""
    n_pcs = adata.obsm["X_pca"].shape[1]

    result = std.propose_pcs(anndata=adata,
                             how=["variance", "cumulative variance", "correlation"],
                             var_method=var_method,
                             **kwargs)

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(pc, (int, np.integer)) for pc in result)
    assert all(1 <= pc <= n_pcs for pc in result)
    assert result == sorted(result)


# -------------------------------- subset_pca --------------------------------


@pytest.mark.parametrize("inplace, kwargs, expected_n_pcs", [
    (False, {"n_pcs": 5, "start": 2}, 3),
    (True, {"select": [2, 4, 6, 8]}, 4),
])
def test_subset_PCA(adata, inplace, kwargs, expected_n_pcs):
    """Test whether number of PCA coordinate dimensions was reduced."""
    adata_copy = adata.copy()
    n_pcs_orig = adata.obsm["X_pca"].shape[1]

    result = std.subset_PCA(adata=adata_copy, inplace=inplace, **kwargs)

    if inplace:
        assert result is None
        assert adata_copy.obsm["X_pca"].shape[1] == expected_n_pcs
    else:
        assert adata_copy.obsm["X_pca"].shape[1] == n_pcs_orig
        assert result.obsm["X_pca"].shape[1] == expected_n_pcs


# -------------------------------- subset_pca --------------------------------


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("method", ["PCA", "LSI"])
def test_dim_red(adata_raw, method, inplace):
    """Test the dim_red function."""
    adata = adata_raw.copy()

    # check there is no dimension reduction and neighbor graph
    assert "X_pca" not in adata.obsm.keys()
    assert "neighbors" not in adata.uns.keys()

    method_kwargs = {}
    if method == "PCA":
        method_kwargs = {"mask_var": None}

    out = std.dim_red(anndata=adata, method=method, method_kwargs=method_kwargs, inplace=inplace)

    # everything is calculated
    if inplace:
        assert "X_pca" in adata.obsm.keys()
        assert "neighbors" in adata.uns.keys()
        assert out is None
    else:
        assert "X_pca" in out.obsm.keys()
        assert "neighbors" in out.uns.keys()
