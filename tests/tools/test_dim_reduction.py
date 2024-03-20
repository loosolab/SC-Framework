"""Test tools/dim_reduction.py functions."""

import pytest
import sctoolbox.tools.dim_reduction as std

import scanpy as sc

# -----------------------------------------------------------------------------
# ---------------------------------- Fixtures ---------------------------------
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def adata_no_pca():
    """Create an anndata object without PCA."""
    return sc.datasets.pbmc3k()


@pytest.fixture(scope="session")
def adata_pca():
    """Create an anndata object with precalculated PCA."""
    return sc.datasets.pbmc3k_processed()


# -----------------------------------------------------------------------------
# ----------------------------------- Tests -----------------------------------
# -----------------------------------------------------------------------------

# ------------------------------------ lsi ------------------------------------

def test_lsi(adata_no_pca):
    """Test lsi success."""
    adata = adata_no_pca.copy()

    std.lsi(adata)
    assert "X_lsi" in adata.obsm and "lsi" in adata.uns and "LSI" in adata.varm


# --------------------------------- define_PC ---------------------------------


def test_define_PC(adata_pca):
    """Test if threshold is returned."""
    assert isinstance(std.define_PC(adata_pca), int)


def test_define_PC_error(adata_no_pca):
    """Test if error without PCA."""
    with pytest.raises(ValueError, match="PCA not found! Please make sure to compute PCA before running this function."):
        std.define_PC(adata_no_pca)


# -------------------------------- propose_pcs --------------------------------


def test_propose_pcs_failure(adata_no_pca):
    """Test the propose_pcs function fails without precomputed PCA."""
    with pytest.raises(ValueError):
        std.propose_pcs(anndata=adata_no_pca)


def test_propose_pcs_succsess(adata_pca):
    """Test propose_pcs success."""
    # test knee finding option
    assert [1, 3, 4, 5, 6] == std.propose_pcs(anndata=adata_pca,
                                              how=["variance", "cumulative variance", "correlation"],
                                              var_method="knee")

    # test percentile finding option
    assert [1, 3, 4, 5] == std.propose_pcs(anndata=adata_pca,
                                           how=["variance", "cumulative variance", "correlation"],
                                           var_method="percent",
                                           perc_thresh=10)


# -------------------------------- subset_pca --------------------------------


def test_subset_PCA(adata_pca):
    """Test whether number of PCA coordinate dimensions was reduced."""
    adata_copy = adata_pca.copy()

    # test range selection, not inplace
    res_adata = std.subset_PCA(adata=adata_copy,
                               n_pcs=5,
                               start=2,
                               inplace=False)

    # test inplace
    assert adata_pca.obsm["X_pca"].shape[1] == adata_copy.obsm["X_pca"].shape[1]
    assert res_adata.obsm["X_pca"].shape[1] != adata_copy.obsm["X_pca"].shape[1]
    # test PC amount
    assert res_adata.obsm["X_pca"].shape[1] == 3

    # test custom selection, inplace
    select = [2, 4, 6, 8]
    cstm_res_adata = std.subset_PCA(adata=adata_copy,
                                    select=select,
                                    inplace=True)

    assert cstm_res_adata is None
    assert adata_copy.obsm["X_pca"].shape[1] == len(select)
    assert adata_copy.obsm["X_pca"].shape[1] != adata_pca.obsm["X_pca"].shape[1]
