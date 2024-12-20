"""Test tools/embedding.py functions."""

import pytest
import scanpy as sc

import sctoolbox.tools.embedding as ste


# ----------------------------- FIXTURES ------------------------------- #


@pytest.fixture(scope="session")
def adata():
    """Create an anndata object with PCA and UMAP."""
    return sc.datasets.pbmc3k_processed()


# ------------------------------ TESTS --------------------------------- #


# ---------------------------- correlation_matrix -----------------------------


@pytest.mark.parametrize("which,basis,n_components,ignore,method", [
    ("obs", "pca", 10, ["n_counts"], "spearmanr"),
    ("var", "pca", None, None, "pearsonr"),
    ("obs", "umap", None, None, "pearsonr"),
    ("obs", "tsne", None, None, "spearmanr")
])
def test_correlation_matrix_success(adata, which, basis, n_components, ignore, method):
    """Test success."""
    cor, pval = ste.correlation_matrix(adata=adata, which=which, basis=basis, n_components=n_components, ignore=ignore, method=method)

    # ensure there is a pvalue to each correlation
    assert cor.shape == pval.shape

    # test number of computed components
    assert len(cor.columns) == n_components if n_components else adata.obsm[f"X_{basis}"].shape[1]

    # test number of computed columns/ variables
    ignore_len = len(ignore) if ignore else 0

    if which == "obs":
        assert len(cor) == len(adata.obs.select_dtypes(include='number').columns) - ignore_len
    elif which == "var":
        assert len(cor) == len(adata.var.select_dtypes(include='number').columns) - ignore_len


def test_correlation_matrix_failure(adata):
    """Test invalid value for 'basis' parameter."""
    # invalid basis
    with pytest.raises(KeyError):
        ste.correlation_matrix(adata, basis="invalid")

    # invalid combination
    with pytest.raises(ValueError):
        ste.correlation_matrix(adata, which="var", basis="umap")


# --------------------------------- wrap_umap ---------------------------------


def test_wrap_umap(adata):
    """Test if X_umap is added to obsm in parallel."""

    adata_dict = {"adata_" + str(i): adata.copy() for i in range(3)}
    for adata in adata_dict.values():
        if "X_umap" in adata.obsm:
            del adata.obsm["X_umap"]

    ste.wrap_umap(adata_dict.values())

    for adata in adata_dict.values():
        assert "X_umap" in adata.obsm


def test_correlation_matrix_warning(adata):
    """Test invalid value for 'ignore' parameter."""
    # invalid basis
    with pytest.warns(Warning):
        ste.correlation_matrix(adata, ignore=["invalid"])
