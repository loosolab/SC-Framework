"""Test tools/embedding.py functions."""

import pytest
import sctoolbox.tools.embedding as ste

import scanpy as sc

# -----------------------------------------------------------------------------
# ---------------------------------- Fixtures ---------------------------------
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def adata():
    """Create an anndata object with PCA and UMAP."""
    return sc.datasets.pbmc3k_processed()


# -----------------------------------------------------------------------------
# ----------------------------------- Tests -----------------------------------
# -----------------------------------------------------------------------------

# ---------------------------- correlation_matrix -----------------------------


@pytest.mark.parametrize("which,basis,n_components,columns,method", [
    ("obs", "pca", 10, ["n_genes", "percent_mito", "n_counts"], "spearmanr"),
    ("var", "pca", None, None, "pearsonr"),
    ("obs", "umap", None, None, "pearsonr"),
    ("obs", "tsne", None, None, "spearmanr")
])
def test_correlation_matrix_success(adata, which, basis, n_components, columns, method):
    """Test success."""
    cor, pval = ste.correlation_matrix(adata=adata, which=which, basis=basis, n_components=n_components, columns=columns, method=method)

    # ensure there is a pvalue to each correlation
    assert cor.shape == pval.shape

    # test number of computed components
    assert len(cor.columns) == n_components if n_components else adata.obsm[f"X_{basis}"].shape[1]

    # test number of computed columns/ variables
    if columns:
        assert len(cor) == len(columns)
    else:
        if which == "obs":
            assert len(cor) == len(adata.obs.select_dtypes(include='number').columns)
        elif which == "var":
            assert len(cor) == len(adata.var.select_dtypes(include='number').columns)


def test_correlation_matrix_failure(adata):
    """Test invalid value for 'basis' and 'columns' parameters."""
    # invalid basis
    with pytest.raises(KeyError):
        ste.correlation_matrix(adata, basis="invalid")

    # invalid combination
    with pytest.raises(ValueError):
        ste.correlation_matrix(adata, which="var", basis="umap")
