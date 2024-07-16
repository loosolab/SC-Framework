"""Test norm correct functions."""

import pytest
import os
import scanpy as sc
import sctoolbox.tools as tools
import sctoolbox.utils as utils



# ------------------------- Fixtures -------------------------#


@pytest.fixture(scope="session")
def adata():
    """Load and returns an anndata object."""

    f = os.path.join(os.path.dirname(__file__), '../data', "adata.h5ad")
    adata = sc.read_h5ad(f)

    # Add batch column
    adata.obs['batch'] = ["a", "b"] * 100

    return adata


# ------------------------- Tests ------------------------- #


@pytest.mark.parametrize("method", [["total", "tfidf"], "total", "tfidf"])
def test_normalize_adata(adata, method):
    """Test that data was normalized."""
    # Execute function
    result = tools.norm_correct.normalize_adata(adata, method=method)
    # If method is a list, get the first element of the resulting dictionary
    if isinstance(method, list):
        method = method[0]
        adata = result[method]
    # If method is a string, get the resulting anndata object
    elif isinstance(method, str):
        adata = result

    # Check if the data was normalized
    mat = adata.X.todense()
    # Check if the data is a float array
    assert not utils.checker.is_integer_array(mat)
    # Check if the data is dimensionally reduced
    if method == "tfidf":
        assert "X_lsi" in adata.obsm and "lsi" in adata.uns and "LSI" in adata.varm
    elif method == "total":
        assert "X_pca" in adata.obsm and "pca" in adata.uns and "PCs" in adata.varm

