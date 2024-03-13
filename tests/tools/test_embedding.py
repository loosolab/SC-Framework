"""Test embedding tool functions."""

import pytest
import scanpy as sc
import os
import sctoolbox.analyser as an


@pytest.fixture(scope="session")
def adata():
    """Load and returns an anndata object."""

    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")
    adata = sc.read_h5ad(f)

    # Add batch column
    adata.obs['batch'] = ["a", "b"] * 100

    return adata


def test_wrap_umap(adata):
    """Test if X_umap is added to obsm in parallel."""

    adata_dict = {"adata_" + str(i): adata.copy() for i in range(3)}
    for adata in adata_dict.values():
        if "X_umap" in adata.obsm:
            del adata.obsm["X_umap"]

    an.wrap_umap(adata_dict.values())

    for adata in adata_dict.values():
        assert "X_umap" in adata.obsm
