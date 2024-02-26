"""Test dimensionality reduction functions."""

import pytest
import os
import numpy as np
import scanpy as sc
import sctoolbox.tools.dim_reduction as dim_reduct


@pytest.fixture
def adata():
    """Fixture for an AnnData object."""
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), 'data', 'atac', 'anndata_2.h5ad'))
    return adata


def test_lsi(adata):
    """Test lsi success."""
    adata_ori = adata.copy()

    dim_reduct.lsi(adata_ori, use_highly_variable=True)
    assert "X_lsi" in adata_ori.obsm and "lsi" in adata_ori.uns and "LSI" in adata_ori.varm
    assert np.sum(adata_ori.varm['LSI'][~adata_ori.var['highly_variable']]) == 0

    dim_reduct.lsi(data=adata,
                   use_highly_variable=False)

    assert np.sum(adata.varm['LSI'][~adata.var['highly_variable']]) != 0
