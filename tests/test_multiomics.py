import pytest
import os
import anndata
import scanpy as sc
import numpy as np

import sctoolbox.multiomics as multi


@pytest.fixture
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")

    return sc.read_h5ad(f)


@pytest.fixture
def adata2(adata):
    """ Build second adata from first """
    adata2 = adata.copy()
    adata2.obsm['X_pca'] = np.random.uniform(low=-3, high=3, size=(200, 50))
    adata2.obsm['X_umap'] = np.random.uniform(low=-30, high=70, size=(200, 3))
    return adata2


def test_merge_anndata(adata, adata2):
    """ Test if anndata are merged correctly """
    merged_adata = multi.merge_anndata({"1": adata, "2": adata2})

    assert isinstance(merged_adata, anndata.AnnData)
    assert list(merged_adata.obsm) == ['X_1_pca', 'X_1_umap', 'X_2_pca', 'X_2_umap']

# ToDo test for var and obs columns
