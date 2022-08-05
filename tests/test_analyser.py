import pytest
import pandas as pd
import numpy as np
import scanpy as sc
import os
import sctoolbox.analyser as an
import sctoolbox.utilities as utils


@pytest.fixture
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")

    return sc.read_h5ad(f)


@pytest.fixture
def adata_no_pca(adata):
    """ Adata without PCA. """
    anndata = adata.copy()

    # remove pca
    anndata.uns.pop("pca")

    return anndata

def test_adata_normalize_total(adata):
    """ Test that data was normalized"""

    an.adata_normalize_total(adata)
    assert utils.is_integer_array(adata.X) is False


def test_norm_log_PCA(adata_no_pca):
    """ Test if the returned adata has pca coordinates and highly variable genes """
    an.norm_log_PCA(adata_no_pca)

    check = ("X_pca" in adata.obsm) and ("highly_variable" in adata.var.columns)
    assert check


def test_define_PC(adata):
    """ Test if threshold is returned. """
    assert isinstance(an.define_PC(adata), int)


def test_define_PC_error(adata_no_pca):
    """ Test if error without PCA. """
    with pytest.raises(ValueError, match="PCA not found! Please make sure to compute PCA before running this function."):
        an.define_PC(adata_no_pca)
