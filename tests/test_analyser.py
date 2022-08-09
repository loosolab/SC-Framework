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

@pytest.fixture
def adata_batch_dict(adata):
    """ Adata containing batch column in obs. """
    anndata_batch = adata.copy()

    # Add batch column
    anndata_batch.obs['batch'] = ["a", "b"] * 100

    return {"adata":anndata_batch}


def test_adata_normalize_total(adata):
    """ Test that data was normalized"""

    an.adata_normalize_total(adata, inplace=True)
    mat = adata.X.todense()
    assert utils.is_integer_array(mat) == False


def test_norm_log_PCA(adata_no_pca):
    """ Test if the returned adata has pca coordinates and highly variable genes """
    an.norm_log_PCA(adata_no_pca, inplace=True)

    check = ("X_pca" in adata_no_pca.obsm) and ("highly_variable" in adata_no_pca.var.columns)
    assert check


def test_define_PC(adata):
    """ Test if threshold is returned. """
    assert isinstance(an.define_PC(adata), int)


def test_define_PC_error(adata_no_pca):
    """ Test if error without PCA. """
    with pytest.raises(ValueError, match="PCA not found! Please make sure to compute PCA before running this function."):
        an.define_PC(adata_no_pca)


def test_evaluate_batch_effect(adata_batch_dict):
    """Test if Axes.Subplot is returned"""

    ax = an.evaluate_batch_effect(adata_batch_dict)
    ax_type = type(ax).__name__

    assert ax_type == "AxesSubplot"


@pytest.mark.parametrize("key",["a", "b"])
def test_evaluate_batch_effect_keyerror(adata_batch_dict, key):
    with pytest.raises(KeyError, match="adata.obsm of the .*"):
        an.evaluate_batch_effect(adata_batch_dict, obsm_key=key)

    with pytest.raises(KeyError, match="adata.obs of the .*"):
        an.evaluate_batch_effect(adata_batch_dict, batch_key=key)