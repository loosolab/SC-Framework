import pytest
import scanpy as sc
import os
import numpy as np
import pandas as pd

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
def adata_batch(adata):
    """ Adata containing batch column in obs. """
    anndata_batch = adata.copy()

    # Add batch column
    anndata_batch.obs['batch'] = ["a", "b"] * 100

    return anndata_batch


@pytest.fixture
def adata_batch_dict(adata_batch):
    """ dict containing Adata containing batch column in obs. """
    anndata_batch_dict = adata_batch.copy()

    return {'adata': anndata_batch_dict}


# ------------------------------ TESTS -------------------------------- #

def test_rename_categories():
    """ Assert if categories were renamed"""

    data = np.random.choice(["C1", "C2", "C3"], size=100)
    series = pd.Series(data).astype("category")
    renamed_series = an.rename_categories(series)

    assert renamed_series.cat.categories.tolist() == ["1", "2", "3"]


def test_wrap_umap(adata):
    """ Test if X_umap is added to obsm in parallel """

    adata_dict = {"adata_" + str(i): adata.copy() for i in range(3)}
    for adata in adata_dict.values():
        if "X_umap" in adata.obsm:
            del adata.obsm["X_umap"]

    an.wrap_umap(adata_dict.values())

    for adata in adata_dict.values():
        assert "X_umap" in adata.obsm


def test_adata_normalize_total(adata):
    """ Test that data was normalized"""
    an.adata_normalize_total(adata, inplace=True)
    mat = adata.X.todense()

    assert not utils.is_integer_array(mat)


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


def test_subset_PCA(adata):
    """ Test whether number of PCA coordinate dimensions was reduced """

    an.subset_PCA(adata, 10)

    assert adata.obsm["X_pca"].shape[1] == 10


def test_evaluate_batch_effect(adata_batch):
    """ Test if AnnData containing LISI column in .obs is returned. """
    ad = an.evaluate_batch_effect(adata_batch, 'batch')
    ad_type = type(ad).__name__
    assert ad_type == "AnnData"
    print(ad.obs)
    assert "LISI_score" in ad.obs


@pytest.mark.parametrize("method", ["bbknn", "mnn", "harmony", "scanorama", "combat"])
def test_batch_correction(adata_batch, method):
    """ Test if batch correction returns an anndata """

    adata_corrected = an.batch_correction(adata_batch, batch_key="batch", method=method)
    adata_type = type(adata_corrected).__name__
    assert adata_type == "AnnData"


def test_wrap_corrections(adata_batch):
    """ Test if wrapper returns a dict, and that the keys contains the given methods """

    methods = ["mnn", "harmony"]
    adata_dict = an.wrap_corrections(adata_batch, batch_key="batch", methods=methods)

    assert isinstance(adata_dict, dict)

    keys = set(adata_dict.keys())
    assert len(set(methods) - keys) == 0


@pytest.mark.parametrize("key", ["a", "b"])
def test_evaluate_batch_effect_keyerror(adata_batch, key):
    with pytest.raises(KeyError, match="adata.obsm .*"):
        an.evaluate_batch_effect(adata_batch, batch_key='batch', obsm_key=key)

    with pytest.raises(KeyError, match="adata.obs .*"):
        an.evaluate_batch_effect(adata_batch, batch_key=key)


def test_wrap_batch_evaluation(adata_batch_dict):
    """ Test if DataFrame containing LISI column in .obs is returned. """
    adata_dict = an.wrap_batch_evaluation(adata_batch_dict, 'batch', inplace=False)
    adata_dict_type = type(adata_dict).__name__
    adata_type = type(adata_dict['adata']).__name__

    assert adata_dict_type == "dict"
    assert adata_type == "AnnData"