import pytest
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

    return {"adata": anndata_batch}


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


def test_evaluate_batch_effect(adata_batch_dict):
    """ Test if DataFrame containing LISI column in .obs is returned. """
    df = an.evaluate_batch_effect(adata_batch_dict['adata'], 'batch')
    df_type = type(df).__name__
    assert df_type == "DataFrame"
    assert "LISI_score_X_umap" in df.obs


@pytest.mark.parametrize("key", ["a", "b"])
def test_evaluate_batch_effect_keyerror(adata_batch_dict, key):
    with pytest.raises(KeyError, match="adata.obsm does not contain the obsm key: .*"):
        an.evaluate_batch_effect(adata_batch_dict['adata'], batch_key='batch', obsm_key=key)

    with pytest.raises(KeyError, match="adata.obs does not contain the obs key: .*"):
        an.evaluate_batch_effect(adata_batch_dict['adata'], batch_key=key)


def test_wrap_batch_evaluation(adata_batch_dict):
    """ Test if DataFrame containing LISI column in .obs is returned. """
    adata_dict = an.wrap_batch_evaluation(adata_batch_dict, 'batch')
    adata_dict_type = type(adata_dict).__name__
    adata_type = type(adata_dict['adata']).__name__

    assert adata_dict_type == "dict"
    assert adata_type == "AnnData"