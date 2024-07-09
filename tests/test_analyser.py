"""Test analysis functions."""

import pytest
import scanpy as sc
import os

import sctoolbox.tools as tools
import sctoolbox.utils as utils


@pytest.fixture(scope="session")
def adata():
    """Load and returns an anndata object."""

    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")
    adata = sc.read_h5ad(f)

    # Add batch column
    adata.obs['batch'] = ["a", "b"] * 100

    return adata


@pytest.fixture
def adata_no_pca(adata):
    """Adata without PCA."""
    anndata = adata.copy()

    # remove pca
    anndata.uns.pop("pca")

    return anndata


@pytest.fixture
def adata_batch_dict(adata):
    """Create dict containing adata with a batch column in obs."""
    anndata_batch_dict = adata.copy()

    return {'adata': anndata_batch_dict}


# ------------------------------ TESTS -------------------------------- #


@pytest.mark.parametrize("method", ["total", "tfidf"])
def test_normalize_adata(adata, method):
    """Test that data was normalized."""
    result_dict = tools.norm_correct.normalize_adata(adata, method=method)
    adata = result_dict[method]
    mat = adata.X.todense()

    assert not utils.checker.is_integer_array(mat)


def test_evaluate_batch_effect(adata):
    """Test if AnnData containing LISI column in .obs is returned."""
    ad = tools.norm_correct.evaluate_batch_effect(adata, 'batch')

    ad_type = type(ad).__name__
    assert ad_type == "AnnData"
    assert "LISI_score" in ad.obs


@pytest.mark.parametrize("method", ["bbknn", "mnn", "harmony", "scanorama", "combat"])
def test_batch_correction(adata, method):
    """Test if batch correction returns an anndata."""

    adata_corrected = tools.norm_correct.batch_correction(adata, batch_key="batch", method=method)
    adata_type = type(adata_corrected).__name__
    assert adata_type == "AnnData"


def test_wrap_corrections(adata):
    """Test if wrapper returns a dict, and that the keys contains the given methods."""

    methods = ["mnn", "scanorama"]  # two fastest methods
    adata_dict = tools.norm_correct.wrap_corrections(adata, batch_key="batch", methods=methods)

    assert isinstance(adata_dict, dict)

    keys = set(adata_dict.keys())
    assert len(set(methods) - keys) == 0


@pytest.mark.parametrize("key", ["a", "b"])
def test_evaluate_batch_effect_keyerror(adata, key):
    """Test evaluate_batch_effect failure."""
    with pytest.raises(KeyError, match="adata.obsm .*"):
        tools.norm_correct.evaluate_batch_effect(adata, batch_key='batch', obsm_key=key)

    with pytest.raises(KeyError, match="adata.obs .*"):
        tools.norm_correct.evaluate_batch_effect(adata, batch_key=key)


def test_wrap_batch_evaluation(adata_batch_dict):
    """Test if DataFrame containing LISI column in .obs is returned."""
    adata_dict = tools.norm_correct.wrap_batch_evaluation(adata_batch_dict, 'batch', inplace=False)
    adata_dict_type = type(adata_dict).__name__
    adata_type = type(adata_dict['adata']).__name__

    assert adata_dict_type == "dict"
    assert adata_type == "AnnData"
