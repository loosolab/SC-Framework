"""Test norm correct functions."""

import pytest
import os
import scanpy as sc
import numpy as np
import anndata as ad
import sctoolbox.tools as tools
import sctoolbox.utils as utils


# ------------------------- Fixtures -------------------------#


@pytest.fixture(scope="session")
def adata():
    """Load and returns an anndata object.

    Returns
    -------
    anndata.AnnData
        RNA-seq AnnData object with batch annotation and highly variable genes.
    """

    f = os.path.join(os.path.dirname(__file__), '../data', "adata.h5ad")
    adata = sc.read_h5ad(f)

    # Add batch column
    adata.obs['batch'] = ["a", "b"] * 100
    adata.obs['batch2'] = (["c", "d", "e"] * 100)[:len(adata.obs)]

    sc.pp.highly_variable_genes(adata)

    return adata


@pytest.fixture
def adata_mm10():
    """Fixture for an AnnData object.

    Returns
    -------
    anndata.AnnData
        ATAC-seq AnnData object.
    """
    adata_mm10 = sc.read_h5ad(os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_atac.h5ad'))
    return adata_mm10


# adapted from muon package
@pytest.fixture
def tfidf_x():
    """Create anndata with random expression.

    Returns
    -------
    anndata.AnnData
        AnnData object with random expression matrix for TF-IDF testing.
    """
    np.random.seed(2020)
    x = np.abs(np.random.normal(size=(4, 5)))
    adata_X = ad.AnnData(x)
    return adata_X


@pytest.fixture
def adata_batch_dict(adata):
    """Create dict containing adata with a batch column in obs.

    Returns
    -------
    dict
        Dictionary with AnnData object containing batch information.
    """
    anndata_batch_dict = adata.copy()

    return {'adata': anndata_batch_dict}


# ------------------------- Tests ------------------------- #


@pytest.mark.parametrize("method", ["tfidf", "total"])
def test_normalize_adata_success(adata_mm10, method):
    """Test normalize_adata success."""
    adata_norm = tools.norm_correct.normalize_adata(adata_mm10, method=method, target_sum=1e6)  # return from function is a dict

    if method == "tfidf":
        assert "X_lsi" in adata_norm.obsm and "lsi" in adata_norm.uns and "LSI" in adata_norm.varm
    elif method == "total":
        assert "X_pca" in adata_norm.obsm and "pca" in adata_norm.uns and "PCs" in adata_norm.varm


@pytest.mark.parametrize("method, keep_layer", [(["total", "tfidf"], "raw"), ("total", None), ("tfidf", "test")])
def test_normalize_adata(adata, method, keep_layer):
    """Test that data was normalized."""
    # Execute function
    result = tools.norm_correct.normalize_adata(adata, method=method, keep_layer=keep_layer, target_sum=1e6)
    # If method is a list, get the first element of the resulting dictionary
    if isinstance(method, list):
        method = method[0]
        adata = result[method]
    # If method is a string, get the resulting anndata object
    elif isinstance(method, str):
        adata = result

    # check for the layers
    if keep_layer:
        assert keep_layer in adata.layers

    # Check if the data was normalized
    mat = adata.X.todense()
    # Check if the data is a float array
    assert not utils.checker.is_integer_array(mat)
    # Check if the data is dimensionally reduced
    if method == "tfidf":
        assert "X_lsi" in adata.obsm and "lsi" in adata.uns and "LSI" in adata.varm
    elif method == "total":
        assert "X_pca" in adata.obsm and "pca" in adata.uns and "PCs" in adata.varm


# adapted from muon package
def test_tfidf(tfidf_x):
    """Test tfidf success."""
    tfidf_x_corrected = tools.norm_correct.tfidf(tfidf_x, log_tf=True, log_idf=True, inplace=False)
    assert str("%.3f" % tfidf_x_corrected.X[0, 0]) == "4.659"
    assert str("%.3f" % tfidf_x_corrected.X[3, 0]) == "4.770"

    tfidf_x.layers["test"] = tfidf_x.X
    tools.norm_correct.tfidf(tfidf_x, log_tf=True, log_idf=True, inplace=True, layer="test")
    assert str("%.3f" % tfidf_x.layers["test"][0, 0]) == "4.659"
    assert str("%.3f" % tfidf_x.layers["test"][3, 0]) == "4.770"


def test_wrap_corrections(adata):
    """Test if wrapper returns a dict, and that the keys contains the given methods."""

    methods = ["mnn", "scanorama"]  # two fastest methods
    adata_dict = tools.norm_correct.wrap_corrections(adata, batch_key="batch", methods=methods, keep_layer="test")

    assert isinstance(adata_dict, dict)

    keys = set(adata_dict.keys())
    assert len(set(methods) - keys) == 0

    for a in adata_dict.values():
        assert "test" in a.layers


@pytest.mark.parametrize("method", ["bbknn", "mnn", "harmony", "scanorama", "combat"])  # TODO excluded "scvi" due to runtime; may be mocked in the future
def test_batch_correction(adata, method):
    """Test if batch correction returns an anndata."""

    adata_corrected = tools.norm_correct.batch_correction(adata, batch_key="batch", method=method)
    assert isinstance(adata_corrected, sc.AnnData)
    # assert the returned adata is a different object
    # this is a workaround to test if the original adata was modified
    assert adata is not adata_corrected


@pytest.mark.parametrize("key", ["batch", ["batch", "batch2"]])
def test_evaluate_batch_effect(adata, key):
    """Test if AnnData containing LISI column in .obs is returned."""
    ad = tools.norm_correct.evaluate_batch_effect(adata, batch_key=key)

    ad_type = type(ad).__name__
    assert ad_type == "AnnData"
    assert "LISI_score" in ad.obs


@pytest.mark.parametrize("key", ["a", "b"])
def test_evaluate_batch_effect_keyerror(adata, key):
    """Test evaluate_batch_effect failure."""
    with pytest.raises(KeyError, match="adata.obsm .*"):
        tools.norm_correct.evaluate_batch_effect(adata, batch_key='batch', obsm_key=key)

    with pytest.raises(KeyError, match="adata.obs .*"):
        tools.norm_correct.evaluate_batch_effect(adata, batch_key=key)


@pytest.mark.parametrize("key", ["batch", ["batch", "batch2"]])
def test_wrap_batch_evaluation(adata_batch_dict, key):
    """Test if DataFrame containing LISI column in .obs is returned."""
    adata_dict = tools.norm_correct.wrap_batch_evaluation(adata_batch_dict, key, inplace=False)
    adata_dict_type = type(adata_dict).__name__
    adata_type = type(adata_dict['adata']).__name__

    assert adata_dict_type == "dict"
    assert adata_type == "AnnData"
