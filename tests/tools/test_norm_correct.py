"""Test norm correct functions."""

import pytest
import scanpy as sc
import numpy as np
import anndata as ad
import sctoolbox.tools as tools
import sctoolbox.utils as utils


# ------------------------- Fixtures -------------------------#


@pytest.fixture
def adata_with_batch(adata_fun_scope):
    """Return an AnnData with batch annotation and highly variable genes.

    Returns
    -------
    anndata.AnnData
        AnnData object with batch annotation and highly variable genes.
    """
    adata_fun_scope.obs['batch'] = (["a", "b"] * ((len(adata_fun_scope) // 2) + 1))[:len(adata_fun_scope)]
    adata_fun_scope.obs['batch2'] = (["c", "d", "e"] * ((len(adata_fun_scope) // 3) + 1))[:len(adata_fun_scope)]

    sc.pp.highly_variable_genes(adata_fun_scope)

    return adata_fun_scope


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
def adata_batch_dict(adata_with_batch):
    """Create dict containing adata with a batch column in obs.

    Returns
    -------
    dict
        Dictionary with AnnData object containing batch information.
    """
    anndata_batch_dict = adata_with_batch.copy()

    return {'adata': anndata_batch_dict}


# ------------------------- Tests ------------------------- #


@pytest.mark.parametrize("method", ["tfidf", "total"])
def test_normalize_adata_success(adata_atac, method):
    """Test normalize_adata success."""
    # The assertions only check for presence of the dim-reduction keys, so a small
    # subset suffices. Both dimensions stay > the default n_comps (50) so the
    # LSI/SVD (svds k < min(shape)) and PCA paths remain valid.
    adata_sub = adata_atac[:100, :200].copy()
    adata_norm = tools.norm_correct.normalize_adata(adata_sub, method=method, target_sum=1e6)  # return from function is a dict

    if method == "tfidf":
        assert "X_lsi" in adata_norm.obsm and "lsi" in adata_norm.uns and "LSI" in adata_norm.varm
    elif method == "total":
        assert "X_pca" in adata_norm.obsm and "pca" in adata_norm.uns and "PCs" in adata_norm.varm


@pytest.mark.parametrize("method, keep_layer", [(["total", "tfidf"], "raw"), ("total", None), ("tfidf", "test")])
def test_normalize_adata(adata_raw_small, method, keep_layer):
    """Test that data was normalized."""
    adata = adata_raw_small.copy()
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


def test_wrap_corrections(adata_with_batch):
    """Test if wrapper returns a dict, and that the keys contains the given methods."""

    methods = ["mnn", "scanorama"]  # two fastest methods
    adata_dict = tools.norm_correct.wrap_corrections(adata_with_batch, batch_key="batch", methods=methods, keep_layer="test")

    assert isinstance(adata_dict, dict)

    keys = set(adata_dict.keys())
    assert len(set(methods) - keys) == 0

    for a in adata_dict.values():
        assert "test" in a.layers


@pytest.mark.parametrize("method", ["bbknn", "mnn", "harmony", "scanorama", "combat"])  # TODO excluded "scvi" due to runtime; may be mocked in the future
def test_batch_correction(adata_with_batch, method):
    """Test if batch correction returns an anndata."""

    adata_corrected = tools.norm_correct.batch_correction(adata_with_batch, batch_key="batch", method=method)
    assert isinstance(adata_corrected, sc.AnnData)
    # assert the returned adata is a different object
    # this is a workaround to test if the original adata was modified
    assert adata_with_batch is not adata_corrected


@pytest.mark.parametrize("key", ["batch", ["batch", "batch2"]])
def test_evaluate_batch_effect(adata_with_batch, key):
    """Test if AnnData containing LISI column in .obs is returned."""
    ad = tools.norm_correct.evaluate_batch_effect(adata_with_batch, batch_key=key)

    ad_type = type(ad).__name__
    assert ad_type == "AnnData"
    assert ad.obs.columns.str.startswith("LISI_score").any()


@pytest.mark.parametrize("key", ["a", "b"])
def test_evaluate_batch_effect_keyerror(adata_with_batch, key):
    """Test evaluate_batch_effect failure."""
    with pytest.raises(KeyError, match="adata.obsm .*"):
        tools.norm_correct.evaluate_batch_effect(adata_with_batch, batch_key='batch', obsm_key=key)

    with pytest.raises(KeyError, match="adata.obs .*"):
        tools.norm_correct.evaluate_batch_effect(adata_with_batch, batch_key=key)


@pytest.mark.parametrize("key", ["batch", ["batch", "batch2"]])
def test_wrap_batch_evaluation(adata_batch_dict, key):
    """Test if DataFrame containing LISI column in .obs is returned."""
    adata_dict = tools.norm_correct.wrap_batch_evaluation(adata_batch_dict, key, inplace=False)
    assert isinstance(adata_dict, dict)
    assert isinstance(adata_dict['adata'], sc.AnnData)
