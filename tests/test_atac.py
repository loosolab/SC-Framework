"""Test atac functions."""

import pytest
import sctoolbox.atac
import os
import scanpy as sc
import numpy as np
import anndata as ad

import sctoolbox.plotting as pl


@pytest.fixture
def adata():
    """Fixture for an AnnData object."""
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad'))
    return adata


# adapted from muon package
@pytest.fixture
def tfidf_x():
    """Create anndata with random expression."""
    np.random.seed(2020)
    x = np.abs(np.random.normal(size=(4, 5)))
    adata_X = ad.AnnData(x)
    return adata_X


# adapted from muon package
def test_tfidf(tfidf_x):
    """Test tfidt success."""
    sctoolbox.atac.tfidf(tfidf_x, log_tf=True, log_idf=True)
    assert str("%.3f" % tfidf_x.X[0, 0]) == "4.659"
    assert str("%.3f" % tfidf_x.X[3, 0]) == "4.770"


def test_lsi(adata):
    """Test lsi success."""
    sctoolbox.atac.tfidf(adata)
    sctoolbox.atac.lsi(adata)
    assert "X_lsi" in adata.obsm and "lsi" in adata.uns and "LSI" in adata.varm


@pytest.mark.parametrize("method", ["tfidf", "total"])
def test_atac_norm(adata, method):
    """Test atac_norm success."""
    adata_norm = sctoolbox.atac.atac_norm(adata, method=method)[method]  # return from function is a dict

    if method == "tfidf":
        assert "X_lsi" in adata_norm.obsm and "lsi" in adata_norm.uns and "LSI" in adata_norm.varm
    elif method == "total":
        assert "X_pca" in adata_norm.obsm and "pca" in adata_norm.uns and "PCs" in adata_norm.varm


def test_add_insertsize_fragments(adata):
    """Test if add_insertsize adds information from a fragmentsfile."""

    adata = adata.copy()
    fragments = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac_fragments.bed')
    sctoolbox.atac.add_insertsize(adata, fragments=fragments)

    assert "insertsize_distribution" in adata.uns
    assert "mean_insertsize" in adata.obs.columns


def test_add_insertsize_bam(adata):
    """Test if add_insertsize adds information from a bamfile."""

    adata = adata.copy()
    bam = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    sctoolbox.atac.add_insertsize(adata, bam=bam)

    assert "insertsize_distribution" in adata.uns
    assert "mean_insertsize" in adata.obs.columns


def test_insertsize_plotting(adata):
    """Test if insertsize plotting works."""

    adata = adata.copy()
    fragments = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac_fragments.bed')
    sctoolbox.atac.add_insertsize(adata, fragments=fragments)

    ax = pl.plot_insertsize(adata)

    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")
