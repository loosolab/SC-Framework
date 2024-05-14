"""Test atac functions."""

import pytest
import os
import scanpy as sc
import numpy as np
import anndata as ad
import yaml

import sctoolbox.plotting as pl
import sctoolbox.tools as tools


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
    tools.norm_correct.tfidf(tfidf_x, log_tf=True, log_idf=True)
    assert str("%.3f" % tfidf_x.X[0, 0]) == "4.659"
    assert str("%.3f" % tfidf_x.X[3, 0]) == "4.770"


@pytest.mark.parametrize("method", ["tfidf", "total"])
def test_atac_norm(adata, method):
    """Test atac_norm success."""
    adata_norm = tools.norm_correct.atac_norm(adata, method=method)[method]  # return from function is a dict

    if method == "tfidf":
        assert "X_lsi" in adata_norm.obsm and "lsi" in adata_norm.uns and "LSI" in adata_norm.varm
    elif method == "total":
        assert "X_pca" in adata_norm.obsm and "pca" in adata_norm.uns and "PCs" in adata_norm.varm


def test_write_TOBIAS_config():
    """Test write_TOBIAS_config success."""

    tools.tobias.write_TOBIAS_config("tobias.yml", bams=["bam1.bam", "bam2.bam"])
    yml = yaml.full_load(open("tobias.yml"))

    assert yml["data"]["1"] == "bam1.bam"


def test_add_insertsize_fragments(adata):
    """Test if add_insertsize adds information from a fragmentsfile."""

    adata = adata.copy()
    fragments = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac_fragments.bed')
    tools.insertsize.add_insertsize(adata, fragments=fragments)

    assert "insertsize_distribution" in adata.uns
    assert "mean_insertsize" in adata.obs.columns


def test_add_insertsize_bam(adata):
    """Test if add_insertsize adds information from a bamfile."""

    adata = adata.copy()
    bam = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    tools.insertsize.add_insertsize(adata, bam=bam)

    assert "insertsize_distribution" in adata.uns
    assert "mean_insertsize" in adata.obs.columns


def test_insertsize_plotting(adata):
    """Test if insertsize plotting works."""

    adata = adata.copy()
    fragments = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac_fragments.bed')
    tools.insertsize.add_insertsize(adata, fragments=fragments)

    ax = pl.plot_insertsize(adata)

    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")
