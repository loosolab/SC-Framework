import pytest
import sctoolbox.atac
import os
import scanpy as sc
import yaml


@pytest.fixture
def adata():
    """ Fixture for an AnnData object. """
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad'))
    return adata


def test_write_TOBIAS_config():

    sctoolbox.atac.write_TOBIAS_config("tobias.yml", bams=["bam1.bam", "bam2.bam"])
    yml = yaml.full_load(open("tobias.yml"))

    assert yml["data"]["1"] == "bam1.bam"


def test_add_insertsize_fragments(adata):
    """ Test if add_insertsize adds information from a fragmentsfile """

    fragments = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac_fragments.bed')
    sctoolbox.atac.add_insertsize(adata, fragments=fragments)

    assert "insertsize_distribution" in adata.uns
    assert "mean_insertsize" in adata.obs.columns


def test_add_insertsize_bam(adata):
    """ Test if add_insertsize adds information from a bamfile """

    bam = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    sctoolbox.atac.add_insertsize(adata, fragments=bam)

    assert "insertsize_distribution" in adata.uns
    assert "mean_insertsize" in adata.obs.columns


def test_insertsize_plotting(adata):
    """ Test if insertsize plotting works """

    fragments = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac_fragments.bed')
    sctoolbox.atac.add_insertsize(adata, fragments=fragments)

    ax = sctoolbox.atac.plot_insertsize(adata)

    ax_type = type(ax).__name__
    assert ax_type == "AxesSubplot"
