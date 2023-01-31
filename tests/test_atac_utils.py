import pytest
import sctoolbox.atac_utils
import os
import scanpy as sc


@pytest.fixture
def adata():
    """ Fixture for an AnnData object. """
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad'))
    return adata


@pytest.fixture
def bamfile():
    """ Fixture for an Bamfile.  """
    bamfile = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    return bamfile


def test_bam_adata_ov(adata, bamfile):
    hitrate = sctoolbox.atac_utils.bam_adata_ov(adata, bamfile, cb_col='CB')
    assert hitrate >= 0.10