"""Test atac utility functions."""

import pytest
import os
import scanpy as sc
import numpy as np

import sctoolbox.utils as utils
import sctoolbox.tools as tools


@pytest.fixture
def adata_atac():
    """Load atac adata."""
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')
    return sc.read_h5ad(adata_f)


@pytest.fixture
def adata_atac_emptyvar(adata_atac):
    """Create adata with empty adata.var."""
    adata = adata_atac.copy()
    adata.var = adata.var.drop(columns=adata.var.columns)
    return adata


@pytest.fixture
def adata_atac_invalid(adata_atac):
    """Create adata with invalid index."""
    adata = adata_atac.copy()
    adata.var.iloc[0, 1] = 500  # start
    adata.var.iloc[0, 2] = 100  # end
    adata.var.reset_index(inplace=True, drop=True)  # remove chromosome-start-stop index
    return adata


@pytest.fixture
def adata_rna():
    """Load rna adata."""
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    return sc.read_h5ad(adata_f)


@pytest.fixture
def bamfile():
    """Fixture for an Bamfile."""
    bamfile = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    return bamfile


def test_bam_adata_ov(adata_atac, bamfile):
    """Test bam_adata_ov success."""
    hitrate = tools.bam.bam_adata_ov(adata_atac, bamfile, cb_tag='CB')
    assert hitrate >= 0.10
