"""Tests for highly_variable.py."""

import pytest
import os
import scanpy as sc
import sctoolbox.tools.highly_variable as hv


# ------------------------- Fixtures -------------------------#


@pytest.fixture
def adata_rna():
    """Load rna anndata."""
    adata_f = os.path.join(os.path.dirname(__file__), '../data', 'adata.h5ad')
    return sc.read_h5ad(adata_f)


@pytest.fixture
def adata_atac():
    """Load atac anndata."""
    adata_f = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_atac.h5ad')
    return sc.read_h5ad(adata_f)


# TODO add precalculated qc adata to save runtime
@pytest.fixture(scope="module")
def adata_atac_qc():
    """Add qc to anndata."""
    adata_f = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_atac.h5ad')
    adata = sc.read_h5ad(adata_f)
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    return adata


# ------------------------- Tests ------------------------- #


@pytest.mark.parametrize("inplace", [True, False])
def test_annot_HVG(adata_rna, inplace):
    """Test if 'highly_variable' column is added to adata.var."""

    sc.pp.log1p(adata_rna)
    out = hv.annot_HVG(adata_rna, inplace=inplace)

    if inplace:
        assert out is None
        assert "highly_variable" in adata_rna.var.columns
    else:
        assert "highly_variable" in out.var.columns


@pytest.mark.parametrize("inplace", [True, False])
def test_get_variable_features(adata_atac_qc, inplace):
    """Test get_variable_features success."""
    adata = adata_atac_qc.copy()

    assert "highly_variable" not in adata.var.columns

    output = hv.get_variable_features(adata=adata,
                                      max_cells=None,
                                      min_cells=0,
                                      show=True,
                                      inplace=inplace)

    if inplace:
        assert output is None
        assert "highly_variable" in adata.var.columns
    else:
        assert "highly_variable" in output.var.columns
        assert "highly_variable" not in adata.var.columns


def test_get_variable_features_fail(adata_atac):
    """Test get_variable_features failure."""
    with pytest.raises(KeyError):
        hv.get_variable_features(adata=adata_atac)
