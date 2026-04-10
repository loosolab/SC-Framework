"""Tests for highly_variable.py."""

import pytest
import scanpy as sc
import sctoolbox.tools.highly_variable as hv


# ------------------------- FIXTURES -------------------------#


# TODO add precalculated qc adata to save runtime
@pytest.fixture()
def adata_atac_qc(adata_atac):
    """Add qc to ATAC anndata.

    Returns
    -------
    anndata.AnnData
        ATAC-seq AnnData object with QC metrics.
    """
    sc.pp.calculate_qc_metrics(adata_atac, inplace=True)

    return adata_atac


# ------------------------- TESTS ------------------------- #


@pytest.mark.parametrize("inplace", [True, False])
def test_annot_HVG(adata, inplace):
    """Test if 'highly_variable' column is added to adata.var."""

    sc.pp.log1p(adata)
    out = hv.annot_HVG(adata, inplace=inplace)

    if inplace:
        assert out is None
        assert "highly_variable" in adata.var.columns
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
