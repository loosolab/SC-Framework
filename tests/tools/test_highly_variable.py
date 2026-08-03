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
def test_annot_HVG(adata_fun_scope, inplace):
    """Test if 'highly_variable' column is added to adata.var."""

    out = hv.annot_HVG(adata_fun_scope, inplace=inplace)

    if inplace:
        assert out is None
        assert "highly_variable" in adata_fun_scope.var.columns
    else:
        assert "highly_variable" in out.var.columns


@pytest.mark.parametrize("inplace", [True, False])
# subsample_target=100 is well below the fixture's feature count, so step > 0 and
# the subsampling branch is exercised; 10000 leaves step == 0 (branch skipped).
@pytest.mark.parametrize("subsample_target", [10000, 100])
def test_get_variable_features(adata_atac_qc, inplace, subsample_target):
    """Test get_variable_features success, including the subsampling branch."""
    adata = adata_atac_qc.copy()

    assert "highly_variable" not in adata.var.columns
    assert adata.n_vars > 100  # guarantees step > 0 for subsample_target=100

    output = hv.get_variable_features(adata=adata,
                                      max_cells=None,
                                      min_cells=0,
                                      subsample_target=subsample_target,
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
