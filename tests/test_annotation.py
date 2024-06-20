"""Test functions related to annotation."""

import pytest
import sctoolbox.tools as anno
import scanpy as sc
import os


# ------------------------- Fixtures -------------------------#


@pytest.fixture
def adata_atac():
    """Load atac anndata."""
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')
    return sc.read_h5ad(adata_f)


# TODO add precalculated qc adata to save runtime
@pytest.fixture(scope="module")
def adata_atac_qc():
    """Add qc to anndata."""
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')
    adata = sc.read_h5ad(adata_f)
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    return adata


@pytest.fixture
def adata_atac_emptyvar(adata_atac):
    """Create anndata with empty adata.var."""
    adata = adata_atac.copy()
    adata.var = adata.var.drop(columns=adata.var.columns)
    return adata


@pytest.fixture
def adata_atac_invalid(adata_atac):
    """Create adata with invalid adata.var index."""
    adata = adata_atac.copy()
    adata.var.iloc[0, 1] = 500  # start
    adata.var.iloc[0, 2] = 100  # end
    adata.var.reset_index(inplace=True, drop=True)  # remove chromosome-start-stop index
    return adata


@pytest.fixture
def adata_rna():
    """Load rna anndata."""
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    return sc.read_h5ad(adata_f)


# ------------------------- Tests ------------------------- #


@pytest.mark.parametrize("inplace", [True, False])
def test_annot_HVG(adata_rna, inplace):
    """Test if 'highly_variable' column is added to adata.var."""

    sc.pp.log1p(adata_rna)
    out = anno.annot_HVG(adata_rna, inplace=inplace)

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

    output = anno.get_variable_features(adata=adata,
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
        anno.get_variable_features(adata=adata_atac)
