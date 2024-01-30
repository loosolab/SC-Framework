"""Test atac utility functions."""

import pytest
import os
import scanpy as sc
import numpy as np

import sctoolbox.utilities as utils
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


@pytest.mark.parametrize("fixture, expected", [("adata_atac", True),  # expects var tables to be unchanged
                                               ("adata_atac_emptyvar", False),  # expects var tables to be changed
                                               ("adata_rna", ValueError),  # expects a valueerror due to missing columns
                                               ("adata_atac_invalid", ValueError)])  # expects a valueerror due to format of columns
def test_format_adata_var(fixture, expected, request):
    """Test whether adata regions can be formatted (or raise an error if not)."""

    adata_orig = request.getfixturevalue(fixture)  # fix for using fixtures in parametrize
    adata_cp = adata_orig.copy()  # make a copy to avoid changing the fixture
    if isinstance(expected, type):
        with pytest.raises(expected):
            utils.format_adata_var(adata_cp, coordinate_columns=["chr", "start", "stop"])

    else:
        utils.format_adata_var(adata_cp, coordinate_columns=["chr", "start", "stop"], columns_added=["chr", "start", "end"])

        assert np.array_equal(adata_orig.var.values, adata_cp.var.values) == expected  # check if the original adata was changed or not


def test_bam_adata_ov(adata_atac, bamfile):
    """Test bam_adata_ov success."""
    hitrate = tools.bam_adata_ov(adata_atac, bamfile, cb_tag='CB')
    assert hitrate >= 0.10