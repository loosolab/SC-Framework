import pytest
import sctoolbox.atac_utils as atac_utils
import os
import scanpy as sc
import numpy as np


@pytest.fixture
def adata_atac():
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')
    return sc.read_h5ad(adata_f)


@pytest.fixture
def adata_atac_emptyvar(adata_atac):
    adata = adata_atac.copy()
    adata.var = adata.var.drop(columns=adata.var.columns)
    return adata


@pytest.fixture
def adata_atac_invalid(adata_atac):
    adata = adata_atac.copy()
    adata.var.iloc[0, 1] = 500  # start
    adata.var.iloc[0, 2] = 100  # end
    adata.var.reset_index(inplace=True, drop=True)  # remove chromosome-start-stop index
    return adata


@pytest.fixture
def adata_rna():
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    return sc.read_h5ad(adata_f)


@pytest.fixture
def bamfile():
    """ Fixture for an Bamfile.  """
    bamfile = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    return bamfile


@pytest.mark.parametrize("fixture, expected", [("adata_atac", True),  # expects var tables to be unchanged
                                               ("adata_atac_emptyvar", False),  # expects var tables to be changed
                                               ("adata_rna", ValueError),  # expects a valueerror due to missing columns
                                               ("adata_atac_invalid", ValueError)])  # expects a valueerror due to format of columns
def test_format_adata_var(fixture, expected, request):
    """ Test whether adata regions can be formatted (or raise an error if not)"""

    adata_orig = request.getfixturevalue(fixture)  # fix for using fixtures in parametrize
    adata_cp = adata_orig.copy()  # make a copy to avoid changing the fixture
    if type(expected) == type:
        with pytest.raises(expected):
            atac_utils.format_adata_var(adata_cp, coordinate_columns=["chr", "start", "stop"])

    else:
        atac_utils.format_adata_var(adata_cp, coordinate_columns=["chr", "start", "stop"], columns_added=["chr", "start", "end"])

        assert np.array_equal(adata_orig.var.values, adata_cp.var.values) == expected  # check if the original adata was changed or not


def test_bam_adata_ov(adata, bamfile):
    hitrate = atac_utils.bam_adata_ov(adata_atac, bamfile, cb_col='CB')
    assert hitrate >= 0.10
