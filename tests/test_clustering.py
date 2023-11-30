"""Test clustering functions."""

import pytest
import numpy as np
import anndata
import pandas as pd
from scipy import sparse
import sctoolbox.tools as tl


def build_adata(mtx):
    """Build mock anndata object."""

    # define the number of observations (obs) and var regions
    n_obs = 30
    n_vars = 30

    obs_df = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    obs_df['cluster'] = [i for i in range(1, 11) for _ in range(3)]
    var_df = pd.DataFrame(index=[f"var_{i}" for i in range(n_vars)])
    var_df['feature'] = np.arange(1, 31, 1)

    adata = anndata.AnnData(X=mtx, obs=obs_df, var=var_df)
    adata.var["total_counts"] = adata.X.sum(axis=1)

    return adata


def unequal_mtx():
    """Build a mock mtx with unequal distribution."""

    index = np.arange(0, 31, 3)
    zero_arr = np.zeros((30, 30))

    for i in range(0, len(index) - 1):
        zero_arr[index[i]:index[i + 1], index[i]:index[i + 1]] = 1

    mtx = sparse.csr_matrix(zero_arr)

    return mtx


def equal_mtx():
    """Build a mock mtx with equal distribution."""

    ones_arr = np.ones((30, 30))
    mtx = sparse.csr_matrix(ones_arr)

    return mtx


@pytest.fixture
def equal_adata():
    """Build a mock anndata object with equally distributed features."""

    mtx = equal_mtx()
    adata = build_adata(mtx)

    return adata


@pytest.fixture
def unequal_adata():
    """Build a mock anndata object with unequally distributed features."""

    mtx = unequal_mtx()
    adata = build_adata(mtx)

    return adata


def test_gini():
    """Test gini function."""

    test_array_equal = np.full(10, 5)

    test_array_unequal = np.zeros(100)
    test_array_unequal[-1] = 100

    assert tl.gini(test_array_equal) == 0
    assert tl.gini(test_array_unequal) == 0.99


def test_calc_ragi(equal_adata, unequal_adata):
    """Test calc_ragi function."""

    _, ragi_equal = tl.calc_ragi(equal_adata, "cluster", None)
    _, ragi_unequal = tl.calc_ragi(unequal_adata, "cluster", None)

    assert ragi_equal == 0
    assert round(ragi_unequal, 2) == 0.9
