"""Test plotting/highly_variable functions."""

import pytest
import sctoolbox.plotting.highly_variable as pl
import numpy as np
import matplotlib.pyplot as plt

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")


# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture
def adata_hvf(adata_fun_scope):
    """AnnData with columns required for HVF distribution plots.

    Returns
    -------
    anndata.AnnData
        AnnData with highly_variable, n_cells_by_counts, variability_score, and n_cells columns.
    """
    adata_fun_scope.var['highly_variable'] = np.random.choice([True, False], size=adata_fun_scope.shape[1])
    adata_fun_scope.var['n_cells_by_counts'] = np.random.normal(size=adata_fun_scope.shape[1])
    adata_fun_scope.var['variability_score'] = np.random.normal(size=adata_fun_scope.shape[1])
    adata_fun_scope.var['n_cells'] = np.random.normal(size=adata_fun_scope.shape[1])

    return adata_fun_scope


# ------------------------------ TESTS --------------------------------- #


def test_violin_HVF_distribution(adata_hvf):
    """Test violin_HVF_distribution."""
    pl.violin_HVF_distribution(adata_hvf)


def test_violin_HVF_distribution_fail(adata):
    """Test if input is invalid."""
    with pytest.raises(KeyError):
        pl.violin_HVF_distribution(adata)


def test_scatter_HVF_distribution(adata_hvf):
    """Test scatter_HVF_distribution."""
    pl.scatter_HVF_distribution(adata_hvf)


def test_scatter_HVF_distribution_fail(adata):
    """Test if input is invalid."""
    with pytest.raises(KeyError):
        pl.scatter_HVF_distribution(adata)
