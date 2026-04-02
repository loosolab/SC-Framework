"""Test plotting/highly_variable functions."""

import pytest
import sctoolbox.plotting.highly_variable as pl
import numpy as np
import matplotlib.pyplot as plt

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")


# ------------------------------ TESTS --------------------------------- #


def test_violin_HVF_distribution(adata):
    """Test violin_HVF_distribution."""
    adata_HVF = adata.copy()
    adata_HVF.var['highly_variable'] = np.random.choice([True, False], size=adata_HVF.shape[1])
    adata_HVF.var['n_cells_by_counts'] = np.random.normal(size=adata_HVF.shape[1])
    pl.violin_HVF_distribution(adata_HVF)
    assert True


def test_violin_HVF_distribution_fail(adata):
    """Test if input is invalid."""
    with pytest.raises(KeyError):
        pl.violin_HVF_distribution(adata)


def test_scatter_HVF_distribution(adata):
    """Test scatter_HVF_distribution."""
    adata_HVF = adata.copy()
    adata_HVF.var['variability_score'] = np.random.normal(size=adata_HVF.shape[1])
    adata_HVF.var['n_cells'] = np.random.normal(size=adata_HVF.shape[1])
    pl.scatter_HVF_distribution(adata_HVF)
    assert True


def test_scatter_HVF_distribution_fail(adata):
    """Test if input is invalid."""
    with pytest.raises(KeyError):
        pl.scatter_HVF_distribution(adata)
