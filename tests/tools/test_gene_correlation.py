"""Tests for gene correlation methods."""
import pytest
import numpy as np
import pandas as pd
import os
import scanpy as sc

from sctoolbox.utils.adata import get_adata_subsets
from sctoolbox.tools.gene_correlation import correlate_conditions, correlate_ref_vs_all, compare_two_conditons


# ---------------------------- FIXTURES -------------------------------- #


@pytest.fixture
def adata():
    """Fixture for simple adata to test with."""
    h5ad = os.path.join(os.path.dirname(__file__), '..', 'data', 'adata.h5ad')
    adata = sc.read_h5ad(h5ad)

    adata.obs["condition"] = np.random.choice(["C1", "C2"], size=adata.shape[0])

    # set gene names as index instead of ensemble ids
    adata.var.reset_index(inplace=True)
    adata.var['gene'] = adata.var['gene'].astype('str')
    adata.var.set_index('gene', inplace=True)
    adata.var_names_make_unique()

    return adata


# ------------------------------ TESTS --------------------------------- #


@pytest.mark.parametrize("gene, save", [("Xkr4", None),
                                        ("Xkr4", "output.png")])
def test_correlate_ref_vs_all(adata, gene, save):
    """Test if correlation between a reference gene to other genes is calculated."""
    results = correlate_ref_vs_all(adata, gene, save=save)

    # Test if dataframe is returned
    assert isinstance(results, pd.DataFrame)

    # Test of columns are matching
    assert list(results.columns) == ['correlation',
                                     'p-value',
                                     'padj',
                                     'correlation_sign',
                                     'correlation_strength',
                                     'reject_0?']


@pytest.mark.parametrize("gene", ["Invalid Gene"])
def test_correlate_ref_vs_all_invalid(adata, gene):
    """Test if error is thrown if given gene is not in dataset."""
    with pytest.raises(Exception):
        correlate_ref_vs_all(adata, gene)


def test_compare_two_conditons(adata):
    """Test if two conditions can be compared wihtout the wrapper function."""

    adata_subsets = get_adata_subsets(adata, groupby="condition")

    corr_A_df = correlate_ref_vs_all(adata_subsets["C1"], "Xkr4")
    corr_B_df = correlate_ref_vs_all(adata_subsets["C2"], "Xkr4")

    comparison = compare_two_conditons(corr_A_df, corr_B_df,
                                       adata_subsets["C1"].shape[1],
                                       adata_subsets["C2"].shape[1])

    # Test if dataframe is returned
    assert isinstance(comparison, pd.DataFrame)

    # Test if output columns are correct
    assert list(comparison.columns) == ['correlation_A',
                                        'p-value_A',
                                        'padj_A',
                                        'correlation_sign_A',
                                        'correlation_strength_A',
                                        'reject_0?_A',
                                        'correlation_B',
                                        'p-value_B',
                                        'padj_B',
                                        'correlation_sign_B',
                                        'correlation_strength_B',
                                        'reject_0?_B',
                                        'comparison z-score',
                                        'comparison p-value']


def test_correlate_conditions(adata):
    """Test wrapper for correlation and comparison."""
    comparison = correlate_conditions(adata, "Xkr4", "condition", "C1", "C2")

    # Test if dataframe is returned
    assert isinstance(comparison, pd.DataFrame)

    # Test if output columns are correct
    assert list(comparison.columns) == ['correlation_A',
                                        'p-value_A',
                                        'padj_A',
                                        'correlation_sign_A',
                                        'correlation_strength_A',
                                        'reject_0?_A',
                                        'correlation_B',
                                        'p-value_B',
                                        'padj_B',
                                        'correlation_sign_B',
                                        'correlation_strength_B',
                                        'reject_0?_B',
                                        'comparison z-score',
                                        'comparison p-value']


def test_invalid_condition_correlate_conditions(adata):
    """Test if error is raised when condition is invalid."""
    with pytest.raises(Exception):
        correlate_conditions(adata, "Xkr4", "condition", "Invalid Condition", "C2")
