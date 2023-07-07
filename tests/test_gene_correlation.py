import pytest
import numpy as np
import os
import scanpy as sc

from sctoolbox.tools.gene_correlation import * 

# ---------------------------- FIXTURES -------------------------------- #

@pytest.fixture
def adata():

    h5ad = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    adata = sc.read_h5ad(h5ad)

    adata.obs["condition"] = np.random.choice(["C1", "C2"], size=adata.shape[0])

    # set gene names as index instead of ensemble ids
    adata.var.reset_index(inplace=True)
    adata.var['gene'] = adata.var['gene'].astype('str')
    adata.var.set_index('gene', inplace=True)
    adata.var_names_make_unique()

    return adata

# ------------------------------ TESTS --------------------------------- #

def test_correlate_ref_vs_all(adata):
    """ Test if correlation between a reference gene to other genes is calculated. """
    results = correlate_ref_vs_all(adata, "Xkr4")

    # Test if dataframe is returned
    assert isinstance(results, pd.DataFrame)

    # Test of columns are matching
    assert list(r.columns) == ['correlation',
                               'p-value',
                               'padj',
                               'correlation_sign',
                               'correlation_strength',
                               'reject_0?']


def test_compare_two_conditons(adata):
    """ Test if two conditions can be compared wihtout the wrapper function. """

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


def test_correlate_and_compare_two_conditions(adata):
    """ Test wrapper for correlation and comparison """
    comparison = correlate_and_compare_two_conditions(adata, "Xkr4", "condition", "C1", "C2")

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
