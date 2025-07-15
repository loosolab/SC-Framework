"""Test receptor-ligand functions."""

import pytest
import os
import pandas as pd
import numpy as np
import scanpy as sc
import random
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import networkx as nx
from unittest.mock import patch, MagicMock
import warnings

import sctoolbox.tools.receptor_ligand as rl


# ------------------------------ FIXTURES -------------------------------- #


# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")


@pytest.fixture
def adata():
    """Load and returns an anndata object."""
    f = os.path.join(os.path.dirname(__file__), '..', 'data', "adata.h5ad")

    obj = sc.read_h5ad(f)

    # add cluster column
    def repeat_items(list, count):
        """
        Repeat list until size reached.

        https://stackoverflow.com/a/54864336/19870975
        """
        return list * (count // len(list)) + list[:(count % len(list))]

    obj.obs["cluster"] = repeat_items([f"cluster {i}" for i in range(10)], len(obj))

    return obj


@pytest.fixture
def db_file():
    """Path to receptor-ligand database."""
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'receptor-ligand', 'mouse_lr_pair.tsv')


@pytest.fixture
def adata_db(adata, db_file):
    """Add interaction db to adata."""
    return rl.download_db(adata=adata,
                          db_path=db_file,
                          ligand_column='ligand_gene_symbol',
                          receptor_column='receptor_gene_symbol',
                          inplace=False,
                          overwrite=False)


@pytest.fixture
def adata_inter(adata_db):
    """Add interaction scores to adata."""
    obj = adata_db.copy()

    # replace with random interactions
    obj.uns["receptor-ligand"]["database"] = pd.DataFrame({
        "ligand_gene_symbol": random.sample(obj.var["gene"].tolist(), k=len(obj.var) // 2),
        "receptor_gene_symbol": random.sample(obj.var["gene"].tolist(), k=len(obj.var) // 2)
    })

    return rl.calculate_interaction_table(adata=obj,
                                          cluster_column="cluster",
                                          gene_index="gene",
                                          normalize=1000,
                                          inplace=False,
                                          overwrite=False)

# ------------------------------ FIXTURES FOR DIFFERENCE TESTING -------------------------------- #


@pytest.fixture
def adata_with_conditions(adata_inter):
    """Add condition information to existing adata_inter fixture."""
    obj = adata_inter.copy()

    # Add condition information
    n_obs = len(obj)
    obj.obs['condition'] = ['control'] * (n_obs // 2) + ['treatment'] * (n_obs - n_obs // 2)
    obj.obs['batch'] = ['batch1'] * (n_obs // 2) + ['batch2'] * (n_obs - n_obs // 2)
    obj.obs['timepoint'] = ['day0'] * (n_obs // 3) + ['day3'] * (n_obs // 3) + ['day7'] * (n_obs - 2 * (n_obs // 3))

    return obj


@pytest.fixture
def diff_results():
    """Create mock differential analysis results."""
    differences = pd.DataFrame({
        'receptor_gene': ['gene1', 'gene2', 'gene3'],
        'ligand_gene': ['gene10', 'gene20', 'gene30'],
        'receptor_cluster': ['cluster 0', 'cluster 1', 'cluster 2'],
        'ligand_cluster': ['cluster 1', 'cluster 2', 'cluster 0'],
        'interaction_score_treatment': [0.8, 0.6, 0.7],
        'interaction_score_control': [0.5, 0.3, 0.9],
        'quantile_rank_treatment': [0.9, 0.6, 0.7],
        'quantile_rank_control': [0.5, 0.3, 0.8],
        'rank_diff_control_vs_treatment': [-0.4, -0.3, 0.1],
        'abs_diff_control_vs_treatment': [0.4, 0.3, 0.1]
    })

    return {
        'condition': {
            'control_vs_treatment': {
                'differences': differences
            }
        }
    }


@pytest.fixture
def adata_with_diff_results(adata_with_conditions, diff_results):
    """Create AnnData with mock differential results."""
    obj = adata_with_conditions.copy()

    # Add the mock results
    obj.uns.setdefault('sctoolbox', {}).setdefault('receptor-ligand', {}).setdefault('condition-differences', {})
    obj.uns['sctoolbox']['receptor-ligand']['condition-differences'] = diff_results

    return obj


# ------------------------------ TESTS -------------------------------- #


# ----- test setup functions ----- #

@pytest.mark.parametrize('db_path,ligand_column,receptor_column',
                         [(None, 'ligand_gene_symbol', 'receptor_gene_symbol'),
                          ('consensus', 'ligand', 'receptor')]
                         )
def test_download_db(adata, db_path, ligand_column, receptor_column, db_file):
    """Assert rl database is added into anndata."""
    obj = adata.copy()

    # adata does not have database
    assert "receptor-ligand" not in obj.uns

    # add database
    rl.download_db(adata=obj,
                   db_path=db_path if db_path else db_file,
                   ligand_column=ligand_column,
                   receptor_column=receptor_column,
                   inplace=True,
                   overwrite=False)

    # adata contains database
    assert "receptor-ligand" in obj.uns
    assert "database" in obj.uns["receptor-ligand"]


@pytest.mark.parametrize('db_path,ligand_column,receptor_column',
                         [(None, 'INVALID', 'receptor_gene_symbol'),
                          (None, 'ligand_gene_symbol', 'INVALID'),
                          ('INVALID', 'ligand', 'receptor')]
                         )
def test_download_db_fail(adata, db_path, ligand_column, receptor_column, db_file):
    """Assert ValueErrors."""
    obj = adata.copy()

    # adata does not have database
    assert "receptor-ligand" not in obj.uns

    with pytest.raises(ValueError):
        rl.download_db(adata=obj,
                       db_path=db_path if db_path else db_file,
                       ligand_column=ligand_column,
                       receptor_column=receptor_column,
                       inplace=True,
                       overwrite=False)

    # adata does not have database
    assert "receptor-ligand" not in obj.uns


def test_interaction_table(adata_db):
    """Assert interaction are computed/ added into anndata."""
    obj = adata_db.copy()

    # adata has db but no scores
    assert "receptor-ligand" in obj.uns
    assert "database" in obj.uns["receptor-ligand"]
    assert "interactions" not in obj.uns["receptor-ligand"]

    # compute rl scores
    with pytest.raises(Exception):
        # raises error because no interactions are found
        rl.calculate_interaction_table(adata=obj,
                                       cluster_column="cluster",
                                       gene_index="gene",
                                       normalize=1000,
                                       inplace=True,
                                       overwrite=False)

    # replace with random interactions
    obj.uns["receptor-ligand"]["database"] = pd.DataFrame({
        "ligand_gene_symbol": random.sample(obj.var["gene"].tolist(), k=len(obj.var) // 2),
        "receptor_gene_symbol": random.sample(obj.var["gene"].tolist(), k=len(obj.var) // 2)
    })

    rl.calculate_interaction_table(adata=obj,
                                   cluster_column="cluster",
                                   gene_index="gene",
                                   normalize=1000,
                                   inplace=True,
                                   overwrite=False)

    # adata contains scores
    assert "interactions" in obj.uns["receptor-ligand"]


# ----- test helpers ----- #

def test_get_interactions(adata_inter):
    """Assert that interactions can be received."""
    interactions_table = rl.get_interactions(adata_inter)

    # output is a pandas table
    assert isinstance(interactions_table, pd.DataFrame)


def test_check_interactions(adata, adata_db, adata_inter):
    """Assert that interaction test is properly checked."""
    # raise error without rl info
    with pytest.raises(ValueError):
        rl._check_interactions(adata)

    # raise error with incomplete rl info
    with pytest.raises(ValueError):
        rl._check_interactions(adata_db)

    # accept
    rl._check_interactions(adata_inter)


# ----- test plotting ----- #

def test_violin(adata_inter):
    """Violin plot is functional."""
    plot = rl.interaction_violin_plot(adata_inter,
                                      min_perc=0,
                                      save=None,
                                      figsize=(5, 30),
                                      dpi=100)

    assert isinstance(plot, np.ndarray)


def test_hairball(adata_inter):
    """Hairball network plot is functional."""
    plot = rl.hairball(adata_inter,
                       min_perc=0,
                       interaction_score=0,
                       interaction_perc=90,
                       save=None,
                       title=None,
                       color_min=0,
                       color_max=None,
                       restrict_to=[],
                       show_count=True)

    assert isinstance(plot, np.ndarray)


def test_cyclone(adata_inter):
    """Cyclone network plot is functional."""
    plot = rl.cyclone(adata=adata_inter,
                      min_perc=70,
                      interaction_score=0,
                      directional=True,
                      sector_size_is_cluster_size=True,
                      show_genes=True,
                      title="Test Title")

    assert isinstance(plot, Figure)


def test_connectionPlot(adata_inter):
    """Test if connectionPlot is working."""
    plot = rl.connectionPlot(adata=adata_inter,
                             restrict_to=None,
                             figsize=(5, 10),
                             dpi=100,
                             connection_alpha="interaction_score",
                             save=None,
                             title=None,
                             receptor_cluster_col="receptor_cluster",
                             receptor_col="receptor_gene",
                             receptor_hue="receptor_score",
                             receptor_size="receptor_percent",
                             ligand_cluster_col="ligand_cluster",
                             ligand_col="ligand_gene",
                             ligand_hue="ligand_score",
                             ligand_size="ligand_percent",
                             filter="receptor_score > 0 & ligand_score > 0 & interaction_score > 0",
                             lw_multiplier=2,
                             wspace=0.4,
                             line_colors="rainbow")

    assert isinstance(plot, np.ndarray)


# ----- Tests for receptor-ligand differences analysis functions. ----- #


@pytest.mark.parametrize(
    "condition_values,condition_columns,expected_success",
    [
        # Basic condition filtering - success case
        (['control'], ['condition'], True),
        # Non-existent condition - failure case
        (['non_existent'], ['condition'], False),
    ]
)
def test_filter_anndata(
    adata_with_conditions,
    condition_values,
    condition_columns,
    expected_success
):
    """Test filtering AnnData based on conditions."""
    with warnings.catch_warnings(record=True):
        # Filter data
        result = rl._filter_anndata(
            adata=adata_with_conditions,
            condition_values=condition_values,
            condition_columns=condition_columns,
            cluster_column='cluster'
        )

        if expected_success:
            assert result is not None
            # Check condition filtering
            for col, val in zip(condition_columns, condition_values):
                assert all(result.obs[col] == val)
        else:
            assert result is None


def test_filter_anndata_mismatch_condition_lengths(adata_with_conditions):
    """Test error when condition values and columns don't match."""
    with pytest.raises(ValueError) as excinfo:
        rl._filter_anndata(
            adata=adata_with_conditions,
            condition_values=['control', 'treatment'],
            condition_columns=['condition'],
            cluster_column='cluster'
        )

    assert "Expected" in str(excinfo.value)


@pytest.mark.parametrize(
    "cluster_filter,gene_filter,should_succeed",
    [
        # Success cases
        (['cluster 0', 'cluster 1'], None, True),
        (None, lambda adata: adata.var['gene'].iloc[:5].tolist(), True),
        (['cluster 0'], lambda adata: adata.var['gene'].iloc[:5].tolist(), True),
        # Failure cases
        (['non_existent_cluster'], None, False),
        (None, lambda _: ['non_existent_gene'], False),
        ([], None, False),
        # Edge cases
        (['cluster 0'], None, True),
        (None, lambda adata: [adata.var['gene'].iloc[0]], True),
    ]
)
def test_filter_anndata_with_filters(
    adata_with_conditions, cluster_filter, gene_filter, should_succeed
):
    """Test filtering AnnData with various cluster and gene filters."""
    # Resolve callable gene filters
    if callable(gene_filter):
        gene_filter = gene_filter(adata_with_conditions)

    with warnings.catch_warnings(record=True):
        # Filter data with provided filters
        result = rl._filter_anndata(
            adata=adata_with_conditions,
            condition_values=['control'],
            condition_columns=['condition'],
            cluster_column='cluster',
            cluster_filter=cluster_filter,
            gene_column='gene',
            gene_filter=gene_filter
        )

        if should_succeed:
            assert result is not None
            if cluster_filter:
                for cluster in cluster_filter:
                    assert cluster in result.obs['cluster'].unique()
            if gene_filter:
                for gene in gene_filter:
                    assert gene in result.var['gene'].tolist()
        else:
            assert result is None


@pytest.mark.parametrize(
    "condition_a,condition_b",
    [
        ('treatment', 'control'),
        ('day0', 'day7'),
        ('group1', 'group2')
    ]
)
def test_calculate_condition_difference_basic(adata_with_conditions, condition_a, condition_b):
    """Test basic functionality of _calculate_condition_difference."""
    # Extract two copies of the data to represent different conditions
    adata_a = adata_with_conditions.copy()
    adata_b = adata_with_conditions.copy()

    # Create a simple mock of the interaction data
    interactions_table = adata_with_conditions.uns["receptor-ligand"]["interactions"].copy()

    # Slightly modify interaction scores for condition B to ensure differences
    interactions_b = interactions_table.copy()
    interactions_b['interaction_score'] = interactions_b['interaction_score'] * 1.2

    # Mock get_interactions to return our test data
    with patch('sctoolbox.tools.receptor_ligand.get_interactions') as mock_get:
        mock_get.side_effect = [interactions_table, interactions_b]

        # Call the function
        result = rl._calculate_condition_difference(
            adata_a=adata_a,
            adata_b=adata_b,
            condition_a_name=condition_a,
            condition_b_name=condition_b
        )

        # Verify basic structure of result
        assert isinstance(result, pd.DataFrame)
        assert f'rank_diff_{condition_b}_vs_{condition_a}' in result.columns
        assert f'abs_diff_{condition_b}_vs_{condition_a}' in result.columns

        # Verify key columns are preserved
        assert all(col in result.columns for col in ['receptor_gene', 'ligand_gene',
                                                     'receptor_cluster', 'ligand_cluster'])


def test_calculate_condition_difference_empty_data(adata_with_conditions):
    """Test behavior with empty interaction data."""
    # Create two copies of the data
    adata_a = adata_with_conditions.copy()
    adata_b = adata_with_conditions.copy()

    # Create an empty interactions DataFrame with the correct structure
    empty_df = pd.DataFrame({
        'receptor_gene': [], 'ligand_gene': [],
        'receptor_cluster': [], 'ligand_cluster': [],
        'interaction_score': []
    })

    # Mock get_interactions to return empty data
    with patch('sctoolbox.tools.receptor_ligand.get_interactions') as mock_get:
        mock_get.side_effect = [empty_df, empty_df]

        # Call the function
        result = rl._calculate_condition_difference(
            adata_a=adata_a,
            adata_b=adata_b,
            condition_a_name='treatment',
            condition_b_name='control'
        )

        # Verify result is empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


def test_calculate_condition_difference_disjoint_data(adata_with_conditions):
    """Test behavior when conditions have completely different interactions."""
    # Create two copies of the data
    adata_a = adata_with_conditions.copy()
    adata_b = adata_with_conditions.copy()

    # Get the interaction table
    interactions_table = adata_with_conditions.uns["receptor-ligand"]["interactions"].copy()

    # Create a completely different interaction table for condition B
    interactions_b = interactions_table.copy()
    interactions_b['receptor_gene'] = interactions_b['receptor_gene'] + '_different'
    interactions_b['ligand_gene'] = interactions_b['ligand_gene'] + '_different'

    # Mock get_interactions to return disjoint data
    with patch('sctoolbox.tools.receptor_ligand.get_interactions') as mock_get:
        mock_get.side_effect = [interactions_table, interactions_b]

        # Call the function
        result = rl._calculate_condition_difference(
            adata_a=adata_a,
            adata_b=adata_b,
            condition_a_name='treatment',
            condition_b_name='control'
        )

        # Verify result is empty DataFrame (no common interactions)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


@pytest.mark.parametrize(
    "min_perc,interaction_score,interaction_perc",
    [
        (10, None, None),  # Only min_perc
        (None, 0.5, None),  # Only interaction_score
        (None, None, 75),   # Only interaction_perc
        (5, 0.3, None),     # Both min_perc and interaction_score
        (5, None, 80),      # Both min_perc and interaction_perc
        (None, None, None)  # No filtering
    ]
)
def test_calculate_condition_difference_filters(
    adata_with_conditions, min_perc, interaction_score, interaction_perc
):
    """Test behavior with different filtering parameters."""
    # Create two copies of the data
    adata_a = adata_with_conditions.copy()
    adata_b = adata_with_conditions.copy()

    # Get the interaction table
    interactions_table = adata_with_conditions.uns["receptor-ligand"]["interactions"].copy()

    # Mock get_interactions to return our test data
    with patch('sctoolbox.tools.receptor_ligand.get_interactions') as mock_get:
        mock_get.return_value = interactions_table

        # Call the function with filtering parameters
        rl._calculate_condition_difference(
            adata_a=adata_a,
            adata_b=adata_b,
            condition_a_name='treatment',
            condition_b_name='control',
            min_perc=min_perc,
            interaction_score=interaction_score,
            interaction_perc=interaction_perc
        )

        # Verify get_interactions was called with correct parameters
        assert mock_get.call_count == 2
        for call in mock_get.call_args_list:
            assert call.kwargs['min_perc'] == min_perc
            assert call.kwargs['interaction_score'] == interaction_score
            assert call.kwargs['interaction_perc'] == interaction_perc


def test_calculate_condition_difference_ranking(adata_inter):
    """Test the ranking calculation is performed correctly."""
    # Create two copies of the data
    adata_a = adata_inter.copy()
    adata_b = adata_inter.copy()

    # Get first few interactions from the existing data
    interactions = adata_inter.uns["receptor-ligand"]["interactions"]
    if len(interactions) >= 3:
        subset = interactions.iloc[:3].copy()
    else:
        # Create synthetic data if not enough rows
        subset = pd.DataFrame({
            'receptor_gene': ['geneA', 'geneB', 'geneC'],
            'ligand_gene': ['geneX', 'geneY', 'geneZ'],
            'receptor_cluster': ['cluster1', 'cluster1', 'cluster1'],
            'ligand_cluster': ['cluster2', 'cluster2', 'cluster2'],
            'interaction_score': [0.1, 0.2, 0.3]
        })

    # Set predictable scores for condition A (ascending)
    interactions_a = subset.copy()
    interactions_a['interaction_score'] = [0.1, 0.2, 0.3]

    # Set opposite scores for condition B (descending)
    interactions_b = subset.copy()
    interactions_b['interaction_score'] = [0.3, 0.2, 0.1]

    with patch('sctoolbox.tools.receptor_ligand.get_interactions') as mock_get:
        mock_get.side_effect = [interactions_a, interactions_b]

        # Call the function
        result = rl._calculate_condition_difference(
            adata_a=adata_a,
            adata_b=adata_b,
            condition_a_name='a',
            condition_b_name='b'
        )

        # Sort by the first key column for consistent ordering
        result = result.sort_values('interaction_score_a').reset_index(drop=True)

        # Verify ranking logic
        # The lowest score in 'a' should have the lowest rank, and highest in 'b'
        # The highest score in 'a' should have the highest rank, and lowest in 'b'
        assert result.loc[0, 'quantile_rank_a'] < result.loc[2, 'quantile_rank_a']
        assert result.loc[0, 'quantile_rank_b'] > result.loc[2, 'quantile_rank_b']

        # Verify difference calculation
        for i in range(len(result)):
            expected_diff = result.loc[i, 'quantile_rank_b'] - result.loc[i, 'quantile_rank_a']
            assert result.loc[i, 'rank_diff_b_vs_a'] == expected_diff
            assert result.loc[i, 'abs_diff_b_vs_a'] == abs(expected_diff)


def test_calculate_condition_difference_error_handling(adata_with_conditions):
    """Test error handling when get_interactions raises an exception."""
    # Create two copies of the data
    adata_a = adata_with_conditions.copy()
    adata_b = adata_with_conditions.copy()

    # Mock get_interactions to raise an error
    with patch('sctoolbox.tools.receptor_ligand.get_interactions') as mock_get:
        mock_get.side_effect = ValueError("Test error")

        # Call the function and expect it to propagate the error
        with pytest.raises(ValueError) as excinfo:
            rl._calculate_condition_difference(
                adata_a=adata_a,
                adata_b=adata_b,
                condition_a_name='treatment',
                condition_b_name='control'
            )

        assert "Test error" in str(excinfo.value)


@pytest.mark.parametrize(
    "condition_values,condition_columns",
    [
        # Case: Empty condition values list
        ([], []),
        # Case: All None values
        ([None], ['cluster']),
        # Case: Multiple None values
        ([None, None], ['cluster', 'nonexistent']),
    ]
)
def test_filter_anndata_empty_valid_pairs(adata, condition_values, condition_columns):
    """Test specifically for: valid_pairs = [(col, val) for col, val in condition_pairs if val is not None]."""
    result = rl._filter_anndata(
        adata=adata,
        condition_values=condition_values,
        condition_columns=condition_columns,
        cluster_column='cluster'
    )

    # This directly tests the "if not valid_pairs: return None" branch
    assert result is None


@pytest.mark.parametrize(
    "gene_column",
    [
        'nonexistent_column',
        '',  # Empty string
    ]
)
def test_filter_anndata_invalid_gene_column(adata, gene_column):
    """Test specifically for: if gene_column is not None and gene_column not in filtered.var.columns."""
    with pytest.raises(ValueError) as excinfo:
        rl._filter_anndata(
            adata=adata,
            condition_values=['cluster 0'],
            condition_columns=['cluster'],
            cluster_column='cluster',
            gene_column=gene_column,
            gene_filter=['any_gene']
        )

    # Verify the error message
    assert "not available in adata.var.columns" in str(excinfo.value)


def test_filter_anndata_gene_mask_zero_sum(adata):
    """Test specifically for: if gene_mask.sum() == 0."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Using completely non-existent genes guarantees gene_mask.sum() == 0
        result = rl._filter_anndata(
            adata=adata,
            condition_values=['cluster 0'],
            condition_columns=['cluster'],
            cluster_column='cluster',
            gene_column='gene',
            gene_filter=['nonexistent_gene1', 'nonexistent_gene2']
        )

        # Verify the function returns None
        assert result is None

        # Check if any warning was raised
        assert len(w) > 0

        # Print the warning messages to debug
        for warning in w:
            print(f"Warning message: {str(warning.message)}")

        assert any(
            "gene" in str(warning.message).lower()
            and (
                "match" in str(warning.message).lower()
                or "valid" in str(warning.message).lower()
                or "found" in str(warning.message).lower())
            for warning in w)


def test_filter_anndata_cluster_mask_zero_sum(adata):
    """Test specifically for: if cluster_mask.sum() == 0."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Using completely non-existent clusters guarantees cluster_mask.sum() == 0
        result = rl._filter_anndata(
            adata=adata,
            condition_values=['cluster 0'],
            condition_columns=['cluster'],
            cluster_column='cluster',
            cluster_filter=['nonexistent_cluster1', 'nonexistent_cluster2']
        )

        # Verify the function returns None
        assert result is None

        # Check if any warning was raised
        assert len(w) > 0

        # Print the actual warning messages to help debug
        for warning in w:
            print(f"Warning message: {str(warning.message)}")

        assert any(
            ("cluster" in str(warning.message).lower() or "cell" in str(warning.message).lower())
            and (
                "match" in str(warning.message).lower()
                or "valid" in str(warning.message).lower()
                or "found" in str(warning.message).lower()
            )
            for warning in w
        )


def test_process_condition_combinations(adata_with_conditions):
    """Test processing condition combinations."""
    with patch('sctoolbox.tools.receptor_ligand._filter_anndata') as mock_filter:
        with patch('sctoolbox.tools.receptor_ligand._calculate_condition_difference') as mock_diff:
            with patch('sctoolbox.tools.receptor_ligand.calculate_interaction_table'):
                # Setup filtered data and results
                filtered_data = adata_with_conditions.copy()
                mock_filter.return_value = filtered_data
                mock_diff.return_value = pd.DataFrame({
                    'receptor_gene': ['gene1', 'gene2'],
                    'ligand_gene': ['gene10', 'gene20'],
                    'receptor_cluster': ['cluster 0', 'cluster 1'],
                    'ligand_cluster': ['cluster 1', 'cluster 2'],
                    'rank_diff_treatment_vs_control': [0.3, -0.2],
                    'abs_diff_treatment_vs_control': [0.3, 0.2]
                })

                result = rl._process_condition_combinations(
                    adata=adata_with_conditions,
                    condition_columns=['condition'],
                    condition_values_dict={'condition': ['control', 'treatment']},
                    cluster_column='cluster',
                    sequential_time_analysis=False
                )

                # Check results
                assert len(result) > 0
                assert 'treatment_vs_control' in result
                assert 'differences' in result['treatment_vs_control']
                assert isinstance(result['treatment_vs_control']['differences'], pd.DataFrame)


@pytest.mark.parametrize("inplace,overwrite,has_existing", [
    (True, True, True),    # Modify inplace, overwrite existing
    (False, False, True),  # Return copy, don't overwrite
])
def test_calculate_condition_differences(
    adata_with_conditions,
    diff_results,
    inplace,
    overwrite,
    has_existing
):
    """Test the main calculate_condition_differences function."""
    # Prepare data - add existing results if needed
    adata = adata_with_conditions.copy()
    if has_existing:
        adata.uns.setdefault('sctoolbox', {}).setdefault('receptor-ligand', {}).setdefault('condition-differences', {})
        adata.uns['sctoolbox']['receptor-ligand']['condition-differences'] = diff_results

    # Mock _process_condition_combinations
    with patch('sctoolbox.tools.receptor_ligand._process_condition_combinations') as mock_process:
        # Set up mock return value
        mock_process.return_value = {
            'condition': {
                'new_comparison': {
                    'differences': pd.DataFrame({
                        'receptor_gene': ['gene1'],
                        'ligand_gene': ['gene10'],
                        'receptor_cluster': ['cluster 0'],
                        'ligand_cluster': ['cluster 1'],
                        'rank_diff_new_vs_old': [0.3],
                        'abs_diff_new_vs_old': [0.3]
                    })
                }
            }
        }

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            # Run with parameters
            result = rl.calculate_condition_differences(
                adata=adata,
                condition_columns=['condition'],
                cluster_column='cluster',
                inplace=inplace,
                overwrite=overwrite
            )

            # Check behavior based on parameters
            if has_existing and not overwrite:
                # Should warn and skip processing
                assert len(w) > 0
                assert "already exists" in str(w[0].message)
                assert not mock_process.called
            else:
                # Should process
                assert mock_process.called

            # Check result based on inplace
            if inplace:
                assert result is None
            else:
                assert result is not None
                # Check the copy is different from the original
                assert id(result) != id(adata)
                # Check that data is equivalent
                assert 'sctoolbox' in result.uns
                assert 'receptor-ligand' in result.uns['sctoolbox']
                assert 'condition-differences' in result.uns['sctoolbox']['receptor-ligand']


@pytest.mark.parametrize("error_condition,expected_error", [
    # Missing time_order
    ({"time_column": "timepoint", "time_order": None}, "time_order must be provided"),
    # Invalid time column
    ({"time_column": "non_existent", "time_order": ['day0']}, "time_column"),
    # Invalid condition filter
    ({"condition_filters": {"non_existent": ["value"]}}, "Invalid keys"),
])
def test_calculate_condition_differences_errors(
    adata_with_conditions,
    error_condition,
    expected_error
):
    """Test error handling in condition differences calculation."""
    # Prepare data with time column if needed
    adata = adata_with_conditions.copy()

    # Build kwargs from the error_condition and base parameters
    kwargs = {
        "adata": adata,
        "condition_columns": ['condition'],
        "cluster_column": 'cluster'
    }
    kwargs.update(error_condition)

    # Test for the expected error
    with pytest.raises(ValueError) as excinfo:
        rl.calculate_condition_differences(**kwargs)

    assert expected_error in str(excinfo.value)


def mock_filter_anndata_for_timepoints(kwargs, valid_timepoints):
    """Mock implementation of _filter_anndata for time-based tests."""
    # Get adata from kwargs
    adata = kwargs.get('adata')
    values = kwargs.get('condition_values', [])
    if not values or values[0] not in valid_timepoints:
        return None
    return adata.copy()


@pytest.mark.parametrize(
    "sequential,timepoints",
    [
        (True, ['day0', 'day3', 'day7']),
        (False, ['day0', 'day3', 'day7']),
        (True, ['invalid1', 'invalid2']),
        (True, ['day0', 'day7']),
        (True, ['day0']),
    ]
)
def test_process_condition_combinations_time_analysis(
    adata_with_conditions, sequential, timepoints
):
    """Test time analysis in process_condition_combinations."""
    valid_timepoints = ['day0', 'day3', 'day7']

    with patch('sctoolbox.tools.receptor_ligand._filter_anndata') as mock_filter:
        with patch('sctoolbox.tools.receptor_ligand._calculate_condition_difference') as mock_diff:
            with patch('sctoolbox.tools.receptor_ligand.calculate_interaction_table'):
                # Setup mock filter function with corrected side_effect
                mock_filter.side_effect = lambda **kwargs: mock_filter_anndata_for_timepoints(
                    kwargs, valid_timepoints
                )

                # Setup mock diff function
                mock_diff.return_value = pd.DataFrame({
                    'receptor_gene': ['gene1'],
                    'ligand_gene': ['gene10'],
                    'rank_diff_dummy': [0.5],
                    'abs_diff_dummy': [0.5]
                })

                with warnings.catch_warnings(record=True):
                    rl._process_condition_combinations(
                        adata=adata_with_conditions,
                        condition_columns=['timepoint'],
                        condition_values_dict={'timepoint': timepoints},
                        cluster_column='cluster',
                        sequential_time_analysis=sequential
                    )


def test_calculate_condition_differences_time_analysis(adata_with_conditions):
    """Test time series analysis functionality."""
    # Mock _process_condition_combinations
    with patch('sctoolbox.tools.receptor_ligand._process_condition_combinations') as mock_process:
        # Set up mock return value
        mock_process.return_value = {'test': {'differences': pd.DataFrame()}}

        # Run with time series settings
        rl.calculate_condition_differences(
            adata=adata_with_conditions,
            condition_columns=['timepoint', 'condition'],
            cluster_column='cluster',
            time_column='timepoint',
            time_order=['day0', 'day3', 'day7']
        )

        # Verify mock_process was called with sequential_time_analysis=True
        assert mock_process.called
        assert mock_process.call_args[1]['sequential_time_analysis'] is True


# ------------------------------ VISUALIZATION TESTS -------------------------------- #

def test_extract_diff_key_columns():
    """Test extracting key columns from differences dataframe."""
    # Test standard column names
    df1 = pd.DataFrame({
        'rank_diff_treatment_vs_control': [0.5],
        'abs_diff_treatment_vs_control': [0.5]
    })
    result1 = rl._extract_diff_key_columns(df1)
    assert result1['condition_a'] == 'control'
    assert result1['condition_b'] == 'treatment'
    assert result1['diff_col'] == 'rank_diff_treatment_vs_control'

    # Test time series column info
    df2 = pd.DataFrame({
        'rank_diff_day7_vs_day0': [0.5],
        'abs_diff_day7_vs_day0': [0.5],
        'time_column': ['timepoint'],
        'time_order': ['day0,day3,day7']
    })
    result2 = rl._extract_diff_key_columns(df2)
    assert result2['condition_a'] == 'day0'
    assert result2['condition_b'] == 'day7'
    assert result2['time_column'] == 'timepoint'
    assert result2['time_order'] == ['day0', 'day3', 'day7']


@pytest.mark.parametrize("graph_data,hub_threshold,expected_hubs", [
    # Hub with 5 connections
    ([('hub1', 'node1'), ('hub1', 'node2'), ('hub1', 'node3'), ('node4', 'hub1'), ('node5', 'hub1')],
     4, ['hub1']),
    # No hubs (threshold too high)
    ([('node1', 'node2'), ('node1', 'node3'), ('node2', 'node3')],
     4, []),
])
def test_identify_hub_networks(graph_data, hub_threshold, expected_hubs):
    """Test the network hub identification."""
    # Create test graph
    G = nx.DiGraph()
    G.add_edges_from(graph_data)

    # Identify hubs
    hub_networks, _ = rl._identify_hub_networks(
        G, hub_threshold=hub_threshold
    )

    # Check hub nodes
    assert set(hub_networks.keys()) == set(expected_hubs)


def test_plot_networks(adata_with_diff_results):
    """Test network plotting functionality."""
    # Test with mocked figure
    with patch('matplotlib.pyplot.figure') as mock_fig:
        with patch('matplotlib.pyplot.close'):
            # Mock figure and axis
            mock_fig.return_value = MagicMock(spec=Figure)
            mock_fig.return_value.add_subplot.return_value = MagicMock()

            # Call plotting function
            rl.condition_differences_network(
                adata=adata_with_diff_results,
                n_top=10,
                split_by_direction=False,
                close_figs=True
            )

            # Just check that the function runs without errors
            assert mock_fig.called


@pytest.mark.parametrize(
    "return_figs,show,save,mock_empty",
    [
        # Success cases
        (True, False, None, False),
        (False, False, None, False),
        (True, False, "test_prefix", False),
        (True, False, ("test_prefix", "png"), False),

        # Failure case
        (True, False, None, True),

        # Edge case
        (True, True, "test_prefix", False),
    ]
)
def test_plot_all_condition_differences(
    adata_with_diff_results, return_figs, show, save, mock_empty
):
    """Test plot_all_condition_differences function."""
    # Prepare test data
    adata = adata_with_diff_results.copy()
    if mock_empty:
        adata.uns['sctoolbox']['receptor-ligand']['condition-differences'] = {}

    # Mock dependencies
    with patch('sctoolbox.tools.receptor_ligand.condition_differences_network') as mock_network:
        with patch('matplotlib.pyplot.figure') as mock_figure:
            with patch('matplotlib.pyplot.show') as mock_show:
                # Create a mock figure with a number attribute
                mock_fig = MagicMock(spec=Figure)
                mock_fig.number = 1
                mock_network.return_value = [mock_fig]

                # Test execution
                if mock_empty:
                    with pytest.raises(ValueError) as excinfo:
                        rl.plot_all_condition_differences(
                            adata=adata,
                            show=show,
                            return_figures=return_figs,
                            save=save
                        )
                    assert "No condition differences found" in str(excinfo.value)
                    return

                # Mock plt.figure(fig.number)
                if show:
                    # For show=True tests, intercept the pyplot.figure call
                    mock_figure.return_value = MagicMock()

                result = rl.plot_all_condition_differences(
                    adata=adata,
                    show=show,
                    return_figures=return_figs,
                    save=save
                )

                # Verify results
                if return_figs:
                    assert isinstance(result, dict)
                    assert all(isinstance(value, list) for value in result.values())
                else:
                    assert result is None

                assert mock_show.called == show

                # Check save parameter passed correctly
                if save and mock_network.called:
                    save_was_used = False
                    for call in mock_network.call_args_list:
                        if 'save' in call[1] and save in call[1]['save']:
                            save_was_used = True
                            break
                    assert save_was_used


@pytest.mark.parametrize(
    "gene,cluster,timepoint,expected_result,mock_mean",
    [
        # Valid parameters
        (lambda adata: adata.var_names[0], "cluster 0", "day0", 0.75, True),
        # Non-existent gene
        ("non_existent_gene", "cluster 0", "day0", 0.0, False),
        # Non-existent cluster
        (lambda adata: adata.var_names[0], "non_existent_cluster", "day0", 0.0, False),
        # Non-existent timepoint
        (lambda adata: adata.var_names[0], "cluster 0", "non_existent_timepoint", 0.0, False),
        # Empty mask (no cells match criteria)
        (lambda adata: adata.var_names[0], "cluster 0", "day0", 0.0, None),
    ]
)
def test__get_gene_expression(
    adata_with_conditions, gene, cluster, timepoint, expected_result, mock_mean
):
    """Test the _get_gene_expression function with various parameters."""
    # Resolve callable gene parameter (to handle adata-dependent values)
    actual_gene = gene(adata_with_conditions) if callable(gene) else gene

    if mock_mean:
        # Mock numpy.mean to return a specific value
        with patch('numpy.mean', return_value=expected_result):
            expression = rl._get_gene_expression(
                adata=adata_with_conditions,
                gene=actual_gene,
                cluster=cluster,
                timepoint=timepoint,
                timepoint_col="timepoint",
                cluster_col="cluster"
            )
    elif mock_mean is None:
        # Mock the mask to be empty (no cells match criteria)
        with patch('numpy.where', return_value=(np.array([]),)):
            expression = rl._get_gene_expression(
                adata=adata_with_conditions,
                gene=actual_gene,
                cluster=cluster,
                timepoint=timepoint,
                timepoint_col="timepoint",
                cluster_col="cluster"
            )
    else:
        # No mocking, test the actual behavior
        expression = rl._get_gene_expression(
            adata=adata_with_conditions,
            gene=actual_gene,
            cluster=cluster,
            timepoint=timepoint,
            timepoint_col="timepoint",
            cluster_col="cluster"
        )

    assert expression == expected_result


@pytest.mark.parametrize(
    "time_order,n_cols,use_global_ylim,title,expect_error",
    [
        # Valid timepoints
        (["day0", "day3", "day7"], 2, False, None, False),
        # Missing timepoint - should raise ValueError
        (["day0", "day3", "nonexistent_day"], 2, False, None, True),
        # Valid with custom layout and title
        (["day0", "day3"], 3, False, "Test Title", False),
        # Valid with global y-limits and title
        (["day0", "day3", "day7"], 2, True, "Custom Title", False),
    ]
)
def test_plot_interaction_timeline(
    adata_with_conditions, time_order, n_cols, use_global_ylim, title, expect_error
):
    """Test plot_interaction_timeline with different parameters, including invalid timepoints."""
    # Get a valid interaction from the data
    df = adata_with_conditions.uns["receptor-ligand"]["interactions"]
    interaction = (
        df["receptor_gene"].iloc[0],
        df["receptor_cluster"].iloc[0],
        df["ligand_gene"].iloc[0],
        df["ligand_cluster"].iloc[0]
    )

    # Mock the gene expression function to return a constant value
    expression_mock = patch(
        'sctoolbox.tools.receptor_ligand._get_gene_expression',
        return_value=0.5
    )

    # Mock the figure saving function
    save_mock = patch.object(Figure, 'savefig')

    # Apply the mocks
    expression_mock.start()
    save_mock.start()

    # Additional patch for global ylim case to prevent comparison errors
    if use_global_ylim:
        max_mock = patch('numpy.max', return_value=0.8)
        max_mock.start()

    try:
        if expect_error:
            # Test case should raise ValueError
            with pytest.raises(ValueError):
                rl.plot_interaction_timeline(
                    adata=adata_with_conditions,
                    interactions=[interaction],
                    timepoint_column="timepoint",
                    cluster_column="cluster",
                    time_order=time_order,
                    n_cols=n_cols,
                    use_global_ylim=use_global_ylim,
                    title=title
                )
        else:
            # Test case should succeed
            fig = rl.plot_interaction_timeline(
                adata=adata_with_conditions,
                interactions=[interaction],
                timepoint_column="timepoint",
                cluster_column="cluster",
                time_order=time_order,
                n_cols=n_cols,
                use_global_ylim=use_global_ylim,
                title=title
            )
            # Verify result
            assert isinstance(fig, Figure)
            assert len(fig.axes) >= 1
            if title:
                assert fig._suptitle.get_text() == title
    finally:
        # Stop all mocks
        expression_mock.stop()
        save_mock.stop()
        if use_global_ylim:
            max_mock.stop()
