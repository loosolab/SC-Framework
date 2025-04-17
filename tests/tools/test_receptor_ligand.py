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

# ------------------------------ HELPER DIFFENRENCE FUNCTION TESTS -------------------------------- #

@pytest.mark.parametrize("inplace", [True, False])
def test_add_uns_info_rl(inplace):
    """Test adding results to .uns."""
    # Create test data
    adata = sc.AnnData(np.random.rand(10, 10))
    value = {'test': {'data': {'differences': pd.DataFrame()}}}

    # Test the function
    result = rl._add_uns_info_rl(adata, value, inplace=inplace)

    if inplace:
        # Check inplace modification
        assert result is None
        assert 'sctoolbox' in adata.uns
        assert 'receptor-ligand' in adata.uns['sctoolbox']
        assert 'condition-differences' in adata.uns['sctoolbox']['receptor-ligand']
        assert adata.uns['sctoolbox']['receptor-ligand']['condition-differences'] == value
    else:
        # Check return of copy
        assert result is not None
        assert 'sctoolbox' not in adata.uns  # Original unchanged
        assert 'sctoolbox' in result.uns
        assert 'receptor-ligand' in result.uns['sctoolbox']
        assert 'condition-differences' in result.uns['sctoolbox']['receptor-ligand']
        assert result.uns['sctoolbox']['receptor-ligand']['condition-differences'] == value


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
    with patch('sctoolbox.tools.receptor_ligand.calculate_interaction_table') as mock_calc:
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
                # Verify interaction table calculation was called
                mock_calc.assert_called_once()
            else:
                assert result is None
                # For failure cases, mock_calc should not be called
                mock_calc.assert_not_called()


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


def test_calculate_condition_difference(adata_with_conditions):
    """Test calculating differences between conditions."""
    # Create copies for different conditions
    adata_a = adata_with_conditions.copy()
    adata_b = adata_with_conditions.copy()

    # Mock get_interactions to return test data
    with patch('sctoolbox.tools.receptor_ligand.get_interactions') as mock_get:
        # Mock different interaction tables for the two conditions
        mock_get.side_effect = [
            pd.DataFrame({  # Condition A
                'receptor_gene': ['gene1', 'gene2', 'gene3'],
                'ligand_gene': ['gene10', 'gene20', 'gene30'],
                'receptor_cluster': ['cluster 0', 'cluster 1', 'cluster 2'],
                'ligand_cluster': ['cluster 1', 'cluster 2', 'cluster 0'],
                'interaction_score': [0.8, 0.6, 0.7]
            }),
            pd.DataFrame({  # Condition B
                'receptor_gene': ['gene1', 'gene2', 'gene3'],
                'ligand_gene': ['gene10', 'gene20', 'gene30'],
                'receptor_cluster': ['cluster 0', 'cluster 1', 'cluster 2'],
                'ligand_cluster': ['cluster 1', 'cluster 2', 'cluster 0'],
                'interaction_score': [0.5, 0.3, 0.9]
            })
        ]

        # Calculate differences
        result = rl._calculate_condition_difference(
            adata_a=adata_a,
            adata_b=adata_b,
            condition_a_name='control',
            condition_b_name='treatment',
            min_perc=None,
            interaction_score=0.5,
            interaction_perc=None
        )

        # Check the result structure
        assert 'rank_diff_treatment_vs_control' in result.columns
        assert 'abs_diff_treatment_vs_control' in result.columns
        assert all(result['abs_diff_treatment_vs_control'] >= 0)


def test_process_condition_combinations(adata_with_conditions):
    """Test processing condition combinations."""
    # Mock filter_anndata and calculate_condition_difference
    with patch('sctoolbox.tools.receptor_ligand._filter_anndata') as mock_filter:
        with patch('sctoolbox.tools.receptor_ligand._calculate_condition_difference') as mock_calc:
            # Setup filtered data and results
            filtered_data = adata_with_conditions.copy()
            mock_filter.return_value = filtered_data
            mock_calc.return_value = pd.DataFrame({
                'receptor_gene': ['gene1', 'gene2'],
                'ligand_gene': ['gene10', 'gene20'],
                'receptor_cluster': ['cluster 0', 'cluster 1'],
                'ligand_cluster': ['cluster 1', 'cluster 2'],
                'rank_diff_treatment_vs_control': [0.3, -0.2],
                'abs_diff_treatment_vs_control': [0.3, 0.2]
            })

            # Process standard condition comparison
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

            # Check structure of the results
            comparison_data = result['treatment_vs_control']
            assert 'differences' in comparison_data
            assert isinstance(comparison_data['differences'], pd.DataFrame)


# ------------------------------ MAIN FUNCTION DIFFERENCE TESTS -------------------------------- #

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
                assert id(result) != id(adata)  # Use this instead of direct 'is not' comparison
                # Check that data is equivalent
                assert 'sctoolbox' in result.uns
                assert 'receptor-ligand' in result.uns['sctoolbox']
                assert 'condition-differences' in result.uns['sctoolbox']['receptor-ligand']


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


# ------------------------------ ERROR CASE TESTS -------------------------------- #

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
            mock_fig.return_value = MagicMock()
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
