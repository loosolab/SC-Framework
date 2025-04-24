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

# ------------------------------ HELPER DIFFERENCE FUNCTION TESTS -------------------------------- #


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


def mock_filter_anndata_for_timepoints(args, kwargs, valid_timepoints):
    """Mock implementation of _filter_anndata for time-based tests."""
    values = kwargs.get('condition_values', [])
    if not values or values[0] not in valid_timepoints:
        return None
    return args[0].copy()


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


def mock_filter_anndata_for_timepoints(args, kwargs, valid_timepoints):
    """Mock implementation of _filter_anndata for time-based tests."""
    # Get adata from kwargs instead of args
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
                # Setup mock filter function
                mock_filter.side_effect = lambda **kwargs: mock_filter_anndata_for_timepoints(
                    (), kwargs, valid_timepoints
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
                assert id(result) != id(adata)
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
    "return_figs,show,save_prefix,mock_empty",
    [
        # Success cases
        (True, False, None, False),
        (False, False, None, False),
        (True, False, "test_prefix", False),

        # Failure case
        (True, False, None, True),

        # Edge case
        (True, True, "test_prefix", False),
    ]
)
def test_plot_all_condition_differences(
    adata_with_diff_results, return_figs, show, save_prefix, mock_empty
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
                            save_prefix=save_prefix
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
                    save_prefix=save_prefix
                )

                # Verify results
                if return_figs:
                    assert isinstance(result, dict)
                    assert all(isinstance(value, list) for value in result.values())
                else:
                    assert result is None

                assert mock_show.called == show

                # Check save parameter passed correctly
                if save_prefix and mock_network.called:
                    save_was_used = False
                    for call in mock_network.call_args_list:
                        if 'save' in call[1] and save_prefix in call[1]['save']:
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
    "timepoints,n_cols,use_global_ylim,title",
    [
        # Default settings
        (None, 2, False, None),
        # Specific timepoints
        (["day0", "day3"], 2, False, None),
        # Custom layout and title
        (None, 3, False, "Test Title"),
        # Global y-limits and title
        (None, 2, True, "Custom Title"),
    ]
)
def test_plot_interactions_overtime(
    adata_with_conditions, timepoints, n_cols, use_global_ylim, title
):
    """Test plot_interactions_overtime with different parameters."""
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
        # Call function with test parameters
        fig = rl.plot_interactions_overtime(
            adata=adata_with_conditions,
            interactions=[interaction],
            timepoint_column="timepoint",
            cluster_column="cluster",
            timepoints=timepoints,
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
