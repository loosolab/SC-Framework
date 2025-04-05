"""Test receptor-ligand functions."""

import pytest
import os
import pandas as pd
import numpy as np
import scanpy as sc
import random
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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


@pytest.fixture
def adata_with_conditions(adata_inter):
    """Add condition and timepoint columns to the AnnData object."""
    obj = adata_inter.copy()

    # Add condition column with treatment and control group
    n_cells = obj.n_obs
    obj.obs['condition'] = ['control'] * (n_cells // 2) + ['treatment'] * (n_cells - n_cells // 2)

    # Add timepoint column
    timepoints = ['tp1', 'tp2', 'tp3']
    obj.obs['timepoint'] = [timepoints[i % 3] for i in range(n_cells)]

    return obj


@pytest.fixture
def mock_diff_results():
    """Create mock difference results for testing."""
    # Create a simple mock structure mimicking the output of calculate_condition_differences
    differences_df = pd.DataFrame({
        'receptor_gene': ['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'],
        'ligand_gene': ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE'],
        'receptor_cluster': ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 1', 'cluster 2'],
        'ligand_cluster': ['cluster 4', 'cluster 5', 'cluster 6', 'cluster 5', 'cluster 6'],
        'interaction_score_a': [0.5, 0.6, 0.7, 0.8, 0.9],
        'interaction_score_b': [0.9, 0.8, 0.7, 0.6, 0.5],
        'quantile_rank_a': [0.2, 0.3, 0.5, 0.7, 0.9],
        'quantile_rank_b': [0.9, 0.8, 0.5, 0.3, 0.1],
        'rank_diff': [0.7, 0.5, 0.0, -0.4, -0.8],
        'abs_diff': [0.7, 0.5, 0.0, 0.4, 0.8]
    })

    # Add attrs to mimic output
    differences_df.attrs = {
        'condition_a': 'Control',
        'condition_b': 'Treatment',
        'timepoint_1': 'tp1',
        'timepoint_2': 'tp2',
        'condition': 'test_experiment',
        'group_name': 'Test Comparison',
        'condition_name': 'test_dimension'
    }

    # Create the nested dictionary structure
    return {
        'test_dimension': {
            'Treatment_vs_Control': {
                'differences': differences_df
            }
        },
        'time_series': {
            'tp2_Treatment_vs_tp1_Treatment': {
                'differences': differences_df.copy()
            }
        }
    }


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


# ----- test difference analysis functions ----- #

def test_condition_differences_network(mock_diff_results):
    """Test that condition_differences_network generates figures."""
    figures = rl.condition_differences_network(
        diff_results=mock_diff_results,
        n_top=5,
        figsize=(10, 8),
        dpi=72,
        split_by_direction=True,
        hub_threshold=2
    )

    # Check if figures were created
    assert isinstance(figures, list)
    assert all(isinstance(fig, Figure) for fig in figures)

    for fig in figures:
        plt.close(fig)


def test_plot_all_condition_differences(mock_diff_results):
    """Test that plot_all_condition_differences works with mock data."""
    figures_dict = rl.plot_all_condition_differences(
        diff_results=mock_diff_results,
        n_top=5,
        figsize=(10, 8),
        dpi=72,
        split_by_direction=True,
        hub_threshold=2,
        show=False,
        return_figures=True
    )

    # Check the returned dictionary structure
    assert isinstance(figures_dict, dict)
    assert all(isinstance(figures_dict[dim], list) for dim in figures_dict)

    for dim in figures_dict:
        for fig in figures_dict[dim]:
            plt.close(fig)


def test_track_clusters_or_genes(mock_diff_results):
    """Test track_clusters_or_genes works with mock data."""
    timepoint_order = ['tp1', 'tp2', 'tp3']

    # Track cluster
    figures = rl.track_clusters_or_genes(
        diff_results=mock_diff_results,
        clusters=['cluster 1', 'cluster 2'],
        timepoint_order=timepoint_order,
        min_interactions=1,
        n_top=5,
        figsize=(10, 8),
        dpi=72,
        split_by_direction=True,
        hub_threshold=2
    )

    # Check if figures were created
    assert isinstance(figures, list)
    assert all(isinstance(fig, Figure) for fig in figures)

    for fig in figures:
        plt.close(fig)

    # Track genes
    figures = rl.track_clusters_or_genes(
        diff_results=mock_diff_results,
        genes=['Gene1', 'GeneA'],
        timepoint_order=timepoint_order,
        min_interactions=1,
        n_top=5,
        figsize=(10, 8),
        dpi=72,
        split_by_direction=True,
        hub_threshold=2
    )

    # Check if figures were created
    assert isinstance(figures, list)
    assert all(isinstance(fig, Figure) for fig in figures)

    for fig in figures:
        plt.close(fig)


def test_calculate_condition_differences(adata_with_conditions):
    """Test if calculate_condition_differences works with condition adata."""
    diff_results = rl.calculate_condition_differences(
        adata=adata_with_conditions,
        condition_columns=['condition'],
        cluster_column='cluster',
        min_perc=0,
        condition_filters={'condition': ['control', 'treatment']},
        inplace=False
    )

    # Verify structure
    assert isinstance(diff_results, dict)
    assert 'condition' in diff_results

    # Check if at least one comparison and difference df
    has_comparisons = False
    for dim, comparisons in diff_results.items():
        if comparisons:
            has_comparisons = True
            for comp_key, comp_data in comparisons.items():
                if 'differences' in comp_data:
                    assert isinstance(comp_data['differences'], pd.DataFrame)
                    break
    assert has_comparisons


def test_calculate_condition_differences_over_time(adata_with_conditions):
    """Test if calculate_condition_differences_over_time works with condition adata."""
    diff_results = rl.calculate_condition_differences_over_time(
        adata=adata_with_conditions,
        timepoint_column='timepoint',
        condition_column='condition',
        condition_value='control',
        cluster_column='cluster',
        timepoint_order=['tp1', 'tp2', 'tp3'],
        min_perc=0,
        inplace=False
    )

    assert isinstance(diff_results, dict)
    assert 'time_series' in diff_results
