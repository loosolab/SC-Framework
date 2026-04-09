"""Test planet plot functions."""

import pytest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import sctoolbox.plotting.planet_plot as pp

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")

# ---------------------------- Script variables --------------------------- #
# global variables for this script

# Common planet plot parameters shared across tests
_X_COL = "category1"
_Y_COL = "category2"
_INPUT_LAYER = "test_layer"
_OBS_COLUMNS = ["obscol1", "obscol2", "obscol3", "obscol4", "obscol5", "obscol6"]

# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture(scope="session")
def adata_planet_plot():
    """Load and returns an anndata object.

    Returns
    -------
    anndata.AnnData
        AnnData object with test layer for planet plot.
    """
    np.random.seed(1)  # set seed for reproducibility

    adata = sc.AnnData(X=np.random.rand(200, 100))

    # create a test layer with a diagonal matrix with 1s starting at index, 0, 50, 100 and 150
    # This ensures that each of the gene has the expression value exactly 4 times as 1 and once for each subset (category1, category2)
    test_layer = np.zeros((200, 100))
    for i in range(50):
        test_layer[i, i] = 1
        test_layer[i + 50, i] = 1
        test_layer[i + 100, i] = 1
        test_layer[i + 150, i] = 1
        test_layer[i, i + 50] = 1
        test_layer[i + 50, i + 50] = 1
        test_layer[i + 100, i + 50] = 1
        test_layer[i + 150, i + 50] = 1

    adata.layers["test_layer"] = test_layer

    # next we create a df for adata.obs
    # create category1 and category2 used for grouping, each column has 2 different values
    # we make all 4 combinations of (category1, category2) appear equal no of times
    category1 = ['A'] * 100 + ['B'] * 100
    category2 = ['a'] * 50 + ['b'] * 50 + ['a'] * 50 + ['b'] * 50
    df = pd.DataFrame({'category1': category1, 'category2': category2})

    # next we create 6 obs columns that each contain single 1 fore each category combination
    for i in range(6):
        df[f'obscol{i + 1}'] = test_layer[:, i]

    adata.obs = df
    return adata


@pytest.fixture(scope="session")
def planet_plot_vars(adata_planet_plot):
    """Preprocessed planet plot data for use as test setup.

    Returns
    -------
    pd.DataFrame
        Preprocessed planet plot variables.
    """
    genes = list(adata_planet_plot.var.index[:4])
    return pp.planet_plot_anndata_preprocess(
        adata=adata_planet_plot,
        x_col=_X_COL,
        y_col=_Y_COL,
        input_layer=_INPUT_LAYER,
        genes=genes,
        gene_symbols=None,
        obs_columns=_OBS_COLUMNS,
    )


# ------------------------------ TESTS --------------------------------- #


@pytest.mark.parametrize("group, threshold, output",
                         [(pd.Series([1, 2, 3, 4]), 1, 3),
                          (pd.Series([0, 0.0, 0.1, 1]), 0, 2),
                          (pd.Series([-1, -0.5, 0.1, 1]), -0.7, 3)])
def test_count_greater_than_threshold(group, threshold, output):
    """Test greater than threshold."""
    threshold_exceedence_count = pp._count_greater_than_threshold(group, threshold)
    assert threshold_exceedence_count == output


@pytest.mark.parametrize("values, min_value, max_value, min_dot_size, max_dot_size, use_log_scale, expected_output",
                         [(np.array([1, 2, 3]), 1, 3, 1, 3, False, np.array([1, 2, 3])),
                          (np.array([10, 100, 1000, 10000]), 10, 10000, 1, 4, True, np.array([1, 2, 3, 4])),
                          (np.array([10, 100, 1000, 10000]), 10, 10000, 1, 1, True, np.array([1, 1, 1, 1])),
                          (np.array([10, 100, 1000, 10000]), 10, 10, 1, 4, True, np.array([1, 1, 1, 1]))])
def test_calculate_dot_sizes(values, min_value, max_value, min_dot_size, max_dot_size, use_log_scale, expected_output):
    """Test greater than threshold for linear and log cases."""
    sizes = pp._calculate_dot_sizes(values, min_value, max_value, min_dot_size, max_dot_size, use_log_scale)
    for i, value in enumerate(expected_output):
        # here we use ceiling function because using log1p instead of log causes a small skew at smaller values where the user should not ideally use log scale.
        # eg. because of log1p, the size of 1.976386 instead of 2 is calculate for the value 100.
        assert value == np.ceil(sizes[i])


@pytest.mark.parametrize("aggregator, to_aggregate, expected_output",
                         [("median", "count", 2),
                          ("median", "expression", 0.2),
                          ("expression_weighted_count", "count", 2.33),
                          ("count_weighted_expression", "expression", 0.233)])
def test_genes_aggregator(aggregator, to_aggregate, expected_output):
    """Test genes aggregator for different cases."""
    mock_data = {'count1': [1],
                 'count2': [2],
                 'count3': [3],
                 'exp1': [0.1],
                 'exp2': [0.2],
                 'exp3': [0.3]}

    df = pd.DataFrame(mock_data)
    agg_val = pp._genes_aggregator(df,
                                   ["count1", "count2", "count3"],
                                   ["exp1", "exp2", "exp3"],
                                   aggregator,
                                   to_aggregate)
    # since we know it returns a single aggregate value
    agg_val = agg_val.values[0]
    assert np.isclose(agg_val, expected_output, rtol=0.01)


def test_planet_plot_anndata_preprocess(adata_planet_plot):
    """Test planet plot preprocess for the given adata."""
    genes = list(adata_planet_plot.var.index[:4])

    # create the expected output array
    row_template = [50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 1.0,
                    0.02, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0,
                    2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0]
    expected_df_array = np.array(
        [[cat1, cat2] + row_template for cat1, cat2 in [("A", "a"), ("A", "b"), ("B", "a"), ("B", "b")]],
        dtype=object
    )

    # using default values for aggregators
    plot_vars = pp.planet_plot_anndata_preprocess(adata=adata_planet_plot,
                                                  x_col=_X_COL,
                                                  y_col=_Y_COL,
                                                  input_layer=_INPUT_LAYER,
                                                  genes=genes,
                                                  gene_symbols=None,
                                                  obs_columns=_OBS_COLUMNS)

    assert np.array_equal(expected_df_array, np.array(plot_vars.values))


@pytest.mark.parametrize("mode, output_mode", [("aggregate", 1), ("planet", 2)])
@pytest.mark.parametrize("size_value, output_size_value", [("count", 1), ("percentage", 1)])
@pytest.mark.parametrize("color_value, output_color_value", [("value", 1), ("percentage_max", 1)])
@pytest.mark.parametrize("planet_columns, planet_color_schemas, output_schemas",
                         [(["0", "1", "2", "3"], None, 0),
                          (["obscol1", "obscol2", "obscol3", "obscol4", "obscol5", "obscol6"], ["Accent", "twilight", "CMRmap", "cividis", "gray", "coolwarm"], 6)])
def test_planet_plot_render(planet_plot_vars,
                            mode,
                            size_value,
                            color_value,
                            planet_columns,
                            planet_color_schemas,
                            output_mode,
                            output_size_value,
                            output_color_value,
                            output_schemas):
    """Test planet plot render for the given adata."""
    axes = pp.planet_plot_render(plot_vars=planet_plot_vars,
                                 x_col=_X_COL,
                                 y_col=_Y_COL,
                                 mode=mode,
                                 size_value=size_value,
                                 color_value=color_value,
                                 planet_columns=planet_columns,
                                 planet_color_schemas=planet_color_schemas)

    # set 0 for output_schemas for aggregate mode
    if mode == "aggregate":
        output_schemas = 0
    # assert count of plot axes
    assert len(axes) == output_mode + output_size_value + output_color_value + output_schemas
