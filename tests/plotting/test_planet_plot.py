"""Test planet plot functions."""
import os
import pytest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import sctoolbox.plotting.planet_plot as pp

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")


# ------------------------------ FIXTURES --------------------------------- #

@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """Load and returns an anndata object."""

    np.random.seed(1)  # set seed for reproducibility

    f = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', "adata.h5ad")
    adata = sc.read_h5ad(f)

    # create a test layer with a diagonal matrix with 1s starting at index, 0,50,100 and 150
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
        df[f'obscol{i+1}'] = test_layer[:, i]

    adata.obs = df
    return adata


# ------------------------------ TESTS --------------------------------- #


@pytest.mark.parametrize("group, threshold, output",
                         [(pd.Series([1, 2, 3, 4]), 1, 3),
                          (pd.Series([0, 0.0, 0.1, 1]), 0, 2),
                          (pd.Series([-1, -0.5, 0.1, 1]), -0.7, 3)])
def test_count_greater_than_threshold(group, threshold, output):
    """Test greater than threshold."""
    threshold_exceedence_count = pp.count_greater_than_threshold(group, threshold)
    assert threshold_exceedence_count == output


@pytest.mark.parametrize("values, min_value, max_value, min_dot_size, max_dot_size, use_log_scale, expected_output",
                         [(np.array([1, 2, 3]), 1, 3, 1, 3, False, np.array([1, 2, 3])),
                          (np.array([10, 100, 1000, 10000]), 10, 10000, 1, 4, True, np.array([1, 2, 3, 4])),
                          (np.array([10, 100, 1000, 10000]), 10, 10000, 1, 1, True, np.array([1, 1, 1, 1])),
                          (np.array([10, 100, 1000, 10000]), 10, 10, 1, 4, True, np.array([1, 1, 1, 1]))])
def test_calculate_dot_sizes(values, min_value, max_value, min_dot_size, max_dot_size, use_log_scale, expected_output):
    """Test greater than threshold for linear and log cases."""
    sizes = pp.calculate_dot_sizes(values, min_value, max_value, min_dot_size, max_dot_size, use_log_scale)
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
    mock_data = {
        'count1': [1],
        'count2': [2],
        'count3': [3],
        'exp1': [0.1],
        'exp2': [0.2],
        'exp3': [0.3]
    }
    df = pd.DataFrame(mock_data)
    agg_val = pp.genes_aggregator(df,
                                  ["count1", "count2", "count3"],
                                  ["exp1", "exp2", "exp3"],
                                  aggregator,
                                  to_aggregate)
    # since we know it returns a single aggregate value
    agg_val = agg_val.values[0]
    assert np.isclose(agg_val, expected_output, rtol=0.01)


def test_planet_plot_anndata_preprocess(adata):
    """ Test planet plot preprocess for the given adata."""
    x_col = "category1"
    y_col = "category2"
    input_layer = "test_layer"
    genes = ["ENSMUSG00000103377", "ENSMUSG00000064842", "ENSMUSG00000104428", "ENSMUSG00000065625"]
    gene_symbols = None
    obs_columns = ["obscol1", "obscol2", "obscol3", "obscol4", "obscol5", "obscol6"]
    expected_df_array = np.array([['A', 'a', 50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                   0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 1.0,
                                   0.02, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0,
                                   2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0],
                                  ['A', 'b', 50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                   0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 1.0,
                                   0.02, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0,
                                   2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0],
                                  ['B', 'a', 50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                   0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 1.0,
                                   0.02, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0,
                                   2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0],
                                  ['B', 'b', 50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                   0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 1.0,
                                   0.02, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0,
                                   2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0, 2.0, 100.0]], dtype=object)
    # using default values for aggregators
    plot_vars = pp.planet_plot_anndata_preprocess(adata=adata,
                                                  x_col=x_col,
                                                  y_col=y_col,
                                                  input_layer=input_layer,
                                                  genes=genes,
                                                  gene_symbols=gene_symbols,
                                                  obs_columns=obs_columns)
    # assert equality with expected
    assert np.array_equal(expected_df_array, np.array(plot_vars.values))


@pytest.mark.parametrize("mode, output_mode", [("aggregate", 1), ("planet", 2)])
@pytest.mark.parametrize("size_value, output_size_value", [("count", 1), ("percentage", 1)])
@pytest.mark.parametrize("color_value, output_color_value", [("value", 1), ("percentage_max", 1)])
@pytest.mark.parametrize("planet_columns, planet_color_schemas, output_schemas",
                         [(["ENSMUSG00000103377", "ENSMUSG00000064842", "ENSMUSG00000104428", "ENSMUSG00000065625"], None, 0),
                          (["obscol1", "obscol2", "obscol3", "obscol4", "obscol5", "obscol6"], ["Accent", "twilight", "CMRmap", "cividis", "gray", "coolwarm"], 6)])
def test_planet_plot_render(adata,
                            mode,
                            size_value,
                            color_value,
                            planet_columns,
                            planet_color_schemas,
                            output_mode,
                            output_size_value,
                            output_color_value,
                            output_schemas):
    """ Test planet plot render for the given adata."""
    x_col = "category1"
    y_col = "category2"
    input_layer = "test_layer"
    genes = ["ENSMUSG00000103377", "ENSMUSG00000064842", "ENSMUSG00000104428", "ENSMUSG00000065625"]
    gene_symbols = None
    obs_columns = ["obscol1", "obscol2", "obscol3", "obscol4", "obscol5", "obscol6"]
    plot_vars = pp.planet_plot_anndata_preprocess(adata=adata, x_col=x_col,
                                                  y_col=y_col,
                                                  input_layer=input_layer,
                                                  genes=genes,
                                                  gene_symbols=gene_symbols,
                                                  obs_columns=obs_columns)
    axes = pp.planet_plot_render(plot_vars=plot_vars,
                                 x_col=x_col,
                                 y_col=y_col,
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
