"""Fixtures available to all tests within the plotting directory."""

import pytest
import matplotlib.axes
import numpy as np
import pandas as pd
import ipywidgets as widgets
import sctoolbox.tools as tools

# ---------------------------- Script variables --------------------------- #
# global variables for this script

__rank_key = "rank_genes_groups"


# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture
def assert_axes_array():
    """Return a helper that asserts an object is a numpy array of matplotlib Axes.

    Returns
    -------
    callable
        Assertion helper function.
    """
    def _assert(axes):
        assert isinstance(axes, np.ndarray)
        assert isinstance(axes[0], matplotlib.axes.Axes)
    return _assert


@pytest.fixture
def df_bidir_bar():
    """Create DataFrame for bidirectional barplot.

    Returns
    -------
    pd.DataFrame
        DataFrame with left and right labels and values for bidirectional barplot.
    """
    return pd.DataFrame(data={'left_label': np.random.choice(["B1", "B2", "B3"], size=5),
                              'right_label': np.random.choice(["A1", "A2", "A3"], size=5),
                              'left_value': np.random.normal(size=5),
                              'right_value': np.random.normal(size=5)})


@pytest.fixture
def df():
    """Create and return a pandas dataframe.

    Returns
    -------
    pd.DataFrame
        Simple dataframe with two columns.
    """
    return pd.DataFrame(data={'col1': [1, 2, 3, 4, 5],
                              'col2': [3, 4, 5, 6, 7]})


@pytest.fixture
def venn_dict():
    """Create arbitrary groups for venn.

    Returns
    -------
    dict
        Dictionary with group names as keys and lists of items as values for Venn diagram.
    """
    return {"Group A": [1, 2, 3, 4, 5, 6],
            "Group B": [2, 3, 7, 8],
            "Group C": [3, 4, 5, 9, 10]}


@pytest.fixture(scope="session")  # reuse the fixture for all tests
def adata_gsea():
    """Minimal adata file for testing.

    Returns
    -------
    anndata.AnnData
        AnnData object with GSEA results.
    """
    from tests.conftest import _make_adata
    adata = _make_adata()

    tools.gsea.gene_set_enrichment(adata,
                                   marker_key=__rank_key,
                                   organism="human",
                                   method="prerank",
                                   inplace=True,
                                   save_table=None)

    return adata


@pytest.fixture
def pairwise_ranked_genes():
    """Return a DataFrame of genes ranked in groups.

    Returns
    -------
    pd.DataFrame
        DataFrame with pairwise gene rankings across groups.
    """
    return pd.DataFrame(data={"1/2_group": ["C1", "C1", "C2", "C2"],
                              "1/3_group": ["C1", "NS", "C2", "C2"],
                              "2/3_group": ["C1", "C1", "NS", "C2"]},
                        index=["GeneA", "GeneB", "GeneC", "GeneD"])


@pytest.fixture
def pairwise_ranked_genes_nosig():
    """Return a DataFrame of genes ranked in groups with none significant.

    Returns
    -------
    pd.DataFrame
        DataFrame with no significant genes in any pairwise comparison.
    """
    return pd.DataFrame(data={"1/2_group": ["NS", "NS", "NS", "NS"],
                              "1/3_group": ["NS", "NS", "NS", "NS"],
                              "2/3_group": ["NS", "NS", "NS", "NS"]},
                        index=["GeneA", "GeneB", "GeneC", "GeneD"])


@pytest.fixture
def slider():
    """Create a slider widget.

    Returns
    -------
    ipywidgets.FloatRangeSlider
        Slider widget with default range.
    """
    return widgets.FloatRangeSlider(value=[5, 7], min=0, max=10, step=1)


@pytest.fixture
def slider_list(slider):
    """Create a list of slider widgets.

    Returns
    -------
    list
        List of slider widgets.
    """
    return [slider for _ in range(2)]


@pytest.fixture
def checkbox():
    """Create a checkbox widget.

    Returns
    -------
    ipywidgets.Checkbox
        Checkbox widget.
    """
    return widgets.Checkbox()


@pytest.fixture
def slider_dict(slider):
    """Create a dict of sliders.

    Returns
    -------
    dict
        Dictionary mapping column names to sliders.
    """
    return {c: slider for c in ['LISI_score_pca', 'qc_float']}


@pytest.fixture
def slider_dict_grouped(slider):
    """Create a nested dict of slider widgets.

    Returns
    -------
    dict
        Nested dictionary mapping columns to groups to sliders.
    """
    return {c: {g: slider for g in ['C1', 'C2', 'C3']} for c in ['LISI_score_pca', 'qc_float']}


@pytest.fixture
def slider_dict_grouped_diff(slider):
    """Create a nested dict of slider widgets with different selections.

    Returns
    -------
    dict
        Nested dictionary with varied slider configurations.
    """
    return {"A": {"1": slider, "2": widgets.FloatRangeSlider(value=[1, 5], min=0, max=10, step=1)},
            "B": {"1": slider, "2": widgets.FloatRangeSlider(value=[3, 4], min=0, max=10, step=1)}}


