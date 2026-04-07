"""Fixtures available to all tests within the plotting directory."""

import pytest
import scanpy as sc
import os
import numpy as np
import tempfile
import shutil
import pandas as pd
import ipywidgets as widgets
import sctoolbox.tools as tools

# ---------------------------- Script variables --------------------------- #
# global variables for this script

__rank_key = "rank_genes_groups"


# ------------------------------ FIXTURES --------------------------------- #


def _make_adata():
    """Load and returns an anndata object.

    Returns
    -------
    anndata.AnnData
        AnnData object with processed data and clustering results.
    """

    np.random.seed(1)  # set seed for reproducibility

    adata = sc.datasets.pbmc68k_reduced()
    adata.raw = None

    # create nonsense layers to enable velocity testing
    adata.layers["spliced"] = adata.X * 2
    adata.layers["unspliced"] = adata.X * 3

    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
    adata.obs["cat"] = adata.obs["condition"].astype("category")

    adata.obs["LISI_score_pca"] = np.random.normal(size=adata.shape[0])
    adata.obs["qc_float"] = np.random.uniform(0, 1, size=adata.shape[0])
    adata.var["qc_float_var"] = np.random.uniform(0, 1, size=adata.shape[1])

    adata.obs["qcvar1"] = np.random.normal(size=adata.shape[0])
    adata.obs["qcvar2"] = np.random.normal(size=adata.shape[0])

    sc.tl.umap(adata, n_components=3)  # to have more than two components available
    sc.tl.tsne(adata)

    sc.tl.rank_genes_groups(adata, groupby='louvain', key_added=__rank_key)

    return adata


@pytest.fixture(scope="session")  # reuse the fixture for all tests
def adata():
    """Create a fixture of the adata with session scope.

    Returns
    -------
    anndata.AnnData
        AnnData object with session scope.
    """
    return _make_adata()


@pytest.fixture(scope="function")  # create a new fixture for each test
def adata_fun_scope():
    """Create a fixture of the adata with function scope.

    Returns
    -------
    anndata.AnnData
        AnnData object with function scope.
    """
    return _make_adata()


@pytest.fixture
def tmp_file():
    """
    Return path for a temporary file.

    Yields
    ------
    A temporary file path
    """
    # TODO replace with the pytest native tempfile fixture
    tmpdir = tempfile.mkdtemp()

    yield os.path.join(tmpdir, "output.pdf")

    # clean up directory and contents
    shutil.rmtree(tmpdir)


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
    adata = _make_adata()

    tools.gsea.gene_set_enrichment(adata,
                                   marker_key=__rank_key,
                                   organism="human",
                                   method="prerank",
                                   inplace=True)

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


@pytest.fixture(scope="session")  # reuse the fixture for all tests
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


@pytest.fixture
def atac_adata():
    """Fixture for an AnnData object.

    Returns
    -------
    anndata.AnnData
        ATAC-seq AnnData object.
    """
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'mm10_atac.h5ad'))
    return adata
