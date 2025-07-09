"""Test general plotting functions."""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scanpy as sc
import sctoolbox.plotting.general as pl

from beartype.roar import BeartypeCallHintParamViolation


# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture
def df_bidir_bar():
    """Create DataFrame for bidirectional barplot."""
    return pd.DataFrame(data={'left_label': np.random.choice(["B1", "B2", "B3"], size=5),
                              'right_label': np.random.choice(["A1", "A2", "A3"], size=5),
                              'left_value': np.random.normal(size=5),
                              'right_value': np.random.normal(size=5)})


@pytest.fixture
def df():
    """Create and return a pandas dataframe."""
    return pd.DataFrame(data={'col1': [1, 2, 3, 4, 5],
                              'col2': [3, 4, 5, 6, 7]})


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """Load and returns an anndata object."""

    np.random.seed(1)  # set seed for reproducibility

    f = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', "adata.h5ad")
    adata = sc.read_h5ad(f)

    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
    adata.obs["clustering"] = np.random.choice(["1", "2", "3", "4"], size=adata.shape[0])
    adata.obs["cat"] = adata.obs["condition"].astype("category")

    adata.obs["LISI_score_pca"] = np.random.normal(size=adata.shape[0])
    adata.obs["qc_float"] = np.random.uniform(0, 1, size=adata.shape[0])
    adata.var["qc_float_var"] = np.random.uniform(0, 1, size=adata.shape[1])

    adata.obs["qcvar1"] = np.random.normal(size=adata.shape[0])
    adata.obs["qcvar2"] = np.random.normal(size=adata.shape[0])

    sc.pp.normalize_total(adata, target_sum=None)
    sc.pp.log1p(adata)

    sc.tl.umap(adata, n_components=3)
    sc.tl.tsne(adata)
    # sc.tl.pca(adata)
    sc.tl.rank_genes_groups(adata, groupby='clustering', method='t-test_overestim_var', n_genes=250)
    # sc.tl.dendrogram(adata, groupby='clustering')

    return adata


@pytest.fixture
def venn_dict():
    """Create arbitrary groups for venn."""
    return {"Group A": [1, 2, 3, 4, 5, 6],
            "Group B": [2, 3, 7, 8],
            "Group C": [3, 4, 5, 9, 10]}


# ------------------------------ TESTS --------------------------------- #


@pytest.mark.parametrize("color", [["clustering", "condition"], "clustering"])
def test_add_figure_title_axis(adata, color):
    """Test if function _add_figure_title runs with axis object(s) as input."""
    axes = sc.pl.umap(adata, color=color, show=False)
    pl._add_figure_title(axes, "UMAP plots", fontsize=20)
    assert True


def test_add_figure_title_axis_dict(adata):
    """Test if function _add_figure_title runs with axis dict as input."""
    markers = ['ENSMUSG00000103377', 'ENSMUSG00000102851']
    axes = sc.pl.dotplot(adata, markers, groupby='condition',
                         dendrogram=True, show=False)
    pl._add_figure_title(axes, "Dotplot", fontsize=20)
    assert True


def test_add_figure_title_axis_clustermap(adata):
    """Test if function _add_figure_title runs with clustermap as input."""
    clustermap = sns.clustermap(adata.obs[['LISI_score_pca', 'qc_float']])
    pl._add_figure_title(clustermap, "Heatmap", fontsize=20)
    assert True


@pytest.mark.parametrize("label", [None, "label"])
def test_add_labels(df, label):
    """Test _add_labels success."""
    if label:
        df["label"] = ["A", "B", "C", "D", "E"]
    texts = pl._add_labels(df, x="col1", y="col2", label_col=label)
    assert isinstance(texts, list)
    assert type(texts[0]).__name__ == "Annotation"


def test_clustermap_dotplot():
    """Test clustermap_dotplot success."""
    table = sc.datasets.pbmc68k_reduced().obs.reset_index()[:10]
    axes = pl.clustermap_dotplot(table=table, x="bulk_labels",
                                 y="index", hue="n_genes",
                                 size="n_counts", palette="viridis",
                                 title="Title", show_grid=True)

    assert isinstance(axes, np.ndarray)
    ax_type = type(axes[0]).__name__
    assert ax_type.startswith("Axes")


def test_bidirectional_barplot(df_bidir_bar):
    """Test bidirectoional_barplot success."""
    pl.bidirectional_barplot(df_bidir_bar, title="Title")
    assert True


def test_bidirectional_barplot_fail(df):
    """Test bidorectional_barplot with invalid input."""
    with pytest.raises(KeyError, match='Column left_label not found in dataframe.'):
        pl.bidirectional_barplot(df)


def test_boxplot(df):
    """Test if Axes object is returned."""
    ax = pl.boxplot(df)
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("ylabel,color_by,hlines", [(True, None, 0.5),
                                                    (False, "clustering", [0.5, 0.5, 0.5, 0.5])])
def test_violinplot(adata, ylabel, color_by, hlines):
    """Test violinplot success."""
    ax = pl.violinplot(adata.obs, "qc_float", color_by=color_by,
                       hlines=hlines, colors=None, ax=None,
                       title="Title", ylabel=ylabel)
    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")


def test_violinplot_fail(adata):
    """Test invalid input for violinplot."""
    with pytest.raises(ValueError, match='not found in column names of table!'):
        pl.violinplot(adata.obs, y="Invalid")

    with pytest.raises(ValueError, match='Color grouping'):
        pl.violinplot(adata.obs, y="qc_float", color_by="Invalid")

    with pytest.raises(ValueError, match='Parameter hlines has to be number or list'):
        pl.violinplot(adata.obs, y="qc_float", hlines={"A": 0.5})

    with pytest.raises(ValueError, match='Invalid dict keys in hlines parameter.'):
        pl.violinplot(adata.obs, y="qc_float",
                      color_by="clustering", hlines={"A": 0.5})


def test_plot_venn(venn_dict):
    """Test plot_venn with 3 and 2 groups."""
    pl.plot_venn(venn_dict, title="Test")
    venn_dict.pop("Group C")
    pl.plot_venn(venn_dict, title="Test")
    assert True


def test_plot_venn_fail(venn_dict):
    """Test for invalid input."""
    venn_dict["Group D"] = [1, 2]
    with pytest.raises(ValueError):
        pl.plot_venn(venn_dict)

    with pytest.raises(BeartypeCallHintParamViolation):
        pl.plot_venn([1, 2, 3, 4, 5])


@pytest.mark.parametrize("columns", [["invalid"], ["not", "present"]])
def test_pairwise_scatter_invalid(adata, columns):
    """Test that invalid columns raise error."""
    with pytest.raises(ValueError):
        pl.pairwise_scatter(adata.obs, columns=columns)

    with pytest.raises(BeartypeCallHintParamViolation):
        pl.pairwise_scatter(adata.obs, columns="invalid")


@pytest.mark.parametrize("thresholds", [None,
                                        {"qcvar1": {"min": 0.1}, "qcvar2": {"min": 0.4}}])
def test_pairwise_scatter(adata, thresholds):
    """Test pairwise scatterplot with different input."""
    axarr = pl.pairwise_scatter(adata.obs, columns=["qcvar1", "qcvar2"], thresholds=thresholds)

    assert axarr.shape == (2, 2)
    assert type(axarr[0, 0]).__name__.startswith("Axes")


@pytest.mark.parametrize("ax, crop", [(None, None), (plt.subplots()[1], 2)])
def test_plot_table(df, ax):
    """Test plot_table with and without predefined ax."""
    ax = pl.plot_table(table=df, crop=crop)

    assert isinstance(ax, plt.Axes)
