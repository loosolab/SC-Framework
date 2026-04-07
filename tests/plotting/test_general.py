"""Test general plotting functions."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import sctoolbox.plotting.general as pl

from beartype.roar import BeartypeCallHintParamViolation


# ------------------------------ TESTS --------------------------------- #


@pytest.mark.parametrize("color", [["louvain", "condition"], "louvain"])
def test_add_figure_title_axis(adata, color):
    """Test if function _add_figure_title runs with axis object(s) as input."""
    axes = sc.pl.umap(adata, color=color, show=False)
    pl._add_figure_title(axes, "UMAP plots", fontsize=20)
    assert True


def test_add_figure_title_axis_dict(adata):
    """Test if function _add_figure_title runs with axis dict as input."""
    markers = list(adata.var.index[:2])  # get the first two gene names
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
                                                    (False, "louvain", [0.5, 0.5, 0.5, 0.5])])
def test_violinplot(adata, ylabel, color_by, hlines):
    """Test violinplot success."""
    ax = pl.violinplot(adata.obs, "qc_float", color_by=color_by,
                       hlines=hlines, colors=None, ax=None,
                       title="Title", ylabel=ylabel)
    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("match,y,color_by,hlines", [
    ('not found in column names of table!', 'Invalid', None, None),
    ('Color grouping', 'qc_float', 'Invalid', None),
    ('Parameter hlines has to be number or list', 'qc_float', None, {"A": 0.5}),
    ('Invalid dict keys in hlines parameter.', 'qc_float', 'louvain', {"A": 0.5}),
])
def test_violinplot_fail(adata, match, y, color_by, hlines):
    """Test invalid input for violinplot."""

    with pytest.raises(ValueError, match=match):
        pl.violinplot(adata.obs, y=y,
                      color_by=color_by, hlines=hlines)


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
def test_plot_table(df, ax, crop):
    """Test plot_table with and without predefined ax."""
    ax = pl.plot_table(table=df, crop=crop)

    assert isinstance(ax, plt.Axes)
