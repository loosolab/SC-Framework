"""Test plotting functions."""

import pytest
import sctoolbox.plotting.marker_genes as pl
import os
import matplotlib.pyplot as plt

from beartype.roar import BeartypeCallHintParamViolation

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")


# ------------------------------ TESTS --------------------------------- #


@pytest.mark.parametrize("dendrogram,genes,key,swap_axes",
                         [(True, ['TNFRSF4', 'SSU72', 'PARK7'], None, True),
                          (False, None, 'rank_genes_groups', False)])
@pytest.mark.parametrize("style", ["dots", "heatmap"])
def test_rank_genes_plot(adata, style, dendrogram, genes, key, swap_axes):
    """Test rank_genes_plot for ranked genes and gene lists."""
    # Gene list
    d = pl.rank_genes_plot(adata, groupby="louvain",
                           genes=genes, key=key,
                           style=style, title="Test",
                           dendrogram=dendrogram,
                           swap_axes=swap_axes)
    assert isinstance(d, dict)


def test_rank_genes_plot_fail(adata):
    """Test rank_genes_plot for invalid input."""
    with pytest.raises(BeartypeCallHintParamViolation):
        pl.rank_genes_plot(adata, groupby="louvain",
                           key='rank_genes_groups',
                           style="Invalid")
    with pytest.raises(KeyError, match='Could not find keys.*'):
        pl.rank_genes_plot(adata, groupby="louvain",
                           key='rank_genes_groups',
                           genes=["A", "B", "C"])  # invalid genes given
    with pytest.raises(ValueError, match="The parameter 'groupby' is needed if 'genes' is given."):
        pl.rank_genes_plot(adata, groupby=None,
                           genes=['TNFRSF4', 'SRM'])


@pytest.mark.parametrize("x,y,norm", [("louvain", "EFHD2", True),
                                      ("SRM", None, False),
                                      ("louvain", "qc_float", True)])
@pytest.mark.parametrize("style", ["violin", "boxplot", "bar"])
def test_grouped_violin(adata, x, y, norm, style):
    """Test grouped_violin success."""

    ax = pl.grouped_violin(adata, x=x, y=y, style=style,
                           groupby="condition", normalize=norm)
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("kwargs,exception,match", [
    ({"x": "Invalid", "y": None, "groupby": "condition"}, ValueError, 'is not a column in adata.obs or a gene in adata.var.index'),
    ({"x": ["louvain", "SRM"], "y": None, "groupby": "condition"}, ValueError, 'x must be either a column in adata.obs or all genes in adata.var.index'),
    ({"x": "louvain", "y": "Invalid", "groupby": "condition"}, ValueError, 'was not found in either adata.obs or adata.var.index'),
    ({"x": "louvain", "y": None, "groupby": "condition"}, ValueError, "Because 'x' is a column in obs, 'y' must be given as parameter"),
    ({"x": "SRM", "y": None, "groupby": "condition", "style": "Invalid"}, BeartypeCallHintParamViolation, None),
])
def test_grouped_violin_fail(adata, kwargs, exception, match):
    """Test grouped_violin fail."""
    with pytest.raises(exception, match=match):
        pl.grouped_violin(adata, **kwargs)


def test_group_expression_boxplot(adata):
    """Test if group_expression_boxplot returns a plot."""
    gene_list = adata.var_names.tolist()[:10]
    ax = pl.group_expression_boxplot(adata, gene_list, groupby="condition")
    ax_type = type(ax).__name__

    # depending on matplotlib version, it can be either AxesSubplot or Axes
    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("groupby, title",
                         [(None, "title"),
                          ("condition", None)])
def test_gene_expression_heatmap(adata, title, groupby):
    """Test gene_expression_heatmap success."""

    genes = adata.var_names.tolist()[:10]
    g = pl.gene_expression_heatmap(adata,
                                   genes=genes,
                                   groupby=groupby, title=title,
                                   col_cluster=True,            # ensure title is tested
                                   show_col_dendrogram=True,    # ensure title is tested
                                   cluster_column="louvain")
    assert type(g).__name__ == "ClusterGrid"


@pytest.mark.parametrize("kwargs, exception",
                         [({"gene_name_column": "invalid"}, KeyError)])
def test_gene_expression_heatmap_error(adata, kwargs, exception):
    """Test gene_expression_heatmap failure."""

    genes = adata.var_names.tolist()[:10]
    with pytest.raises(exception):
        pl.gene_expression_heatmap(adata, genes=genes, cluster_column="louvain", **kwargs)


def test_plot_differential_genes(pairwise_ranked_genes):
    """Test plot_differential_genes success."""
    ax = pl.plot_differential_genes(pairwise_ranked_genes)
    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")


def test_plot_differential_genes_fail(pairwise_ranked_genes_nosig):
    """Test if ValueError is raised if no significant genes are found."""
    with pytest.raises(ValueError, match='No significant differentially expressed genes in the data. Abort.'):
        pl.plot_differential_genes(pairwise_ranked_genes_nosig)


@pytest.mark.parametrize("gene_list,save,figsize",
                         [(['EFHD2', 'SSU72', 'PARK7'], None, (2, 2)),
                          ("SRM", "out.png", None)])
def test_plot_gene_correlation(adata, gene_list, save, figsize):
    """Test gene correlation."""
    axes = pl.plot_gene_correlation(adata, "TNFRSF4", gene_list,
                                    save=save, figsize=figsize)
    assert type(axes).__name__ == "ndarray"
    assert type(axes[0]).__name__.startswith("Axes")

    if save:
        os.remove(save)
