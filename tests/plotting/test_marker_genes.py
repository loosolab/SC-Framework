"""Test plotting functions."""

import pytest
import matplotlib
import sctoolbox.plotting.marker_genes as pl
import matplotlib.pyplot as plt

from beartype.roar import BeartypeCallHintParamViolation

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")


# ------------------------------ TESTS --------------------------------- #


@pytest.mark.parametrize("dendrogram,use_genes,key,swap_axes",
                         [(True, True, None, True),
                          (False, False, 'rank_genes_groups', False)])
@pytest.mark.parametrize("style", ["dots", "heatmap"])
def test_rank_genes_plot(adata, style, dendrogram, use_genes, key, swap_axes):
    """Test rank_genes_plot for ranked genes and gene lists."""
    genes = adata.var_names[:3].tolist() if use_genes else None
    d = pl.rank_genes_plot(adata, groupby="louvain",
                           genes=genes, key=key,
                           style=style, title="Test",
                           dendrogram=dendrogram,
                           swap_axes=swap_axes)
    assert isinstance(d, dict)


@pytest.mark.parametrize("kwargs,exception,match", [
    ({"groupby": "louvain", "key": "rank_genes_groups", "style": "Invalid"},
     BeartypeCallHintParamViolation, None),
    ({"groupby": "louvain", "key": "rank_genes_groups", "genes": ["A", "B", "C"]},  # genes not in dataset
     KeyError, "Could not find keys.*"),
    ({"groupby": None, "genes": "<VAR_NAMES[:2]>"},
     ValueError, "The parameter 'groupby' is needed if 'genes' is given."),
])
def test_rank_genes_plot_fail(adata, kwargs, exception, match):
    """Test rank_genes_plot for invalid input."""
    kwargs = {k: (adata.var_names[:2].tolist() if v == "<VAR_NAMES[:2]>" else v)
              for k, v in kwargs.items()}
    with pytest.raises(exception, match=match):
        pl.rank_genes_plot(adata, **kwargs)


@pytest.mark.parametrize("x,y,norm", [("louvain", "<GENE_1>", True),
                                      ("<GENE_0>", None, False),
                                      ("louvain", "qc_float", True)])
@pytest.mark.parametrize("style", ["violin", "boxplot", "bar"])
def test_grouped_violin(adata, x, y, norm, style):
    """Test grouped_violin success."""
    x = adata.var_names[0] if x == "<GENE_0>" else x
    y = adata.var_names[1] if y == "<GENE_1>" else y
    ax = pl.grouped_violin(adata, x=x, y=y, style=style,
                           groupby="condition", normalize=norm)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize("kwargs,exception,match", [
    ({"x": "Invalid", "y": None, "groupby": "condition"}, ValueError, 'is not a column in adata.obs or a gene in adata.var.index'),
    ({"x": ["louvain", "SRM"], "y": None, "groupby": "condition"}, ValueError, 'x must be either a column in adata.obs or all genes in adata.var.index'),
    ({"x": "louvain", "y": "Invalid", "groupby": "condition"}, ValueError, 'was not found in either adata.obs or adata.var.index'),
    ({"x": "louvain", "y": None, "groupby": "condition"}, ValueError, "Because 'x' is a column in obs, 'y' must be given as parameter"),
    ({"x": "<GENE_0>", "y": None, "groupby": "condition", "style": "Invalid"}, BeartypeCallHintParamViolation, None),
])
def test_grouped_violin_fail(adata, kwargs, exception, match):
    """Test grouped_violin fail."""
    kwargs = {k: (adata.var_names[0] if v == "<GENE_0>" else v) for k, v in kwargs.items()}
    with pytest.raises(exception, match=match):
        pl.grouped_violin(adata, **kwargs)


def test_group_expression_boxplot(adata):
    """Test if group_expression_boxplot returns a plot."""
    gene_list = adata.var_names.tolist()[:10]
    ax = pl.group_expression_boxplot(adata, gene_list, groupby="condition")
    assert isinstance(ax, matplotlib.axes.Axes)


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
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_differential_genes_fail(pairwise_ranked_genes_nosig):
    """Test if ValueError is raised if no significant genes are found."""
    with pytest.raises(ValueError, match='No significant differentially expressed genes in the data. Abort.'):
        pl.plot_differential_genes(pairwise_ranked_genes_nosig)


@pytest.mark.parametrize("use_list,save,figsize",
                         [(True, None, (2, 2)),
                          (False, "out.png", None)])
def test_plot_gene_correlation(adata, use_list, save, figsize, assert_axes_array, tmp_path):
    """Test gene correlation."""
    gene_list = adata.var_names[1:4].tolist() if use_list else adata.var_names[1]
    save_path = str(tmp_path / save) if save else None
    axes = pl.plot_gene_correlation(adata, adata.var_names[0], gene_list,
                                    save=save_path, figsize=figsize)
    assert_axes_array(axes)
