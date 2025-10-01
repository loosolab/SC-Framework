"""Test plotting functions."""

import pytest
import sctoolbox.plotting.marker_genes as pl
import scanpy as sc
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from beartype.roar import BeartypeCallHintParamViolation

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")


# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """Load and returns an anndata object."""

    np.random.seed(1)  # set seed for reproducibility

    adata = sc.datasets.pbmc3k_processed()
    adata.raw = None

    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
    adata.obs["clustering"] = np.random.choice(["1", "2", "3", "4"], size=adata.shape[0])
    adata.obs["cat"] = adata.obs["condition"].astype("category")

    adata.obs["LISI_score_pca"] = np.random.normal(size=adata.shape[0])
    adata.obs["qc_float"] = np.random.uniform(0, 1, size=adata.shape[0])
    adata.var["qc_float_var"] = np.random.uniform(0, 1, size=adata.shape[1])

    adata.obs["qcvar1"] = np.random.normal(size=adata.shape[0])
    adata.obs["qcvar2"] = np.random.normal(size=adata.shape[0])

    # sc.pp.normalize_total(adata, target_sum=None)
    # sc.pp.log1p(adata)

    # sc.tl.umap(adata, n_components=3)  # to have more than two components available
    # sc.tl.tsne(adata)
    # sc.tl.pca(adata)
    sc.tl.rank_genes_groups(adata, groupby='clustering', method='t-test_overestim_var', n_genes=250)
    # sc.tl.dendrogram(adata, groupby='clustering')

    return adata


@pytest.fixture
def pairwise_ranked_genes():
    """Return a DataFrame of genes ranked in groups."""
    return pd.DataFrame(data={"1/2_group": ["C1", "C1", "C2", "C2"],
                              "1/3_group": ["C1", "NS", "C2", "C2"],
                              "2/3_group": ["C1", "C1", "NS", "C2"]},
                        index=["GeneA", "GeneB", "GeneC", "GeneD"])


@pytest.fixture
def pairwise_ranked_genes_nosig():
    """Return a DataFrame of genes ranked in groups with none significant."""
    return pd.DataFrame(data={"1/2_group": ["NS", "NS", "NS", "NS"],
                              "1/3_group": ["NS", "NS", "NS", "NS"],
                              "2/3_group": ["NS", "NS", "NS", "NS"]},
                        index=["GeneA", "GeneB", "GeneC", "GeneD"])


# ------------------------------ TESTS --------------------------------- #
@pytest.mark.parametrize("dendrogram,genes,key,swap_axes",
                         [(True, ['TNFRSF4', 'CPSF3L', 'ATAD3C'], None, True),
                          (False, None, 'rank_genes_groups', False)])
@pytest.mark.parametrize("style", ["dots", "heatmap"])
def test_rank_genes_plot(adata, style, dendrogram, genes, key, swap_axes):
    """Test rank_genes_plot for ranked genes and gene lists."""
    # Gene list
    d = pl.rank_genes_plot(adata, groupby="clustering",
                           genes=genes, key=key,
                           style=style, title="Test",
                           dendrogram=dendrogram,
                           swap_axes=swap_axes)
    assert isinstance(d, dict)


def test_rank_genes_plot_fail(adata):
    """Test rank_genes_plot for invalid input."""
    with pytest.raises(BeartypeCallHintParamViolation):
        pl.rank_genes_plot(adata, groupby="clustering",
                           key='rank_genes_groups',
                           style="Invalid")
    with pytest.raises(KeyError, match='Could not find keys.*'):
        pl.rank_genes_plot(adata, groupby="clustering",
                           key='rank_genes_groups',
                           genes=["A", "B", "C"])  # invalid genes given
    with pytest.raises(ValueError, match="The parameter 'groupby' is needed if 'genes' is given."):
        pl.rank_genes_plot(adata, groupby=None,
                           genes=['RER1', 'TNFRSF25'])


@pytest.mark.parametrize("x,y,norm", [("clustering", "C1orf86", True),
                                      ("SRM", None, False),
                                      ("clustering", "qc_float", True)])
@pytest.mark.parametrize("style", ["violin", "boxplot", "bar"])
def test_grouped_violin(adata, x, y, norm, style):
    """Test grouped_violin success."""

    ax = pl.grouped_violin(adata, x=x, y=y, style=style,
                           groupby="condition", normalize=norm)
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


def test_grouped_violin_fail(adata):
    """Test grouped_violin fail."""

    with pytest.raises(ValueError, match='is not a column in adata.obs or a gene in adata.var.index'):
        pl.grouped_violin(adata, x="Invalid", y=None, groupby="condition")
    with pytest.raises(ValueError, match='x must be either a column in adata.obs or all genes in adata.var.index'):
        pl.grouped_violin(adata, x=["clustering", "UBIAD1"], y=None, groupby="condition")
    with pytest.raises(ValueError, match='was not found in either adata.obs or adata.var.index'):
        pl.grouped_violin(adata, x="clustering", y="Invalid", groupby="condition")
    with pytest.raises(ValueError, match="Because 'x' is a column in obs, 'y' must be given as parameter"):
        pl.grouped_violin(adata, x="clustering", y=None, groupby="condition")
    with pytest.raises(BeartypeCallHintParamViolation):
        pl.grouped_violin(adata, x="PRDM2", y=None, groupby="condition", style="Invalid")


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
                                   cluster_column="clustering")
    assert type(g).__name__ == "ClusterGrid"


@pytest.mark.parametrize("kwargs, exception",
                         [({"gene_name_column": "invalid"}, KeyError)])
def test_gene_expression_heatmap_error(adata, kwargs, exception):
    """Test gene_expression_heatmap failure."""

    genes = adata.var_names.tolist()[:10]
    with pytest.raises(exception):
        pl.gene_expression_heatmap(adata, genes=genes, cluster_column="clustering", **kwargs)


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
                         [(['EFHD2', 'DDI2', 'SPEN'], None, (2, 2)),
                          ("SDHB", "out.png", None)])
def test_plot_gene_correlation(adata, gene_list, save, figsize):
    """Test gene correlation."""
    axes = pl.plot_gene_correlation(adata, "TNFRSF4", gene_list,
                                    save=save, figsize=figsize)
    assert type(axes).__name__ == "ndarray"
    assert type(axes[0]).__name__.startswith("Axes")

    if save:
        os.remove(save)
