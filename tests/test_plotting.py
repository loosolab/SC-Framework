import pytest
import sctoolbox.plotting as pl
import scanpy as sc
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import sctoolbox.tools as tool
import seaborn as sns
import ipywidgets as widgets
import functools
import matplotlib.pyplot as plt


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")
    adata = sc.read_h5ad(f)
    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
    adata.obs["clustering"] = np.random.choice(["1", "2", "3", "4"], size=adata.shape[0])
    adata.obs["cat"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
    adata.obs["cat"] = adata.obs["cat"].astype("category")
    adata.obs["LISI_score_pca"] = np.random.normal(size=adata.shape[0])
    adata.obs["qc_float"] = np.random.uniform(0, 1, size=adata.shape[0])
    adata.var["qc_float_var"] = np.random.uniform(0, 1, size=adata.shape[1])

    sc.pp.normalize_total(adata, target_sum=None)
    sc.pp.log1p(adata)

    sc.tl.umap(adata, n_components=3)
    sc.tl.tsne(adata)
    sc.tl.pca(adata)
    sc.tl.rank_genes_groups(adata, groupby='clustering', method='t-test_overestim_var', n_genes=250)
    sc.tl.dendrogram(adata, groupby='clustering')

    return adata


@pytest.fixture
def df():
    """Create and return a pandas dataframe"""
    return pd.DataFrame(data={'col1': [1, 2, 3, 4, 5],
                              'col2': [3, 4, 5, 6, 7]})


@pytest.fixture
def pairwise_ranked_genes():
    return pd.DataFrame(data={"1/2_group": ["C1", "C1", "C2", "C2"],
                              "1/3_group": ["C1", "NS", "C2", "C2"],
                              "2/3_group": ["C1", "C1", "NS", "C2"]},
                        index=["GeneA", "GeneB", "GeneC", "GeneD"])


@pytest.fixture
def df_bidir_bar():
    return pd.DataFrame(data={'left_label': np.random.choice(["B1", "B2", "B3"], size=5),
                              'right_label': np.random.choice(["A1", "A2", "A3"], size=5),
                              'left_value': np.random.normal(size=5),
                              'right_value': np.random.normal(size=5)})


@pytest.fixture
def venn_dict():
    return {"Group A": [1, 2, 3, 4, 5, 6],
            "Group B": [2, 3, 7, 8],
            "Group C": [3, 4, 5, 9, 10]}


@pytest.fixture
def tmp_file():
    """ Return path for a temporary file. """
    tmpdir = tempfile.mkdtemp()

    yield os.path.join(tmpdir, "output.pdf")

    # clean up directory and contents
    shutil.rmtree(tmpdir)


@pytest.fixture
def slider():
    return widgets.FloatSlider(value=7.5, min=0, max=10.0, step=0.1)


@pytest.fixture
def slider_list(slider):
    return [slider for _ in range(2)]


@pytest.fixture
def checkbox():
    return widgets.Checkbox()


# ------------------------------ TESTS --------------------------------- #


def test_sc_colormap():
    """ Test whether sc_colormap returns a colormap """

    cmap = pl.sc_colormap()
    assert type(cmap).__name__ == "ListedColormap"


@pytest.mark.parametrize("n_selected", [None, 1, 2])
def test_plot_pca_variance(adata, n_selected):
    """ Test if Axes object is returned. """
    ax = pl.plot_pca_variance(adata, n_selected=n_selected)
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


def test_plot_pca_variance_fail(adata):
    """ Test if failes on invalid method. """
    # generate invalid method
    invalid = "-".join(list(adata.uns.keys())) + "-invalid"

    with pytest.raises(KeyError):
        pl.plot_pca_variance(adata, method=invalid)


@pytest.mark.parametrize("method", ["umap"])  # , "tsne"]) # tsne option is currently broken and sends the function to sleep. Will be added if fixed.
def test_search_dim_red_parameters(adata, method):
    """ Test if search_dim_red_parameters returns an array of axes. """

    axarr = pl._search_dim_red_parameters(adata, color="condition",
                                          method=method,
                                          min_dist_range=(0.1, 0.3, 0.1),
                                          spread_range=(2.0, 3.0, 0.5),
                                          learning_rate_range=(100, 300, 100),
                                          perplexity_range=(20, 30, 5))
    assert type(axarr).__name__ == "ndarray"
    assert axarr.shape == (2, 2)


def test_invalid_method_search_dim_red_parameter(adata):
    with pytest.raises(ValueError):
        pl._search_dim_red_parameters(adata, color="condition",
                                      method="invalid")


@pytest.mark.parametrize("range", [(0.1, 0.2, 0.1, 0.1), (0.1, 0.2, 0.3)])
def test_search_dim_red_parameters_ranges(adata, range):
    """ Test that invalid ranges raise ValueError. """

    with pytest.raises(ValueError):
        pl._search_dim_red_parameters(adata, method="umap",
                                      color="condition",
                                      min_dist_range=range,
                                      spread_range=(2.0, 3.0, 0.5))

    with pytest.raises(ValueError):
        pl._search_dim_red_parameters(adata, method="umap",
                                      color="condition",
                                      spread_range=range,
                                      min_dist_range=(0.1, 0.3, 0.1))


@pytest.mark.parametrize("embedding", ["pca", "umap", "tsne"])
def test_plot_group_embeddings(adata, embedding):

    axarr = pl.plot_group_embeddings(adata, groupby="condition",
                                     embedding=embedding, ncols=2)

    assert axarr.shape == (2, 2)


@pytest.mark.parametrize("embedding, var_list", [("pca", "list"),
                                                 ("umap", "condition"),
                                                 ("tsne", "list")])
def test_compare_embeddings(adata, embedding, var_list):

    adata_cp = adata.copy()

    # check that function can deal with missing vars
    adata_cp.obs.drop(columns=["condition"], inplace=True)

    adata_list = [adata, adata_cp]
    if var_list == "list":
        # notfound will be excluded
        var_list = [adata.var.index[0], "condition", "notfound"]
        res = (2, 2)
    else:
        res = (1, 2)
    axarr = pl.compare_embeddings(adata_list, var_list, embedding=embedding)

    assert axarr.shape == res


def test_invalid_var_list_compare_embeddings(adata):
    with pytest.raises(ValueError):
        adata_cp = adata.copy()
        adata_list = [adata, adata_cp]
        pl.compare_embeddings(adata_list, ["invalid_1", "invalid_2"],
                              embedding="umap")


@pytest.mark.parametrize("method", ["leiden", "louvain"])
def test_search_clustering_parameters(adata, method):
    """ Test if search_clustering_parameters returns an array of axes. """

    axarr = pl.search_clustering_parameters(adata, method=method,
                                            resolution_range=(0.1, 0.31, 0.1),
                                            ncols=2)
    assert type(axarr).__name__ == "ndarray"
    assert axarr.shape == (2, 2)


def test_wrong_embeding_search_clustering_parameters(adata):
    with pytest.raises(KeyError):
        pl.search_clustering_parameters(adata, embedding="Invalid")


@pytest.mark.parametrize("method,resrange", [("leiden", (0.1, 0.2, 0.1, 0.1)),
                                             ("leiden", (0.1, 0.2, 0.3)),
                                             ("unknown", (0.1, 0.3, 0.1))])
def test_search_clustering_parameters_errors(adata, method, resrange):

    with pytest.raises(ValueError):
        pl.search_clustering_parameters(adata, resolution_range=resrange,
                                        method=method)


def test_anndata_overview(adata, tmp_file):
    """ Test anndata_overview success and file generation. """
    adatas = {"raw": adata, "corrected": adata}

    assert not os.path.exists(tmp_file)

    pl.anndata_overview(
        adatas=adatas,
        color_by=list(adata.obs.columns) + [adata.var_names.tolist()[0]],
        plots=["PCA", "PCA-var", "UMAP", "tSNE", "LISI"],
        figsize=None,
        output=None,
        dpi=300
    )

    assert not os.path.exists(tmp_file)

    pl.anndata_overview(
        adatas=adatas,
        color_by=list(adata.obs.columns),
        plots=["PCA"],
        figsize=None,
        output=tmp_file,
        dpi=300
    )

    assert os.path.exists(tmp_file)


def test_anndata_overview_fail_color_by(adata):
    """ Test invalid parameter inputs """
    adatas = {"raw": adata}

    # invalid color_by
    # no input
    with pytest.raises(ValueError, match="Couldn't find column"):
        pl.anndata_overview(
            adatas=adatas,
            color_by=None,
            plots=["PCA"],
            figsize=None,
            output=None,
            dpi=300
        )

    # wrong input
    with pytest.raises(ValueError, match="Couldn't find column"):
        pl.anndata_overview(
            adatas=adatas,
            color_by="-".join(list(adata.obs.columns)) + "-invalid",
            plots=["PCA"],
            figsize=None,
            output=None,
            dpi=300
        )


def test_anndata_overview_fail(adata):
    """ Test invalid parameter inputs """
    adatas_invalid = {"raw": adata, "invalid": "Not an anndata"}
    adata_cp = adata.copy()
    adata_cp.obs = adata_cp.obs.drop(["LISI_score_pca"], axis=1)
    adatas = {"raw": adata_cp}

    # invalid datatype
    with pytest.raises(ValueError, match="All items in 'adatas'"):
        pl.anndata_overview(
            adatas=adatas_invalid,
            color_by=list(adata.obs.columns) + [adata.var_names.tolist()[0]],
            plots=["PCA"],
            figsize=None,
            output=None,
            dpi=300
        )

    # Missing LISI score
    with pytest.raises(ValueError, match="No LISI scores found"):
        pl.anndata_overview(
            adatas=adatas,
            color_by=list(adata_cp.obs.columns) + [adata_cp.var_names.tolist()[0]],
            plots=["LISI"],
            figsize=None,
            output=None,
            dpi=300
        )


def test_anndata_overview_fail_plots(adata):
    """ Test invalid parameter inputs """
    adatas = {"raw": adata}

    # invalid plots
    # no input
    with pytest.raises(ValueError, match="Invalid plot specified:"):
        pl.anndata_overview(
            adatas=adatas,
            color_by=list(adata.obs.columns),
            plots=None,
            figsize=None,
            output=None,
            dpi=300
        )

    # wrong input
    with pytest.raises(ValueError, match="Invalid plot specified:"):
        pl.anndata_overview(
            adatas=adatas,
            color_by=list(adata.obs.columns),
            plots=["PCA", "invalid"],
            figsize=None,
            output=None,
            dpi=300
        )


def test_group_expression_boxplot(adata):
    """ Test if group_expression_boxplot returns a plot """
    gene_list = adata.var_names.tolist()[:10]
    ax = pl.group_expression_boxplot(adata, gene_list, groupby="condition")
    ax_type = type(ax).__name__

    # depending on matplotlib version, it can be either AxesSubplot or Axes
    assert ax_type.startswith("Axes")


def test_boxplot(df):
    """ Test if Axes object is returned. """
    ax = pl.boxplot(df)
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("color", ["ENSMUSG00000102693", "clustering", "qc_float"])
def test_plot_3D_UMAP(adata, color):
    """ Test if 3d plot is written to html """

    # Run 3d plotting
    pl.plot_3D_UMAP(adata, color=color, save="3D_test")

    # Assert creation of file
    assert os.path.isfile("3D_test.html")
    os.remove("3D_test.html")


def test_invalid_color_plot_3D_UMAP(adata):
    """ Test if plot_3D_UMAP return KeyError if color paramater cannot be found in adata"""
    with pytest.raises(KeyError):
        pl.plot_3D_UMAP(adata, color="invalid", save="3D_test")


def test_group_correlation(adata):
    """ Test if plot is written to pdf """

    # Run group correlation
    pl.group_correlation(adata, groupby="condition",
                         save="group_correlation.pdf")

    # Assert creation of file
    assert os.path.isfile("group_correlation.pdf")
    os.remove("group_correlation.pdf")


@pytest.mark.parametrize("groupby", [None, "condition"])
@pytest.mark.parametrize("add_labels", [True, False])
def test_n_cells_barplot(adata, groupby, add_labels):

    axarr = pl.n_cells_barplot(adata, "clustering", groupby=groupby,
                               add_labels=add_labels)

    if groupby is None:
        assert len(axarr) == 1
    else:
        assert len(axarr) == 2


@pytest.mark.parametrize("x,y,norm", [("clustering", "ENSMUSG00000102693", True),
                                      ("ENSMUSG00000102693", None, False),
                                      ("clustering", "qc_float", True)])
@pytest.mark.parametrize("style", ["violin", "boxplot", "bar"])
def test_grouped_violin(adata, x, y, norm, style):

    ax = pl.grouped_violin(adata, x=x, y=y, style=style,
                           groupby="condition", normalize=norm)
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


def test_grouped_violin_fail(adata):
    with pytest.raises(ValueError, match='is not a column in adata.obs or a gene in adata.var.index'):
        pl.grouped_violin(adata, x="Invalid", y=None, groupby="condition")
    with pytest.raises(ValueError, match='x must be either a column in adata.obs or all genes in adata.var.index'):
        pl.grouped_violin(adata, x=["clustering", "ENSMUSG00000102693"], y=None, groupby="condition")
    with pytest.raises(ValueError, match='was not found in either adata.obs or adata.var.index'):
        pl.grouped_violin(adata, x="clustering", y="Invalid", groupby="condition")
    with pytest.raises(ValueError, match="Because 'x' is a column in obs, 'y' must be given as parameter"):
        pl.grouped_violin(adata, x="clustering", y=None, groupby="condition")
    with pytest.raises(ValueError, match="Style 'Invalid' is not valid for this function."):
        pl.grouped_violin(adata, x="ENSMUSG00000102693", y=None, groupby="condition", style="Invalid")


@pytest.mark.parametrize("show_umap", [True, False])
def test_marker_gene_clustering(adata, show_umap):
    """ Test marker_gene_clustering"""

    marker_dict = {"Celltype A": ['ENSMUSG00000103377', 'ENSMUSG00000104428'],
                   "Celltype B": ['ENSMUSG00000102272']}

    axes_list = pl.marker_gene_clustering(adata, "condition",
                                          marker_dict, show_umap=show_umap)
    assert isinstance(axes_list, list)
    ax_type = type(axes_list[0]).__name__
    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("how", ["vertical", "horizontal"])
def test_flip_embedding(adata, how):
    tmp = adata.copy()
    key = "X_umap"
    pl.flip_embedding(adata, key=key, how=how)

    if how == "vertical":
        assert all(adata.obsm[key][:, 1] == -tmp.obsm[key][:, 1])
    elif how == "horizontal":
        assert all(adata.obsm[key][:, 0] == -tmp.obsm[key][:, 0])


def test_invalid_flip_embedding(adata):
    with pytest.raises(ValueError):
        pl.flip_embedding(adata, how="invalid")

    with pytest.raises(KeyError):
        pl.flip_embedding(adata, key="invalid")


@pytest.mark.parametrize("n, res", [(500, 12), (1000, 8),
                                    (5000, 8), (10000, 3), (20000, 3)])
def test_get_3d_dotsize(n, res):
    assert pl._get_3d_dotsize(int(n)) == res


@pytest.mark.parametrize("marker", ["ENSMUSG00000103377",
                                    ["ENSMUSG00000103377", 'ENSMUSG00000104428']])
def test_umap_marker_overview(adata, marker):
    """ Test umap_marker_overview """
    axes_list = pl.umap_marker_overview(adata, marker)

    assert isinstance(axes_list, list)
    ax_type = type(axes_list[0]).__name__
    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("color,title", [("condition", "Condition"),
                                         (["condition", "clustering"], None),
                                         (["condition", "clustering"], ["Condition", "Clustering"])])
def test_umap_pub(adata, color, title):
    """ Test umap_pub plotting with different color and title parameter. """
    axes_list = pl.umap_pub(adata, color=color, title=title)

    assert type(axes_list).__name__ == "list"
    ax_type = type(axes_list[0]).__name__
    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("color,title", [("condition", ["Title 1", "Title 2"]),
                                         (["condition", "clustering"], "Title 1")])
def test_invalid_parameter_len_umap_pub(adata, color, title):
    """ Test case if color and title are not the same lenght """
    with pytest.raises(ValueError):
        pl.umap_pub(adata, color=color, title=title)


@pytest.mark.parametrize("color", [["clustering", "condition"], "clustering"])
def test_add_figure_title_axis(adata, color):
    """ Test if function _add_figure_title runs with axis object(s) as input """
    axes = sc.pl.umap(adata, color=color, show=False)
    pl._add_figure_title(axes, "UMAP plots", fontsize=20)
    assert True


def test_add_figure_title_axis_dict(adata):
    """ Test if function _add_figure_title runs with axis dict as input """
    markers = ['ENSMUSG00000103377', 'ENSMUSG00000102851']
    axes = sc.pl.dotplot(adata, markers, groupby='condition',
                         dendrogram=True, show=False)
    pl._add_figure_title(axes, "Dotplot", fontsize=20)
    assert True


def test_add_figure_title_axis_clustermap(adata):
    """ Test if function _add_figure_title runs with clustermap as input """
    clustermap = sns.clustermap(adata.obs[['LISI_score_pca', 'qc_float']])
    pl._add_figure_title(clustermap, "Heatmap", fontsize=20)
    assert True


@pytest.mark.parametrize("label", [None, "label"])
def test_add_labels(df, label):
    if label:
        df["label"] = ["A", "B", "C", "D", "E"]
    texts = pl._add_labels(df, x="col1", y="col2", label_col=label)
    assert isinstance(texts, list)
    assert type(texts[0]).__name__ == "Annotation"


@pytest.mark.parametrize("array,mini,maxi", [(np.array([1, 2, 3]), 0, 1),
                                             (np.array([[1, 2, 3], [1, 2, 3]]), 1, 100),
                                             (np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]]), 1, 5)])
def test_scale_values(array, mini, maxi):
    """ Test that scaled values are in given range. """
    result = pl._scale_values(array, mini, maxi)

    assert len(result) == len(array)
    if len(result.shape) == 1:
        assert all((mini <= result) & (result <= maxi))
    else:
        for i in range(len(result)):
            assert all((mini <= result[i]) & (result[i] <= maxi))


def test_clustermap_dotplot():
    """ Test clustermap_dotplot. """
    table = sc.datasets.pbmc68k_reduced().obs.reset_index()[:10]
    pl.clustermap_dotplot(table=table, x="bulk_labels",
                          y="index", color="n_genes",
                          size="n_counts", cmap="viridis",
                          vmin=0, vmax=10)
    assert True


def test_bidirectional_barplot(df_bidir_bar):
    """ Test bidirectoional_barplot. """
    pl.bidirectional_barplot(df_bidir_bar, title="Title")
    assert True


def test_bidirectional_barplot_fail(df):
    """ test bidorectional_barplot with invalid input. """
    with pytest.raises(ValueError):
        pl.bidirectional_barplot(df)


@pytest.mark.parametrize("ylabel,color_by,hlines", [(True, None, 0.5),
                                                    (False, "clustering", [0.5, 0.5, 0.5, 0.5])])
def test_violinplot(adata, ylabel, color_by, hlines):
    """ Test violinplot. """
    ax = pl.violinplot(adata.obs, "qc_float", color_by=color_by,
                       hlines=hlines, colors=None, ax=None,
                       title="Title", ylabel=ylabel)
    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")


def test_violinplot_fail(adata):
    """ Test invalid input for violinplot. """
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
    """ Test plot_venn with 3 and 2 groups. """
    pl.plot_venn(venn_dict, title="Test")
    venn_dict.pop("Group C")
    pl.plot_venn(venn_dict, title="Test")
    assert True


def test_plot_venn_fail(venn_dict):
    """ Test for invalid input. """
    venn_dict["Group D"] = [1, 2]
    with pytest.raises(ValueError):
        pl.plot_venn(venn_dict)

    with pytest.raises(ValueError):
        pl.plot_venn([1, 2, 3, 4, 5])


def test_violin_HVF_distribution(adata):
    """ Test violin_HVF_distribution. """
    adata_HVF = adata.copy()
    adata_HVF.var['highly_variable'] = np.random.choice([True, False], size=adata_HVF.shape[1])
    adata_HVF.var['n_cells_by_counts'] = np.random.normal(size=adata_HVF.shape[1])
    pl.violin_HVF_distribution(adata_HVF)
    assert True


def test_violin_HVF_distribution_fail(adata):
    """ Test if input is invalid. """
    with pytest.raises(KeyError):
        pl.violin_HVF_distribution(adata)


def test_scatter_HVF_distribution(adata):
    """ Test scatter_HVF_distribution. """
    adata_HVF = adata.copy()
    adata_HVF.var['variability_score'] = np.random.normal(size=adata_HVF.shape[1])
    adata_HVF.var['n_cells'] = np.random.normal(size=adata_HVF.shape[1])
    pl.scatter_HVF_distribution(adata_HVF)
    assert True


def test_scatter_HVF_distribution_fail(adata):
    """ Test if input is invalid. """
    with pytest.raises(KeyError):
        pl.scatter_HVF_distribution(adata)


@pytest.mark.parametrize("dendrogram,genes,key,swap_axes",
                         [(True, ['ENSMUSG00000102851',
                                  'ENSMUSG00000102272',
                                  'ENSMUSG00000101571'], None, True),
                          (False, None, 'rank_genes_groups', False)])
@pytest.mark.parametrize("style", ["dots", "heatmap"])
def test_rank_genes_plot(adata, style, dendrogram, genes, key, swap_axes):
    """ Test rank_genes_plot for ranked genes and gene lists. """
    # Gene list
    d = pl.rank_genes_plot(adata, groupby="clustering",
                           genes=genes, key=key,
                           style=style, title="Test",
                           dendrogram=dendrogram,
                           swap_axes=swap_axes)
    assert isinstance(d, dict)


def test_rank_genes_plot_fail(adata):
    """ Test rank_genes_plot for invalid input. """
    with pytest.raises(ValueError, match='style must be one of'):
        pl.rank_genes_plot(adata, groupby="clustering",
                           key='rank_genes_groups',
                           style="Invalid")
    with pytest.raises(ValueError, match='Only one of genes or key can be specified.'):
        pl.rank_genes_plot(adata, groupby="clustering",
                           key='rank_genes_groups',
                           genes=["A", "B", "C"])
    with pytest.raises(ValueError, match="The parameter 'groupby' is needed if 'genes' is given."):
        pl.rank_genes_plot(adata, groupby=None,
                           genes=['ENSMUSG00000102851', 'ENSMUSG00000102272'])


@pytest.mark.parametrize("groupby", [None, "condition"])
@pytest.mark.parametrize("title", [None, "Title"])
def test_gene_expression_heatmap(adata, title, groupby):
    """ Test gene_expression_heatmap. """
    g = pl.gene_expression_heatmap(adata,
                                   genes=['ENSMUSG00000102851',
                                          'ENSMUSG00000102272'],
                                   groupby=groupby, title=title,
                                   cluster_column="clustering")
    assert type(g).__name__ == "ClusterGrid"


@pytest.mark.parametrize("gene_list", [None, ['ENSMUSG00000102851',
                                              'ENSMUSG00000102272']])
@pytest.mark.parametrize("figsize", [None, (10, 10)])
def test_group_heatmap(adata, gene_list, figsize):
    """ Test group heatmap. """
    pl.group_heatmap(adata, "clustering", gene_list=gene_list,
                     figsize=figsize)


def test_plot_differential_genes(pairwise_ranked_genes):
    ax = pl.plot_differential_genes(pairwise_ranked_genes)
    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")


def test_plot_differential_genes_fail(adata):
    ranked_genes = tool.pairwise_rank_genes(adata, groupby="clustering")
    with pytest.raises(ValueError, match='No significant differentially expressed genes in the data. Abort.'):
        pl.plot_differential_genes(ranked_genes)


@pytest.mark.parametrize("sortby", [None, "condition"])
@pytest.mark.parametrize("title", [None, "condition"])
@pytest.mark.parametrize("figsize", [None, (10, 10)])
@pytest.mark.parametrize("layer", [None, "spliced"])
def test_pseudotime_heatmap(adata, sortby, title, figsize, layer):
    ax = pl.pseudotime_heatmap(adata, ['ENSMUSG00000103377',
                                       'ENSMUSG00000102851'],
                               sortby=sortby, title=title,
                               figsize=figsize, layer=layer)
    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")


def test_link_sliders(slider_list):
    linkage_list = pl._link_sliders(slider_list)
    assert isinstance(linkage_list, list)
    assert type(linkage_list[0]).__name__ == 'link'


@pytest.mark.parametrize("global_threshold", [True, False])
def test_toggle_linkage(checkbox, slider_list, global_threshold):
    """ Test if toggle_linkage runs without error. """
    column = "Test"
    linkage_dict = dict()
    linkage_dict[column] = pl._link_sliders(slider_list) if global_threshold is True else None
    checkbox.observe(functools.partial(pl._toggle_linkage,
                                       linkage_dict=linkage_dict,
                                       slider_list=slider_list,
                                       key=column), names=["value"])
    assert True


def test_update_threshold(slider):
    """ Test if update_threshold runs wihtout error. """
    fig, _ = plt.subplots()
    slider.observe(functools.partial(pl._update_thresholds, fig=fig,
                                     min_line=1, min_shade=1,
                                     max_line=1, max_shade=1),
                   names=["value"])
    assert True


@pytest.mark.parametrize("columns, which, groupby", [(['qc_float', 'LISI_score_pca'], "obs", "condition"),
                                                     (['qc_float', 'LISI_score_pca'], "obs", "cat"),
                                                     (['qc_float_var'], "var", None)])
@pytest.mark.parametrize("color_list", [None, sns.color_palette("Set1", 3)])
@pytest.mark.parametrize("title", [None, "Title"])
def test_quality_violin(adata, groupby, columns, which, title, color_list):
    figure, slider = pl.quality_violin(adata, columns=columns, groupby=groupby,
                                       which=which, title=title, color_list=color_list)
    assert type(figure).__name__ == "Figure"
    assert isinstance(slider, dict)


def test_quality_violin_fail(adata):
    with pytest.raises(ValueError, match="'which' must be either 'obs' or 'var'."):
        pl.quality_violin(adata, columns=["qc_float"], which="Invalid")
    with pytest.raises(ValueError, match="Increase the color_list variable"):
        pl.quality_violin(adata, groupby="condition", columns=["qc_float"],
                          color_list=sns.color_palette("Set1", 1))
    with pytest.raises(ValueError, match="Length of header does not match"):
        pl.quality_violin(adata, groupby="condition", columns=["qc_float"],
                          header=[])
    with pytest.raises(ValueError, match="The following columns from 'columns' were not found"):
        pl.quality_violin(adata, columns=["Invalid"])
