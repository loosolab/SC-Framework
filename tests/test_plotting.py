import pytest
import sctoolbox.plotting
import scanpy as sc
import os
import tempfile
import shutil
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")
    adata = sc.read_h5ad(f)
    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
    adata.obs["clustering"] = np.random.choice(["1", "2", "3", "4"], size=adata.shape[0])
    adata.obs["LISI_score_pca"] = np.random.normal(size=adata.shape[0])
    adata.obs["qc_float"] = np.random.uniform(0, 1, size=adata.shape[0])

    sc.tl.umap(adata, n_components=3)
    sc.tl.tsne(adata)
    sc.tl.pca(adata)

    return adata


@pytest.fixture
def df():
    """Create and return a pandas dataframe"""
    return pd.DataFrame(data={'col1': [1, 2, 3, 4, 5],
                              'col2': [3, 4, 5, 6, 7]})


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


# ------------------------------ TESTS --------------------------------- #


def test_sc_colormap():
    """ Test whether sc_colormap returns a colormap """

    cmap = sctoolbox.plotting.sc_colormap()
    assert type(cmap).__name__ == "ListedColormap"


@pytest.mark.parametrize("n_selected", [None, 1, 2])
def test_plot_pca_variance(adata, n_selected):
    """ Test if Axes object is returned. """
    ax = sctoolbox.plotting.plot_pca_variance(adata, n_selected=n_selected)
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


def test_plot_pca_variance_fail(adata):
    """ Test if failes on invalid method. """
    # generate invalid method
    invalid = "-".join(list(adata.uns.keys())) + "-invalid"

    with pytest.raises(KeyError):
        sctoolbox.plotting.plot_pca_variance(adata, method=invalid)


@pytest.mark.parametrize("method", ["umap"])  # , "tsne"]) # tsne option is currently broken and sends the function to sleep. Will be added if fixed.
def test_search_dim_red_parameters(adata, method):
    """ Test if search_dim_red_parameters returns an array of axes. """

    axarr = sctoolbox.plotting._search_dim_red_parameters(adata,
                                                          color="condition",
                                                          method=method,
                                                          min_dist_range=(0.1, 0.3, 0.1),
                                                          spread_range=(2.0, 3.0, 0.5),
                                                          learning_rate_range=(100, 300, 100),
                                                          perplexity_range=(20, 30, 5))
    assert type(axarr).__name__ == "ndarray"
    assert axarr.shape == (2, 2)


def test_invalid_method_search_dim_red_parameter(adata):
    with pytest.raises(ValueError):
        sctoolbox.plotting._search_dim_red_parameters(adata,
                                                      color="condition",
                                                      method="invalid")


@pytest.mark.parametrize("range", [(0.1, 0.2, 0.1, 0.1), (0.1, 0.2, 0.3)])
def test_search_dim_red_parameters_ranges(adata, range):
    """ Test that invalid ranges raise ValueError. """

    with pytest.raises(ValueError):
        sctoolbox.plotting._search_dim_red_parameters(adata,
                                                      method="umap",
                                                      color="condition",
                                                      min_dist_range=range,
                                                      spread_range=(2.0, 3.0, 0.5))

    with pytest.raises(ValueError):
        sctoolbox.plotting._search_dim_red_parameters(adata,
                                                      method="umap",
                                                      color="condition",
                                                      spread_range=range,
                                                      min_dist_range=(0.1, 0.3, 0.1))


@pytest.mark.parametrize("embedding", ["pca", "umap", "tsne"])
def test_plot_group_embeddings(adata, embedding):

    axarr = sctoolbox.plotting.plot_group_embeddings(adata, groupby="condition", embedding=embedding, ncols=2)

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
        var_list = [adata.var.index[0], "condition", "notfound"]  # notfound will be excluded
        res = (2, 2)
    else:
        res = (1, 2)
    axarr = sctoolbox.plotting.compare_embeddings(adata_list, var_list, embedding=embedding)

    assert axarr.shape == res


def test_invalid_var_list_compare_embeddings(adata):
    with pytest.raises(ValueError):
        adata_cp = adata.copy()
        adata_list = [adata, adata_cp]
        sctoolbox.plotting.compare_embeddings(adata_list, ["invalid_1", "invalid_2"], embedding="umap")


@pytest.mark.parametrize("method", ["leiden", "louvain"])
def test_search_clustering_parameters(adata, method):
    """ Test if search_clustering_parameters returns an array of axes. """

    axarr = sctoolbox.plotting.search_clustering_parameters(adata, method=method, resolution_range=(0.1, 0.31, 0.1), ncols=2)

    assert type(axarr).__name__ == "ndarray"
    assert axarr.shape == (2, 2)


def test_wrong_embeding_search_clustering_parameters(adata):
    with pytest.raises(KeyError):
        sctoolbox.plotting.search_clustering_parameters(adata,
                                                        embedding="Invalid")


@pytest.mark.parametrize("method,resrange", [("leiden", (0.1, 0.2, 0.1, 0.1)),
                                             ("leiden", (0.1, 0.2, 0.3)),
                                             ("unknown", (0.1, 0.3, 0.1))])
def test_search_clustering_parameters_errors(adata, method, resrange):

    with pytest.raises(ValueError):
        sctoolbox.plotting.search_clustering_parameters(adata, resolution_range=resrange, method=method)


def test_anndata_overview(adata, tmp_file):
    """ Test anndata_overview success and file generation. """
    adatas = {"raw": adata, "corrected": adata}

    assert not os.path.exists(tmp_file)

    sctoolbox.plotting.anndata_overview(
        adatas=adatas,
        color_by=list(adata.obs.columns) + [adata.var_names.tolist()[0]],
        plots=["PCA", "PCA-var", "UMAP", "tSNE", "LISI"],
        figsize=None,
        output=None,
        dpi=300
    )

    assert not os.path.exists(tmp_file)

    sctoolbox.plotting.anndata_overview(
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
        sctoolbox.plotting.anndata_overview(
            adatas=adatas,
            color_by=None,
            plots=["PCA"],
            figsize=None,
            output=None,
            dpi=300
        )

    # wrong input
    with pytest.raises(ValueError, match="Couldn't find column"):
        sctoolbox.plotting.anndata_overview(
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
        sctoolbox.plotting.anndata_overview(
            adatas=adatas_invalid,
            color_by=list(adata.obs.columns) + [adata.var_names.tolist()[0]],
            plots=["PCA"],
            figsize=None,
            output=None,
            dpi=300
        )

    # Missing LISI score
    with pytest.raises(ValueError, match="No LISI scores found"):
        sctoolbox.plotting.anndata_overview(
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
        sctoolbox.plotting.anndata_overview(
            adatas=adatas,
            color_by=list(adata.obs.columns),
            plots=None,
            figsize=None,
            output=None,
            dpi=300
        )

    # wrong input
    with pytest.raises(ValueError, match="Invalid plot specified:"):
        sctoolbox.plotting.anndata_overview(
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
    ax = sctoolbox.plotting.group_expression_boxplot(adata, gene_list, groupby="condition")
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")  # depending on matplotlib version, it can be either AxesSubplot or Axes


def test_boxplot(df):
    """ Test if Axes object is returned. """
    ax = sctoolbox.plotting.boxplot(df)
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("color", ["ENSMUSG00000102693", "clustering", "qc_float"])
def test_plot_3D_UMAP(adata, color):
    """ Test if 3d plot is written to html """

    # Run 3d plotting
    sctoolbox.plotting.plot_3D_UMAP(adata, color=color, save="3D_test")

    # Assert creation of file
    assert os.path.isfile("3D_test.html")
    os.remove("3D_test.html")


def test_invalid_color_plot_3D_UMAP(adata):
    """ Test if plot_3D_UMAP return KeyError if color paramater cannot be found in adata"""
    with pytest.raises(KeyError):
        sctoolbox.plotting.plot_3D_UMAP(adata, color="invalid", save="3D_test")


def test_group_correlation(adata):
    """ Test if plot is written to pdf """

    # Run group correlation
    sctoolbox.plotting.group_correlation(adata, groupby="condition", save="group_correlation.pdf")

    # Assert creation of file
    assert os.path.isfile("group_correlation.pdf")
    os.remove("group_correlation.pdf")


@pytest.mark.parametrize("groupby", [None, "condition"])
def test_n_cells_barplot(adata, groupby):

    axarr = sctoolbox.plotting.n_cells_barplot(adata, "clustering", groupby=groupby)

    if groupby is None:
        assert len(axarr) == 1
    else:
        assert len(axarr) == 2


@pytest.mark.parametrize("x,y,groupby", [("clustering", "ENSMUSG00000102693", "condition"),
                                         ("ENSMUSG00000102693", None, "condition"),
                                         ("clustering", "qc_float", "condition")])
def test_grouped_violin(adata, x, y, groupby):

    ax = sctoolbox.plotting.grouped_violin(adata, x=x, y=y, groupby=groupby)
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("show_umap", [True, False])
def test_marker_gene_clustering(adata, show_umap):
    """ Test marker_gene_clustering"""

    marker_dict = {"Celltype A": ['ENSMUSG00000103377', 'ENSMUSG00000104428'],
                   "Celltype B": ['ENSMUSG00000102272']}

    axes_list = sctoolbox.plotting.marker_gene_clustering(adata, "condition",
                                                          marker_dict,
                                                          show_umap=show_umap)
    assert isinstance(axes_list, list)
    ax_type = type(axes_list[0]).__name__
    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("how", ["vertical", "horizontal"])
def test_flip_embedding(adata, how):
    tmp = adata.copy()
    key = "X_umap"
    sctoolbox.plotting.flip_embedding(adata, key=key, how=how)

    if how == "vertical":
        assert all(adata.obsm[key][:, 1] == -tmp.obsm[key][:, 1])
    elif how == "horizontal":
        assert all(adata.obsm[key][:, 0] == -tmp.obsm[key][:, 0])


def test_invalid_flip_embedding(adata):
    with pytest.raises(ValueError):
        sctoolbox.plotting.flip_embedding(adata, how="invalid")

    with pytest.raises(KeyError):
        sctoolbox.plotting.flip_embedding(adata, key="invalid")


@pytest.mark.parametrize("n, res", [(500, 12), (1000, 8),
                                    (5000, 8), (10000, 3), (20000, 3)])
def test_get_3d_dotsize(n, res):
    assert sctoolbox.plotting._get_3d_dotsize(int(n)) == res


@pytest.mark.parametrize("marker", ["ENSMUSG00000103377",
                                    ["ENSMUSG00000103377", 'ENSMUSG00000104428']])
def test_umap_marker_overview(adata, marker):
    """ Test umap_marker_overview """
    axes_list = sctoolbox.plotting.umap_marker_overview(adata, marker)

    assert isinstance(axes_list, list)
    ax_type = type(axes_list[0]).__name__
    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("color,title", [("condition", "Condition"),
                                         (["condition", "clustering"], None),
                                         (["condition", "clustering"], ["Condition", "Clustering"])])
def test_umap_pub(adata, color, title):
    """ Test umap_pub plotting with different color and title parameter. """
    axes_list = sctoolbox.plotting.umap_pub(adata, color=color, title=title)

    assert type(axes_list).__name__ == "list"
    ax_type = type(axes_list[0]).__name__
    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("color,title", [("condition", ["Title 1", "Title 2"]),
                                         (["condition", "clustering"], "Title 1")])
def test_invalid_parameter_len_umap_pub(adata, color, title):
    """ Test case if color and title are not the same lenght """
    with pytest.raises(ValueError):
        sctoolbox.plotting.umap_pub(adata, color=color, title=title)


@pytest.mark.parametrize("color", [["clustering", "condition"], "clustering"])
def test_add_figure_title(adata, color):
    """ Test if function _add_figure_title runs without error when called correctly. """
    axes = sc.pl.umap(adata, color=color, show=False)
    sctoolbox.plotting._add_figure_title(axes, "UMAP plots", fontsize=20)

    assert True


def test_add_labels(adata):
    # TODO
    assert True


@pytest.mark.parametrize("array,mini,maxi", [(np.array([1, 2, 3]), 0, 1),
                                             (np.array([[1, 2, 3], [1, 2, 3]]), 1, 100),
                                             (np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]]), 1, 5)])
def test_scale_values(array, mini, maxi):
    """ Test that scaled values are in given range. """
    result = sctoolbox.plotting._scale_values(array, mini, maxi)

    assert len(result) == len(array)
    if len(result.shape) == 1:
        assert all((mini <= result) & (result <= maxi))
    else:
        for i in range(len(result)):
            assert all((mini <= result[i]) & (result[i] <= maxi))


def test_clustermap_dotplot():
    """ Test clustermap_dotplot. """
    table = sc.datasets.pbmc68k_reduced().obs.reset_index()[:10]
    sctoolbox.plotting.clustermap_dotplot(table=table, x="bulk_labels",
                                          y="index", color="n_genes",
                                          size="n_counts", cmap="viridis",
                                          vmin=0, vmax=10)
    assert True


def test_bidirectional_barplot(df_bidir_bar):
    """ Test bidirectoional_barplot. """
    sctoolbox.plotting.bidirectional_barplot(df_bidir_bar, title="Title")
    assert True


def test_bidirectional_barplot_fail(df):
    """ test bidorectional_barplot with invalid input. """
    with pytest.raises(ValueError):
        sctoolbox.plotting.bidirectional_barplot(df)


@pytest.mark.parametrize("ylabel,color_by,hlines", [(True, None, 0.5),
                                                    (False, "clustering",[0.5, 0.5, 0.5, 0.5])])
def test_violinplot(adata, ylabel, color_by, hlines):
    """ Test violinplot. """
    ax = sctoolbox.plotting.violinplot(adata.obs, "qc_float", color_by=color_by,
                                       hlines=hlines, colors=None, ax=None,
                                       title="Title", ylabel=ylabel)
    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")


def test_violinplot_fail(adata):
    """ Test invalid input for violinplot. """
    with pytest.raises(ValueError, match='not found in column names of table!'):
        sctoolbox.plotting.violinplot(adata.obs, y="Invalid")

    with pytest.raises(ValueError, match='Color grouping'):
        sctoolbox.plotting.violinplot(adata.obs, y="qc_float", color_by="Invalid")

    with pytest.raises(ValueError, match='Parameter hlines has to be number or list'):
        sctoolbox.plotting.violinplot(adata.obs, y="qc_float", hlines={"A": 0.5})

    with pytest.raises(ValueError, match='Invalid dict keys in hlines parameter.'):
        sctoolbox.plotting.violinplot(adata.obs, y="qc_float",
                                      color_by="clustering", hlines={"A": 0.5})


def test_plot_venn(venn_dict):
    """ Test plot_venn with 3 and 2 groups. """
    sctoolbox.plotting.plot_venn(venn_dict, title="Test")
    venn_dict.pop("Group C")
    sctoolbox.plotting.plot_venn(venn_dict, title="Test")
    assert True


def test_plot_venn_fail(venn_dict):
    """ Test for invalid input. """
    venn_dict["Group D"] = [1, 2]
    with pytest.raises(ValueError):
        sctoolbox.plotting.plot_venn(venn_dict)

    with pytest.raises(ValueError):
        sctoolbox.plotting.plot_venn([1, 2, 3, 4, 5])


def test_violin_HVF_distribution(adata):
    """ Test violin_HVF_distribution. """
    adata_HVF = adata.copy()
    adata_HVF.var['highly_variable'] = np.random.choice([True, False], size=adata_HVF.shape[1])
    adata_HVF.var['n_cells_by_counts'] = np.random.normal(size=adata_HVF.shape[1])
    sctoolbox.plotting.violin_HVF_distribution(adata_HVF)
    assert True


def test_violin_HVF_distribution_fail(adata):
    """ Test if input is invalid. """
    with pytest.raises(KeyError):
        sctoolbox.plotting.violin_HVF_distribution(adata)


def test_scatter_HVF_distribution(adata):
    """ Test scatter_HVF_distribution. """
    adata_HVF = adata.copy()
    adata_HVF.var['variability_score'] = np.random.normal(size=adata_HVF.shape[1])
    adata_HVF.var['n_cells'] = np.random.normal(size=adata_HVF.shape[1])
    sctoolbox.plotting.scatter_HVF_distribution(adata_HVF)
    assert True


def test_scatter_HVF_distribution_fail(adata):
    """ Test if input is invalid. """
    with pytest.raises(KeyError):
        sctoolbox.plotting.scatter_HVF_distribution(adata)
