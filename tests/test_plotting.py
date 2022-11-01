import pytest
import sctoolbox.plotting
import scanpy as sc
import os
import tempfile
import shutil
import pandas as pd
import numpy as np


@pytest.fixture
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")
    adata = sc.read_h5ad(f)
    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
    adata.obs["clustering"] = np.random.choice(["1", "2", "3", "4"], size=adata.shape[0])

    sc.tl.umap(adata)
    sc.tl.tsne(adata)
    sc.tl.pca(adata)

    return adata


@pytest.fixture
def df():
    """Create and return a pandas dataframe"""
    return pd.DataFrame(data={'col1': [1, 2, 3, 4, 5],
                              'col2': [3, 4, 5, 6, 7]})


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


def test_plot_pca_variance(adata):
    """ Test if Axes object is returned. """
    ax = sctoolbox.plotting.plot_pca_variance(adata)
    ax_type = type(ax).__name__

    assert ax_type == "AxesSubplot"


def test_plot_pca_variance_fail(adata):
    """ Test if failes on invalid method. """
    # generate invalid method
    invalid = "-".join(list(adata.uns.keys())) + "-invalid"

    with pytest.raises(KeyError):
        sctoolbox.plotting.plot_pca_variance(adata, method=invalid)


def test_search_umap_parameters(adata):
    """ Test if search_umap_parameters returns an array of axes. """

    axarr = sctoolbox.plotting.search_umap_parameters(adata,
                                                      metacol="condition",
                                                      dist_range=(0.1, 0.3, 0.1),
                                                      spread_range=(2.0, 3.0, 0.5))

    assert type(axarr).__name__ == "ndarray"
    assert axarr.shape == (2, 2)


@pytest.mark.parametrize("range", [(0.1, 0.2, 0.1, 0.1), (0.1, 0.2, 0.3)])
def test_search_umap_parameters_ranges(adata, range):

    with pytest.raises(ValueError):
        sctoolbox.plotting.search_umap_parameters(adata,
                                                metacol="condition",
                                                dist_range=range)

    with pytest.raises(ValueError):
        sctoolbox.plotting.search_umap_parameters(adata,
                                                    metacol="condition",
                                                    spread_range=range)


@pytest.mark.parametrize("embedding", ["pca", "umap", "tsne"])
def test_plot_group_embeddings(adata, embedding):

    axarr = sctoolbox.plotting.plot_group_embeddings(adata, groupby="condition", embedding=embedding, ncols=2)

    assert axarr.shape == (2, 2)


@pytest.mark.parametrize("embedding", ["pca", "umap", "tsne"])
def test_compare_embeddings(adata, embedding):

    adata_cp = adata.copy()
    adata_cp.obs.drop(columns=["condition"], inplace=True)  # check that function can deal with missing vars

    adata_list = [adata, adata_cp]
    var_list = [adata.var.index[0], "condition"]
    axarr = sctoolbox.plotting.compare_embeddings(adata_list, var_list, embedding=embedding)

    assert axarr.shape == (2, 2)


def test_search_clustering_parameters(adata):
    """ Test if search_clustering_parameters returns an array of axes. """

    axarr = sctoolbox.plotting.search_clustering_parameters(adata, resolution_range=(0.1, 0.3, 0.1))

    assert type(axarr).__name__ == "ndarray"
    assert axarr.shape == (1, 2)


def test_anndata_overview(adata, tmp_file):
    """ Test anndata_overview success and file generation. """
    adatas = {"raw": adata, "corrected": adata}

    assert not os.path.exists(tmp_file)

    sctoolbox.plotting.anndata_overview(
        adatas=adatas,
        color_by=list(adata.obs.columns) + [adata.var_names.tolist()[0]],
        plots=["PCA", "PCA-var", "UMAP"],
        figsize=None,
        output=None,
        dpi=300
    )

    assert not os.path.exists(tmp_file)

    sctoolbox.plotting.anndata_overview(
        adatas=adatas,
        color_by=list(adata.obs.columns),
        plots=["PCA", "PCA-var", "UMAP"],
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
            plots=["PCA", "PCA-var", "UMAP"],
            figsize=None,
            output=None,
            dpi=300
        )

    # wrong input
    with pytest.raises(ValueError, match="Couldn't find column"):
        sctoolbox.plotting.anndata_overview(
            adatas=adatas,
            color_by="-".join(list(adata.obs.columns)) + "-invalid",
            plots=["PCA", "PCA-var", "UMAP"],
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


def test_boxplot(df):
    """ Test if Axes object is returned. """
    ax = sctoolbox.plotting.boxplot(df)
    ax_type = type(ax).__name__

    assert ax_type == "AxesSubplot"


def test_plot_3D_UMAP(adata):
    """ Test if 3d plot is written to html """

    sc.tl.umap(adata, n_components=3)

    # Run 3d plotting
    color = adata.var.index[0]
    sctoolbox.plotting.plot_3D_UMAP(adata, color=color, save="3D_test")

    # Assert creation of file
    assert os.path.isfile("3D_test.html")


@pytest.mark.parametrize("groupby", [None, "condition"])
def test_n_cells_barplot(adata, groupby):

    axarr = sctoolbox.plotting.n_cells_barplot(adata, "clustering", groupby=groupby)

    if groupby is None:
        assert len(axarr) == 1
    else:
        assert len(axarr) == 2
