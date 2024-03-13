"""Test plotting functions."""

import pytest
import sctoolbox.plotting as pl
import scanpy as sc
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from beartype.roar import BeartypeCallHintParamViolation

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")


# ------------------------------ FIXTURES --------------------------------- #

quant_folder = os.path.join(os.path.dirname(__file__), 'data', 'quant')


@pytest.fixture(scope="session")  # re-use the fixture for all tests
def adata():
    """Load and returns an anndata object."""

    np.random.seed(1)  # set seed for reproducibility

    f = os.path.join(os.path.dirname(__file__), '..', 'data', "adata.h5ad")
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
def df():
    """Create and return a pandas dataframe."""
    return pd.DataFrame(data={'col1': [1, 2, 3, 4, 5],
                              'col2': [3, 4, 5, 6, 7]})


@pytest.fixture
def tmp_file():
    """Return path for a temporary file."""
    tmpdir = tempfile.mkdtemp()

    yield os.path.join(tmpdir, "output.pdf")

    # clean up directory and contents
    shutil.rmtree(tmpdir)


# ------------------------------ TESTS --------------------------------- #

def test_sc_colormap():
    """Test whether sc_colormap returns a colormap."""

    cmap = pl.sc_colormap()
    assert type(cmap).__name__ == "ListedColormap"


@pytest.mark.parametrize("how", ["vertical", "horizontal"])
def test_flip_embedding(adata, how):
    """Test flip_embedding success."""
    tmp = adata.copy()
    key = "X_umap"
    pl.flip_embedding(adata, key=key, how=how)

    if how == "vertical":
        assert all(adata.obsm[key][:, 1] == -tmp.obsm[key][:, 1])
    elif how == "horizontal":
        assert all(adata.obsm[key][:, 0] == -tmp.obsm[key][:, 0])


def test_invalid_flip_embedding(adata):
    """Test flip_embedding failure."""
    with pytest.raises(BeartypeCallHintParamViolation):
        pl.flip_embedding(adata, how="invalid")

    with pytest.raises(KeyError):
        pl.flip_embedding(adata, key="invalid")


@pytest.mark.parametrize("kwargs", [{"show_title": True, "show_contour": True, "components": "1,2"},
                                    {"show_title": False, "show_contour": False, "components": ["1,2", "2,3"]}])
@pytest.mark.parametrize("style", ["dots", "density", "hexbin"])
def test_embedding(adata, style, kwargs):
    """Assert embedding works and returns Axes object."""

    # Collect test colors
    colors = ["qcvar1"]   # continous obs variable
    colors.append(adata.var.index[0])  # continous gene variable
    colors.append(None)          # no color / density plot
    if style != "hexbin":
        colors.append("clustering")  # categorical obs variable; only available for dots/density

    axes_list = pl.plot_embedding(adata, color=colors, style=style, **kwargs)

    # Assert number of plots
    components = kwargs.get("components", "1,2")
    n_components = 1 if isinstance(components, str) else len(components)
    assert len(axes_list) == len(colors) * n_components

    # Assert type of output
    ax_type = type(axes_list[0]).__name__
    assert ax_type.startswith("Axes")


def test_embedding_single(adata):
    """Test that embedding works with single color."""
    axarr = pl.plot_embedding(adata, color="qcvar1")

    ax_type = type(axarr[0]).__name__
    assert ax_type.startswith("Axes")


def test_embedding_error(adata):
    """Test that embedding raises error for invalid input."""
    with pytest.raises(ValueError):
        pl.plot_embedding(adata, components="3,4")


def test_search_umap_parameters(adata):
    """Test if search_umap_parameters returns an array of axes."""

    axarr = pl.search_umap_parameters(adata, color="condition",
                                      min_dist_range=(0.1, 0.3, 0.1),
                                      spread_range=(2.0, 3.0, 0.5))
    assert type(axarr).__name__ == "ndarray"
    assert axarr.shape == (2, 2)


def test_search_tsne_parameters(adata):
    """Test if search_tsne_parameters returns an array of axes."""

    axarr = pl.search_tsne_parameters(adata, color="condition",
                                      learning_rate_range=(100, 300, 100),
                                      perplexity_range=(20, 30, 5))
    assert type(axarr).__name__ == "ndarray"
    assert axarr.shape == (2, 2)


def test_invalid_method_search_dim_red_parameter(adata):
    """Test if error is raised for invalid method."""
    with pytest.raises(BeartypeCallHintParamViolation):
        pl._search_dim_red_parameters(adata, color="condition",
                                      method="invalid")


@pytest.mark.parametrize("range", [(0.1, 0.2, 0.1, 0.1), (0.1, 0.2, 0.3)])
def test_search_dim_red_parameters_ranges(adata, range):
    """Test that invalid ranges raise ValueError."""

    with pytest.raises((BeartypeCallHintParamViolation, ValueError)):
        pl._search_dim_red_parameters(adata, method="umap",
                                      color="condition",
                                      min_dist_range=range,
                                      spread_range=(2.0, 3.0, 0.5))

    with pytest.raises((BeartypeCallHintParamViolation, ValueError)):
        pl._search_dim_red_parameters(adata, method="umap",
                                      color="condition",
                                      spread_range=range,
                                      min_dist_range=(0.1, 0.3, 0.1))


@pytest.mark.parametrize("embedding", ["pca", "umap", "tsne"])
def test_plot_group_embeddings(adata, embedding):
    """Test if plot_group_embeddings runs through."""

    axarr = pl.plot_group_embeddings(adata, groupby="condition",
                                     embedding=embedding, ncols=2)

    assert axarr.shape == (2, 2)


@pytest.mark.parametrize("embedding, var_list", [("pca", "list"),
                                                 ("umap", "condition"),
                                                 ("tsne", "list")])
def test_compare_embeddings(adata, embedding, var_list):
    """Test if compare_embeddings runs trough."""

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
    """Test if compare_embeddings raises error."""
    with pytest.raises(ValueError):
        adata_cp = adata.copy()
        adata_list = [adata, adata_cp]
        pl.compare_embeddings(adata_list, ["invalid_1", "invalid_2"],
                              embedding="umap")


@pytest.mark.parametrize("n, res", [(500, 12), (1000, 8),
                                    (5000, 8), (10000, 3), (20000, 3)])
def test_get_3d_dotsize(n, res):
    """Test _get_3d_dotsize success."""
    assert pl._get_3d_dotsize(int(n)) == res


@pytest.mark.parametrize("color", ["ENSMUSG00000102693", "clustering", "qc_float"])
def test_plot_3D_UMAP(adata, color):
    """Test if 3d plot is written to html."""

    # Run 3d plotting
    pl.plot_3D_UMAP(adata, color=color, save="3D_test")

    # Assert creation of file
    assert os.path.isfile("3D_test.html")
    os.remove("3D_test.html")


def test_invalid_color_plot_3D_UMAP(adata):
    """Test if plot_3D_UMAP return KeyError if color paramater cannot be found in adata."""
    with pytest.raises(KeyError):
        pl.plot_3D_UMAP(adata, color="invalid", save="3D_test")


@pytest.mark.parametrize("marker", ["ENSMUSG00000103377",
                                    ["ENSMUSG00000103377", 'ENSMUSG00000104428']])
def test_umap_marker_overview(adata, marker):
    """Test umap_marker_overview."""
    axes_list = pl.umap_marker_overview(adata, marker)

    assert isinstance(axes_list, list)
    ax_type = type(axes_list[0]).__name__
    assert ax_type.startswith("Axes")


def test_anndata_overview(adata, tmp_file):
    """Test anndata_overview success and file generation."""
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
    """Test invalid parameter inputs."""
    adatas = {"raw": adata}

    # invalid color_by
    # no input
    with pytest.raises(BeartypeCallHintParamViolation):
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
    """Test invalid parameter inputs."""
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
    """Test invalid parameter inputs."""
    adatas = {"raw": adata}

    # invalid plots
    # no input
    with pytest.raises(BeartypeCallHintParamViolation):
        pl.anndata_overview(
            adatas=adatas,
            color_by=list(adata.obs.columns),
            plots=None,
            figsize=None,
            output=None,
            dpi=300
        )

    # wrong input
    with pytest.raises((BeartypeCallHintParamViolation, ValueError)):
        pl.anndata_overview(
            adatas=adatas,
            color_by=list(adata.obs.columns),
            plots=["PCA", "invalid"],
            figsize=None,
            output=None,
            dpi=300
        )


@pytest.mark.parametrize("n_selected", [None, 1, 2])
def test_plot_pca_variance(adata, n_selected):
    """Test if Axes object is returned."""
    ax = pl.plot_pca_variance(adata, n_selected=n_selected)
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


def test_plot_pca_variance_fail(adata):
    """Test if function fails on invalid parameters."""

    with pytest.raises(KeyError, match="The given method"):
        pl.plot_pca_variance(adata, method="invalid")

    with pytest.raises(BeartypeCallHintParamViolation):
        pl.plot_pca_variance(adata, ax="invalid")


@pytest.mark.parametrize("kwargs", [{"which": "var", "method": "spearmanr"},
                                    {"basis": "umap", "method": "pearsonr"},
                                    {"basis": "umap", "plot_values": "pvalues"}])
def test_plot_pca_correlation(adata, kwargs):
    """Test if Axes object is returned without error."""

    ax = pl.plot_pca_correlation(adata, title="Title", **kwargs)
    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


@pytest.mark.parametrize("kwargs", [{"basis": "umap", "which": "var"},  # var is only available for pca coordinates
                                    {"basis": "invalid"},
                                    {"method": "invalid"},
                                    {"which": "invalid"},
                                    {"columns": ["invalid", "columns"]}])
def test_plot_pca_correlation_fail(adata, kwargs):
    """Test that an exception is raised upon error."""

    with pytest.raises((BeartypeCallHintParamViolation, KeyError, ValueError)):
        pl.plot_pca_correlation(adata, **kwargs)
