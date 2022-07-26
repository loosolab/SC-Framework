import pytest
import sctoolbox.plotting
import scanpy as sc
import os
import tempfile
import shutil

@pytest.fixture
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")

    return sc.read_h5ad(f)

@pytest.fixture
def tmp_file():
    """ Return path for a temporary file. """
    tmpdir = tempfile.mkdtemp()

    yield os.path.join(tmpdir, "output.pdf")

    # clean up directory and contents
    shutil.rmtree(tmpdir)



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

def test_anndata_overview(adata, tmp_file):
    """ Test anndata_overview success and file generation. """
    adatas = {"raw": adata}

    assert not os.path.exists(tmp_file)

    sctoolbox.plotting.anndata_overview(
        adatas=adatas,
        color_by=list(adata.obs.columns),
        plots=["PCA", "PCA-var", "UMAP"],
        figsize=None,
        output=None,
        dpi=300
    )

    assert not os.path.exists(tmp_file)

    sctoolbox.plotting.anndata_overview(
        adatas=adatas,
        color_by=adata.obs.columns,
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
    with pytest.raises(ValueError):
        # no input
        sctoolbox.plotting.anndata_overview(
            adatas=adatas,
            color_by=None,
            plots=["PCA", "PCA-var", "UMAP"],
            figsize=None,
            output=None,
            dpi=300
        )
        # wrong input
        sctoolbox.plotting.anndata_overview(
            adatas=adatas,
            color_by=adata.obs.columns.join() + "-invalid",
            plots=["PCA", "PCA-var", "UMAP"],
            figsize=None,
            output=None,
            dpi=300
        )


def test_anndata_overview_fail_plots(adata):
    """ Test invalid parameter inputs """
    adatas = {"raw": adata}

    # invalid plots
    with pytest.raises(ValueError):
        # no input
        sctoolbox.plotting.anndata_overview(
            adatas=adatas,
            color_by=list(adata.obs.columns),
            plots=None,
            figsize=None,
            output=None,
            dpi=300
        )
        # wrong input
        sctoolbox.plotting.anndata_overview(
            adatas=adatas,
            color_by=list(adata.obs.columns),
            plots=["PCA", "invalid"],
            figsize=None,
            output=None,
            dpi=300
        )
