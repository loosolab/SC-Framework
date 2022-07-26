import sctoolbox.plotting
import scanpy as sc
import os

@pytset.fixture
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")

    return sc.read_h5ad(f)


def test_plot_pca_variance(adata):
    ax = sctoolbox.plotting.plot_pca_variance(adata)
    ax_type = type(ax).__name__

    assert ax_type == "AxesSubplot"
