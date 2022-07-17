import os
import scanpy as sc
import sctoolbox.plotting

def test_plot_3D_UMAP():

    f = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    adata = sc.read_h5ad(f)

    #Run 3d plotting
    color = adata.var.index[0]
    sctoolbox.plotting.plot_3D_UMAP(adata, color=color, save="3D_test")

    #Assert creation of file
    assert os.path.isfile("3D_test.html")
