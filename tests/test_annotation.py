import pytest
import sctoolbox.annotation as anno
import scanpy as sc
import os


@pytest.fixture
def adata():
    """ Create anndata object. """
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    return sc.read_h5ad(adata_f)


def test_add_cellxgene_annotation(adata):
    """ Test if 'cellxgene_clusters' column is added to adata.obs. """
    csv_f = os.path.join(os.path.dirname(__file__), 'data', 'cellxgene_anno.csv')
    anno.add_cellxgene_annotation(adata, csv_f)

    assert "cellxgene_clusters" in adata.obs.columns


def test_annot_HVG(adata):
    """ Test if 'highly_vairable' column is added to adata.var. """
    sc.pp.log1p(adata)
    anno.annot_HVG(adata)

    assert "highly_variable" in adata.var.columns
