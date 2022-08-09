import pytest
import sctoolbox.annotation as anno
import scanpy as sc
import os

@pytest.fixture
def adata():
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    return sc.read_h5ad(adata_f)

def test_add_cellxgene_annotation(adata):
    
    csv_f = os.path.join(os.path.dirname(__file__), 'data', 'cellxgene_anno.csv')
    anno.add_cellxgene_annotation(adata, csv_f)

    assert "cellxgene_clusters" in adata.obs.columns

def test_annot_HVG(adata):

    sc.pp.log1p(adata)
    anno.annot_HVG(adata)

    assert "highly_variable" in adata.var.columns