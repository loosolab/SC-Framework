import os
import scanpy as sc
import sctoolbox.annotation
import pytest
import sctoolbox.annotation as anno
import scanpy as sc
import os


def test_annotate_adata():

    gtf_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_genes.gtf')
    adata_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')
    adata_atac = sc.read_h5ad(adata_path)

    sctoolbox.annotation.annotate_adata(adata_atac, gtf=gtf_path)

    assert 'gene_id' in adata.var.columns


def test_annotate_narrowPeak():

    gtf_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_genes.gtf')
    peaks_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'cropped_testing.narrowPeak')

    annotation_table = sctoolbox.annotation.annotate_narrowPeak(peaks_path, gtf=gtf_path)

    assert 'gene_id' in annotation_table


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
