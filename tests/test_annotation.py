import os
import scanpy as sc
import sctoolbox.annotation


def test_annotate_adata():

    gtf_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_genes.gtf')
    adata_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')
    adata = sc.read_h5ad(adata_path)

    sctoolbox.annotation.annotate_adata(adata, gtf=gtf_path)

    assert 'gene_id' in adata.var.columns
