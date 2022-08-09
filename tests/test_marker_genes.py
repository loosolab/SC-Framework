import pytest
import os
import scanpy as sc
import sctoolbox.marker_genes


def test_get_chromosome_genes():
    """ Test if get_chromosome_genes get the right genes from the gtf """

    gtf = os.path.join(os.path.dirname(__file__), 'data', 'genes.gtf')

    with pytest.raises(Exception):
        sctoolbox.marker_genes.get_chromosome_genes(gtf, "NA")

    genes_chr1 = sctoolbox.marker_genes.get_chromosome_genes(gtf, "chr1")
    genes_chr11 = sctoolbox.marker_genes.get_chromosome_genes(gtf, "chr11")

    assert genes_chr1 == ["DDX11L1", "WASH7P", "MIR6859-1"]
    assert genes_chr11 == ["DGAT2"]


def test_label_genes():
    """ Test of genes are labeled in adata.var """

    h5ad = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    adata = sc.read_h5ad(h5ad)

    sctoolbox.marker_genes.label_genes(adata, species="mouse")

    added_columns = ["is_ribo", "is_mito", "cellcycle", "is_gender"]
    missing = set(added_columns) - set(adata.var.columns)  # test that columns were added

    assert len(missing) == 0
