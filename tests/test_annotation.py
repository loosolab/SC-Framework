import argparse

import sctoolbox.annotation
import pytest
import sctoolbox.annotation as anno
import scanpy as sc
import os


@pytest.fixture
def adata_atac():
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')
    return sc.read_h5ad(adata_f)


def test_annotate_adata(adata_atac):

    gtf_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_genes.gtf')

    sctoolbox.annotation.annotate_adata(adata_atac, gtf=gtf_path)

    assert 'gene_id' in adata_atac.var.columns


def test_annotate_narrowPeak():

    gtf_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_genes.gtf')
    peaks_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'cropped_testing.narrowPeak')

    annotation_table = sctoolbox.annotation.annotate_narrowPeak(peaks_path, gtf=gtf_path)

    assert 'gene_id' in annotation_table


@pytest.fixture
def adata_rna():
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    return sc.read_h5ad(adata_f)


def test_add_cellxgene_annotation(adata_rna):
    
    csv_f = os.path.join(os.path.dirname(__file__), 'data', 'cellxgene_anno.csv')
    anno.add_cellxgene_annotation(adata_rna, csv_f)

    assert "cellxgene_clusters" in adata_rna.obs.columns


def test_annot_HVG(adata_rna):

    sc.pp.log1p(adata_rna)
    anno.annot_HVG(adata_rna)

    assert "highly_variable" in adata_rna.var.columns

def test_make_tmp():

    temp_dir = sctoolbox.annotation.make_tmp("")
    tmp_exists = os.path.exists(temp_dir)

    sctoolbox.annotation.rm_tmp(temp_dir)
    tmp_removed = not(os.path.exists(temp_dir))

    assert tmp_exists and tmp_removed

def test_gtf_integrity():

    gtf_no_header = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_genes.gtf')
    gtf = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.gtf')
    gff = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.gff3')
    gtf_gz = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.gtf.gz')
    gtf_missing_col = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_missing_column_gencode.v41.gtf')
    gtf_corrupted_format = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_corrupted_format_gencode.v41.gtf')

    # Test for passed tests
    result_gtf_no_header = sctoolbox.annotation.gtf_integrity(gtf_no_header)
    result_gtf = sctoolbox.annotation.gtf_integrity(gtf)
    assert result_gtf_no_header and result_gtf

    #Test if exceptions raised
    with pytest.raises(argparse.ArgumentTypeError) as e_gff_info:
        sctoolbox.annotation.gtf_integrity(gff)

    with pytest.raises(argparse.ArgumentTypeError) as e_gz_info:
        sctoolbox.annotation.gtf_integrity(gtf_gz)

    with pytest.raises(argparse.ArgumentTypeError) as e_missing_col_info:
        sctoolbox.annotation.gtf_integrity(gtf_missing_col)

    with pytest.raises(argparse.ArgumentTypeError) as e_corrupted_info:
        sctoolbox.annotation.gtf_integrity(gtf_corrupted_format)

    # Check for the exceptions types
    assert e_gff_info.value.args[0] == 'Expected filetype gtf not gff3'
    assert e_gz_info.value.args[0] == 'gtf file is compressed'
    assert e_missing_col_info.value.args[0] == 'Number of columns in the gtf file unequal 9'
    assert e_corrupted_info.value.args[0] == 'gtf file is corrupted'