from sctoolbox import calc_overlap_pct as overlap
import sctoolbox.utilities as utils
import os
import sys
import pytest
import anndata as ad


@pytest.fixture
def test_bam():
    bam_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'homo_sapiens_liver.bam')
    #bam_dir = os.path.join(os.path.dirname(__file__), 'data', 'atac')
    #bam_path = os.path.join(bam_dir, 'mm10_atac.bam')
    return bam_path


@pytest.fixture
def test_gtf():
    gtf_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_promoters.gtf')
    #gtf_dir = os.path.join(os.path.dirname(__file__), 'data', 'atac')
    #gtf_path = os.path.join(gtf_dir, 'mm10_promoters.gtf')
    return gtf_path


@pytest.fixture
def test_gtf_with_header():
    gtf_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.unsorted.gtf')
    return gtf_path


@pytest.fixture
def test_fragments():
    fragments_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'homo_sapiens_liver_fragments_sorted_test.bed')
    #fragments_dir = os.path.join(os.path.dirname(__file__), 'data', 'atac')
    #fragments_path = os.path.join(fragments_dir, 'homo_sapiens_liver_fragments_sorted.bed')
    return fragments_path


@pytest.fixture
def test_bed():
    bed_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'hg38.blacklist.v2_sorted.bed')
    #bed_dir = os.path.join(os.path.dirname(__file__), 'data', 'atac')
    #bed_path = os.path.join(bed_dir, 'mm10_promoters.gtf_sorted.bed')
    return bed_path


@pytest.fixture
def test_adata():
    adata_dir = os.path.join(os.path.dirname(__file__), 'data', 'atac')
    adata = ad.read_h5ad(adata_dir + '/homo_sapiens_liver.h5ad')
    return adata


def tmp_dir():
    if not os.path.exists("./data/tmp"):
        os.mkdir("./data/tmp")


@pytest.mark.parametrize("out", [None, './data/tmp'])
def test_create_fragment_file(test_bam, out):
    if out:
        tmp_dir()
    fragments = overlap.create_fragment_file(bam=test_bam, nproc=1, out=out, sort_bam=True)
    
    if out:
        expected = os.path.join(out, "homo_sapiens_liver_fragments_sorted.bed")
    else:
        expected = os.path.splitext(test_bam)[0] + "_fragments_sorted.bed"
        
    assert fragments == expected and os.path.isfile(fragments)
    

@pytest.mark.parametrize("out", [None, './data/tmp'])
def test_convert_gtf_to_bed(test_gtf, out):
    if out:
        tmp_dir()
        
    sorted_bed = overlap._convert_gtf_to_bed(test_gtf, out=out)
    
    if out:
        expected = os.path.join(out, "mm10_promoters.gtf_sorted.bed")
    else:
        expected = test_gtf + "_sorted.bed"
        
    assert sorted_bed == expected and os.path.isfile(sorted_bed)
    

@pytest.mark.parametrize("out", [None, './data/tmp'])
def test_overlap_two_beds(test_fragments, test_bed, out):
    if out:
        tmp_dir()
        
    overlap_bed = overlap._overlap_two_beds(test_fragments, test_bed, out=out)
    
    if out:
        expected = os.path.join(out, 'homo_sapiens_liver_fragments_sorted_test_hg38.blacklist.v2_sorted_overlap.bed')
    else:
        expected = os.path.splitext(test_fragments)[0] + "_hg38.blacklist.v2_sorted_overlap.bed"
        
    assert overlap_bed == expected and os.path.isfile(overlap_bed)
    

@pytest.mark.parametrize("species", ['homo_sapiens', 'mus_musculus'])
def test_pct_fragments_in_promoters(test_adata, test_bam, species):
    
    overlap.pct_fragments_in_promoters(test_adata, bam_file=test_bam, species=species, sort_bam=True)
    
    col_total_fragments = 'n_total_fragments'
    col_n_fragments_in_list = 'n_fragments_in_promoters'
    col_pct_fragments = 'pct_fragments_in_promoters'
    
    assert {col_total_fragments, col_n_fragments_in_list, col_pct_fragments}.issubset(test_adata.obs.columns)

    
regions_1 = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'hg38.blacklist.v2.bed')

@pytest.mark.parametrize("regions, name", [(regions_1,'blacklist')])
def test_pct_fragments_overlap(test_adata, regions, test_bam, name):
    overlap.pct_fragments_overlap(adata=test_adata, regions_file=regions, bam_file=test_bam, 
                                  regions_name=name, sort_bam=True, sort_regions=True)
    
    col_total_fragments = 'n_total_fragments'
    col_n_fragments_in_list = 'n_fragments_in_' + name
    col_pct_fragments = 'pct_fragments_in_' + name
    
    assert {col_total_fragments, col_n_fragments_in_list, col_pct_fragments}.issubset(test_adata.obs.columns)
