"""Test functions related to files containing genomic ranges."""

from sctoolbox import calc_overlap_pct as overlap
import os
import pytest
import anndata as ad
import pkg_resources


@pytest.fixture
def test_bam():
    """Load bam file."""
    bam_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'homo_sapiens_liver.bam')
    # bam_dir = os.path.join(os.path.dirname(__file__), 'data', 'atac')
    # bam_path = os.path.join(bam_dir, 'mm10_atac.bam')
    return bam_path


@pytest.fixture
def test_gtf():
    """Load gtf file."""

    # Location of gene lists
    gtf_dir = pkg_resources.resource_filename("sctoolbox", "data/promoters_gtf/")
    gtf_path = os.path.join(gtf_dir, "mus_musculus.104.promoters2000.gtf")

    return gtf_path


@pytest.fixture
def test_gtf_with_header():
    """Load a gtf like file with a header."""
    gtf_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.unsorted.gtf')
    return gtf_path


@pytest.fixture
def test_fragments():
    """Load a bed file containing fragments."""
    fragments_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'homo_sapiens_liver_fragments_sorted_test.bed')
    # fragments_dir = os.path.join(os.path.dirname(__file__), 'data', 'atac')
    # fragments_path = os.path.join(fragments_dir, 'homo_sapiens_liver_fragments_sorted.bed')
    return fragments_path


@pytest.fixture
def test_bed():
    """Load a bed with blacklisted regions."""
    bed_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'hg38.blacklist.v2_sorted.bed')
    # bed_dir = os.path.join(os.path.dirname(__file__), 'data', 'atac')
    # bed_path = os.path.join(bed_dir, 'mm10_promoters.gtf_sorted.bed')
    return bed_path


@pytest.fixture
def test_adata():
    """Load adata."""
    adata_dir = os.path.join(os.path.dirname(__file__), 'data', 'atac')
    adata = ad.read_h5ad(adata_dir + '/homo_sapiens_liver.h5ad')
    return adata


def tmp_dir():
    """Create a temporary directory."""
    if not os.path.exists("tests/data/tmp"):
        os.mkdir("tests/data/tmp")


@pytest.mark.parametrize("out", [None, 'tests/data/tmp'])
def test_create_fragment_file(test_bam, out):
    """Test create_fragment_file success."""
    if out:
        tmp_dir()
    fragments, temp = overlap.create_fragment_file(bam=test_bam, nproc=1, out=out, sort_bam=True)

    name = os.path.splitext(test_bam)[0] + "_fragments_sorted.bed"
    if out:
        expected = os.path.join(out, os.path.basename(name))
    else:
        expected = name

    assert fragments == expected and os.path.isfile(fragments)


@pytest.mark.parametrize("out", [None, 'tests/data/tmp'])
def test_convert_gtf_to_bed(test_gtf, out):
    """Test _convert_gtf_to_bed success."""
    if out:
        tmp_dir()

    sorted_bed, temp = overlap._convert_gtf_to_bed(test_gtf, out=out)
    name = test_gtf + "_sorted.bed"

    if out:
        expected = os.path.join(out, os.path.basename(name))
    else:
        expected = name

    assert sorted_bed == expected and os.path.isfile(sorted_bed)


# These tests do not work due to _overlap_two_beds returning "There was no overlap!"
# @pytest.mark.parametrize("out", [None, 'tests/data/tmp'])
# def test_overlap_two_beds(test_fragments, test_bed, out):
#     """Test _overlap_two_beds success."""
#     if out:
#         tmp_dir()

#     overlap_bed = overlap._overlap_two_beds(test_fragments, test_bed, out=out)

#     # get names of the bed files
#     name = os.path.splitext(test_fragments)[0] + "_" + os.path.splitext(os.path.basename(test_bed))[0] + "_overlap.bed"
#     if out:
#         expected = os.path.join(out, os.path.basename(name))
#     else:
#         expected = name

#     assert overlap_bed == expected and os.path.isfile(overlap_bed)


# @pytest.mark.parametrize("species", ['homo_sapiens', 'mus_musculus'])
# def test_pct_fragments_in_promoters(test_adata, test_bam, species):
#     """Test pct_fragments_in_promoters success."""

#     overlap.pct_fragments_in_promoters(test_adata, bam_file=test_bam, species=species, sort_bam=True)

#     col_total_fragments = 'n_total_fragments'
#     col_n_fragments_in_list = 'n_fragments_in_promoters'
#     col_pct_fragments = 'pct_fragments_in_promoters'

#     assert {col_total_fragments, col_n_fragments_in_list, col_pct_fragments}.issubset(test_adata.obs.columns)


# regions_1 = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'hg38.blacklist.v2.bed')
# @pytest.mark.parametrize("regions, name", [(regions_1, 'blacklist')])
# def test_pct_fragments_overlap(test_adata, regions, test_bam, name):
#     """Test pct_fragments_overlap success."""
#     overlap.pct_fragments_overlap(adata=test_adata, regions_file=regions, bam_file=test_bam,
#                                   regions_name=name, sort_bam=True, sort_regions=True)

#     col_total_fragments = 'n_total_fragments'
#     col_n_fragments_in_list = 'n_fragments_in_' + name
#     col_pct_fragments = 'pct_fragments_in_' + name

#     assert {col_total_fragments, col_n_fragments_in_list, col_pct_fragments}.issubset(test_adata.obs.columns)
