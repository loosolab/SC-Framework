"""Test functions related to files containing genomic ranges."""

import sctoolbox.tools as tl
import sctoolbox.utils as ul
import os
import pytest
import importlib_resources


# ---------------------------- FIXTURES -------------------------------- #


@pytest.fixture
def gtf():
    """Load gtf file.

    Returns
    -------
    str
        Path to GTF file with promoter annotations.
    """

    # Location of gene lists
    gtf_dir = importlib_resources.files("sctoolbox") / "data" / "promoters_gtf"
    gtf_path = os.path.join(gtf_dir, "mus_musculus.104.promoters2000.gtf")

    return gtf_path


@pytest.fixture
def gtf_with_header():
    """Load a gtf like file with a header.

    Returns
    -------
    str
        Path to GTF file with header.
    """
    gtf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.unsorted.gtf')

    return gtf_path


@pytest.fixture
def bed():
    """Load a bed with blacklisted regions.

    Returns
    -------
    str
        Path to BED file with blacklisted regions.
    """
    bed_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'hg38.blacklist.v2_sorted.bed')

    return bed_path


# --------------------------- TESTS --------------------------------- #


@pytest.mark.parametrize("out", [None, 'tmp'])
def test_convert_gtf_to_bed(tmpdir, gtf, out):
    """Test _convert_gtf_to_bed success."""
    if out:
        out = str(tmpdir)

    sorted_bed, temp = tl.calc_overlap_fc._convert_gtf_to_bed(gtf, out=out)
    name = gtf + "_sorted.bed"

    if out:
        expected = os.path.join(out, os.path.basename(name))
    else:
        expected = os.path.join(os.getcwd(), os.path.basename(name))

    assert sorted_bed == expected and os.path.isfile(sorted_bed)

    ul.io.rm_tmp(temp_files=temp)


@pytest.mark.parametrize("regions_file", ['bed', 'gtf'])
@pytest.mark.parametrize("bam_file,fragments_file", [('bam', None), (None, 'fragments')])
def testfc_fragments_in_regions(tmpdir, adata_atac, bed, gtf, atac_bam_file, sorted_fragments, regions_file, bam_file, fragments_file):
    """Test fc_fragments_in_regions function for run completion."""
    if regions_file == 'bed':
        regions_file = bed
    elif regions_file == 'gtf':
        regions_file = gtf

    if bam_file == 'bam':
        bam_file = atac_bam_file
    if fragments_file == 'fragments':
        fragments_file = sorted_fragments

    tl.calc_overlap_fc.fc_fragments_in_regions(adata_atac,
                                               regions_file=regions_file,
                                               bam_file=bam_file,
                                               fragments_file=fragments_file,
                                               regions_name='promoters',
                                               temp_dir=str(tmpdir))

    assert 'fold_change_promoters_fragments' in adata_atac.obs.columns
