"""Test functions related to files containing genomic ranges."""

import sctoolbox.tools as tools
import sctoolbox.utils as utils
import os
import pytest
import anndata as ad
import pkg_resources


@pytest.fixture
def bam():
    """Load bam file."""
    bam_dir = os.path.join(os.path.dirname(__file__), 'data', 'atac')
    bam_path = os.path.join(bam_dir, 'mm10_atac.bam')

    return bam_path


@pytest.fixture
def gtf():
    """Load gtf file."""

    # Location of gene lists
    gtf_dir = pkg_resources.resource_filename("sctoolbox", "data/promoters_gtf/")
    gtf_path = os.path.join(gtf_dir, "mus_musculus.104.promoters2000.gtf")

    return gtf_path


@pytest.fixture
def gtf_with_header():
    """Load a gtf like file with a header."""
    gtf_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.unsorted.gtf')

    return gtf_path


@pytest.fixture
def fragments():
    """Load a bed file containing fragments."""
    fragments_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_sorted_fragments.bed')

    return fragments_path


@pytest.fixture
def bed():
    """Load a bed with blacklisted regions."""
    bed_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'hg38.blacklist.v2_sorted.bed')

    return bed_path


@pytest.fixture
def adata():
    """Load adata."""
    adata_dir = os.path.join(os.path.dirname(__file__), 'data', 'atac')
    adata = ad.read_h5ad(adata_dir + '/mm10_atac.h5ad')

    return adata


@pytest.mark.parametrize("out", [None, 'tmp'])
def test_convert_gtf_to_bed(tmpdir, gtf, out):
    """Test _convert_gtf_to_bed success."""
    if out:
        out = str(tmpdir)

    sorted_bed, temp = tools._convert_gtf_to_bed(gtf, out=out)
    name = gtf + "_sorted.bed"

    if out:
        expected = os.path.join(out, os.path.basename(name))
    else:
        expected = os.path.join(os.getcwd(), os.path.basename(name))

    assert sorted_bed == expected and os.path.isfile(sorted_bed)

    utils.rm_tmp(temp_files=temp)


@pytest.mark.parametrize("regions_file", ['bed', 'gtf'])
@pytest.mark.parametrize("bam_file,fragments_file", [('bam', None), (None, 'fragments')])
def test_fc_fragments_in_regions(tmpdir, adata, bed, gtf, bam, fragments, regions_file, bam_file, fragments_file):
    """Test fc_fragments_in_regions function for run completion."""
    if regions_file == 'bed':
        regions_file = bed
    elif regions_file == 'gtf':
        regions_file = gtf

    if bam_file == 'bam':
        bam_file = bam
    if fragments_file == 'fragments':
        fragments_file = fragments

    tools.fc_fragments_in_regions(adata,
                                  regions_file=regions_file,
                                  bam_file=bam_file,
                                  fragments_file=fragments_file,
                                  regions_name='promoters',
                                  temp_dir=str(tmpdir))

    assert 'fold_change_promoters_fragments' in adata.obs.columns
