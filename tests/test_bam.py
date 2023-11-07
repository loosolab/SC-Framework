"""Test bam related functions."""

import os
import shutil
import pytest
import sctoolbox.bam
import glob
import scanpy as sc


@pytest.fixture
def bam_handle():
    """Fixture for a bam file handle."""
    bam_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    handle = sctoolbox.bam.open_bam(bam_f, "rb")

    return handle


def test_open_bam(bam_handle):  # this is indirectly a test of sctoolbox.bam.open_bam
    """Test open_bam success."""

    assert type(bam_handle).__name__ == "AlignmentFile"


def test_get_bam_reads(bam_handle):
    """Test get_bam_reads success."""

    total = sctoolbox.bam.get_bam_reads(bam_handle)

    assert total == 10000


def test_split_bam_clusters(bam_handle):
    """Test split_bam_clusters success."""

    bam_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')

    # Get input reads
    n_reads_input = sctoolbox.bam.get_bam_reads(bam_handle)

    # Split bam
    adata = sc.read_h5ad(adata_f)
    sctoolbox.bam.split_bam_clusters(adata, bam_f, groupby="Sample", parallel=True)

    # Check if the bam file is split and the right size
    output_bams = glob.glob("split_Sample*.bam")
    handles = [sctoolbox.bam.open_bam(f, "rb") for f in output_bams]
    n_reads_output = sum([sctoolbox.bam.get_bam_reads(handle) for handle in handles])

    assert n_reads_input == n_reads_output  # this is true because all groups are represented in the bam

    # Clean up
    for bam in output_bams:
        os.remove(bam)


def test_bam_to_bigwig():
    """Test whether the bigwig is written."""

    bigwig_out = "mm10_atac.bw"

    bam_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    bigwig_f = sctoolbox.bam.bam_to_bigwig(bam_f, output=bigwig_out, bgtobw_path="scripts/bedGraphToBigWig")  # tests are run from root

    assert os.path.exists(bigwig_f)

    os.remove(bigwig_out)


@pytest.mark.parametrize("bam_name, outdir, barcode_regex",
                         [('mm10_atac', None, None),
                          ('homo_sapiens_liver', 'fragment_file_output', "[^.]*"),
                          ('homo_sapiens_liver_sorted', None, "[^.]*")])
def test_create_fragment_file(bam_name, outdir, barcode_regex):
    """Test create_fragment_file success."""

    barcode_tag = "CB"
    if barcode_regex:
        barcode_tag = None

    bam_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', bam_name + ".bam")
    fragments_f = sctoolbox.bam.create_fragment_file(bam=bam_f, nproc=1, outdir=outdir,
                                                     barcode_tag=barcode_tag, barcode_regex=barcode_regex,  # homo_sapiens_liver has the barcode in the read name
                                                     index=True)  # requires bgzip and tabix

    outdir_fmt = os.path.dirname(bam_f) if outdir is None else outdir
    expected = os.path.join(outdir_fmt, bam_name + "_fragments.tsv")

    assert fragments_f == expected and os.path.isfile(fragments_f) and os.stat(fragments_f).st_size > 0

    # Clean up framgnets and output folder (if created)
    os.remove(fragments_f)
    if outdir is not None:
        shutil.rmtree(outdir)


def test_create_fragment_file_multiprocessing():
    """Assert that the result is the same regardless of number of cores used."""

    bam_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'homo_sapiens_liver_sorted.bam')

    n_fragments = []
    for nproc in [1, 4]:
        fragments_f = sctoolbox.bam.create_fragment_file(bam=bam_f, nproc=nproc, barcode_tag=None, barcode_regex="[^.]*")  # homo_sapiens_liver has the barcode in the read name
        n_fragments.append(len(open(fragments_f).readlines()))

    assert len(set(n_fragments)) == 1

    os.remove(fragments_f)
