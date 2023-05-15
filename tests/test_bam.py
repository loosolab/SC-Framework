import os
import pytest
import sctoolbox.bam
import glob
import scanpy as sc


@pytest.fixture
def bam_handle():
    """ Fixture for a bam file handle. """
    bam_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    handle = sctoolbox.bam.open_bam(bam_f, "rb")

    return handle


def test_open_bam(bam_handle):  # this is indirectly a test of sctoolbox.bam.open_bam

    assert type(bam_handle).__name__ == "AlignmentFile"


def test_get_bam_reads(bam_handle):

    total = sctoolbox.bam.get_bam_reads(bam_handle)

    assert total == 10000


def test_split_bam_clusters(bam_handle):

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
    """ Test whether the bigwig is written. """

    bigwig_out = "mm10_atac.bw"

    bam_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    bigwig_f = sctoolbox.bam.bam_to_bigwig(bam_f, output=bigwig_out, bgtobw_path="scripts/bedGraphToBigWig")  # tests are run from root

    assert os.path.exists(bigwig_f)

    os.remove(bigwig_out)
