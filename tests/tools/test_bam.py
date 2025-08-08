"""Test bam related functions."""

import pytest
import os
import shutil
import glob
import logging
import scanpy as sc
import random
import re
from contextlib import contextmanager

import sctoolbox.tools.bam as stb

# ---------------------------- HELPER ------------------------------- #


@contextmanager
def add_logger_handler(logger, handler):
    """Temporarily add a handler to the given logger."""
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)

# ----------------------------- FIXTURES ------------------------------- #


@pytest.fixture
def bam_file():
    """Fixture pointing to test bam."""
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'mm10_atac.bam')


@pytest.fixture
def bam_handle(bam_file):
    """Fixture for a bam file handle."""
    handle = stb.open_bam(bam_file, "rb")

    return handle


@pytest.fixture
def barcodes(bam_file, bam_handle):
    """Return 100 randomly selected barcodes."""
    # bam_handle is not used to avoid consuming the iterator
    read_count = stb.open_bam(bam_file, "rb").count()

    # select random indexes
    indices = random.sample(range(0, read_count), k=100)

    return [read.get_tag("CB") for index, read in enumerate(bam_handle) if index in indices]


@pytest.fixture(scope="session")
def adata():
    """Load and returns an anndata object."""

    # has .X of type numpy.array
    obj = sc.read_h5ad(os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'mm10_atac.h5ad'))

    return obj


@pytest.fixture
def adata_atac():
    """Load atac adata."""
    adata_f = os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'mm10_atac.h5ad')
    return sc.read_h5ad(adata_f)


@pytest.fixture
def adata_atac_emptyvar(adata_atac):
    """Create adata with empty adata.var."""
    adata = adata_atac.copy()
    adata.var = adata.var.drop(columns=adata.var.columns)
    return adata


# ------------------------------ TESTS --------------------------------- #


def test_bam_adata_ov(adata_atac, bam_file):
    """Test bam_adata_ov success."""
    hitrate = stb.bam_adata_ov(adata_atac, bam_file, cb_tag='CB')
    assert hitrate >= 0.10


def test_check_barcode_tag(adata, bam_file, mocker, caplog):
    """Tests the barcode overlap amount between adata and bam file."""
    with caplog.at_level(logging.INFO), add_logger_handler(stb.logger, caplog.handler):
        # test overlap == 0%
        mocker.patch('sctoolbox.tools.bam.bam_adata_ov', return_value=0)
        stb.check_barcode_tag(adata=adata, bamfile=bam_file, cb_tag="CB")
        assert 'None of the barcodes from the bamfile found in the .obs table.\nConsider if you are using the wrong column cb-tag or bamfile.' in caplog.text

        # test overlap <= 5%
        mocker.patch('sctoolbox.tools.bam.bam_adata_ov', return_value=0.05)
        stb.check_barcode_tag(adata=adata, bamfile=bam_file, cb_tag="CB")
        assert 'Only 5% or less of the barcodes from the bamfile found in the .obs table.\nConsider if you are using the wrong column for cb-tag or bamfile.' in caplog.text

        # test overlap > 5%
        mocker.patch('sctoolbox.tools.bam.bam_adata_ov', return_value=0.8)
        stb.check_barcode_tag(adata=adata, bamfile=bam_file, cb_tag="CB")
        assert 'Barcode tag: OK' in caplog.text

        # test overlap error (TODO don't know how this could be triggered)
        with pytest.raises(ValueError):
            mocker.patch('sctoolbox.tools.bam.bam_adata_ov', return_value=float("nan"))
            stb.check_barcode_tag(adata=adata, bamfile=bam_file, cb_tag="CB")


def test_subset_bam(bam_file, barcodes, caplog, tmpdir):
    """Check the subset_bam function."""
    outfile = tmpdir / "subset.bam"

    with add_logger_handler(stb.logger, caplog.handler):
        # check success
        stb.subset_bam(bam_in=bam_file,
                       bam_out=str(outfile),
                       barcodes=barcodes,
                       read_tag="CB",
                       pysam_threads=4,
                       overwrite=False)

        assert bool(re.match(r"Wrote \d+ reads to output bam", caplog.messages[-1]))
        assert outfile.isfile()

        # check overwrite warning
        stb.subset_bam(bam_in=bam_file,
                       bam_out=str(outfile),
                       barcodes=barcodes,
                       read_tag="CB",
                       pysam_threads=4,
                       overwrite=False)

        assert f"Output file {str(outfile)} exists. Skipping." in caplog.text


@pytest.mark.parametrize("parallel,sort_bams,index_bams", [(True, True, True), (False, False, False)])
def test_split_bam_clusters(bam_handle, bam_file, adata, parallel, sort_bams, index_bams):
    """Test split_bam_clusters success."""
    # Get input reads
    n_reads_input = stb.get_bam_reads(bam_handle)

    # Split bam
    stb.split_bam_clusters(adata, bam_file, groupby="Sample", parallel=parallel, sort_bams=sort_bams, index_bams=index_bams, writer_threads=len(set(adata.obs["Sample"])) + 1)

    # Check if the bam file is split and the right size
    output_bams = glob.glob("split_Sample*.bam")
    handles = [stb.open_bam(f, "rb") for f in output_bams]
    n_reads_output = sum([stb.get_bam_reads(handle) for handle in handles])

    assert n_reads_input == n_reads_output  # this is true because all groups are represented in the bam

    # Clean up
    for bam in output_bams:
        os.remove(bam)


def test_failure_split_bam_clusters(bam_file, adata):
    """Test split_bam_clusters failure."""
    # test groupby
    with pytest.raises(ValueError):
        stb.split_bam_clusters(adata, bam_file, groupby="SOME_NONEXISTENT_COLUMN_NAME")

    # test barcode_col
    with pytest.raises(ValueError):
        stb.split_bam_clusters(adata, bam_file, groupby="Sample", barcode_col="SOME_NONEXISTENT_TAG")

    # test sort_bam/index_bam
    with pytest.raises(ValueError):
        stb.split_bam_clusters(adata, bam_file, groupby="Sample", sort_bams=False, index_bams=True)


def test_open_bam(bam_handle):  # this is indirectly a test of sctoolbox.bam.open_bam
    """Test open_bam success."""

    assert type(bam_handle).__name__ == "AlignmentFile"


def test_get_bam_reads(bam_handle):
    """Test get_bam_reads success."""

    total = stb.get_bam_reads(bam_handle)

    assert total == 10000


def test_bam_to_bigwig():
    """Test whether the bigwig is written."""

    bigwig_out = "mm10_atac.bw"

    bam_f = os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'mm10_atac.bam')
    bigwig_f = stb.bam_to_bigwig(bam_f, output=bigwig_out, bgtobw_path="scripts/bedGraphToBigWig")  # tests are run from root

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

    bam_f = os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', bam_name + ".bam")
    fragments_f = stb.create_fragment_file(bam=bam_f,
                                           nproc=1,
                                           outdir=outdir,
                                           barcode_tag=barcode_tag,
                                           barcode_regex=barcode_regex,  # homo_sapiens_liver has the barcode in the read name
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

    bam_f = os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'homo_sapiens_liver_sorted.bam')

    n_fragments = []
    for nproc in [1, 4]:
        fragments_f = stb.create_fragment_file(bam=bam_f, nproc=nproc, barcode_tag=None, barcode_regex="[^.]*")  # homo_sapiens_liver has the barcode in the read name
        n_fragments.append(len(open(fragments_f).readlines()))

    assert len(set(n_fragments)) == 1

    os.remove(fragments_f)
