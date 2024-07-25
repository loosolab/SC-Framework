"""Test the tsse_score related functions."""
import pytest
import numpy as np
import scanpy as sc
import sctoolbox.tools as tools
import os


# ------------------------------ FIXTURES -------------------------------- #


@pytest.fixture
def adata():
    """Fixture for an AnnData object."""
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_atac.h5ad'))
    return adata


@pytest.fixture
def fragments():
    """Fixture for a fragments object."""
    fragments = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_atac_fragments.bed')
    return fragments


@pytest.fixture
def gtf():
    """Fixture for a gtf object."""
    gtf = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_genes.gtf')
    return gtf


@pytest.fixture
def tss_file():
    """Fixture for a tss_file object."""
    tss_file = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_tss.bed')
    return tss_file


# ------------------------------ TESTS --------------------------------- #


def test_write_TSS(gtf):
    """Test write_TSS function."""
    # Build temporary TSS file path
    temp_dir = os.path.join(os.path.dirname(__file__), '../data', 'atac')
    tss_file = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_genes_tss.bed')
    # Write TSS file
    tss_list, tempfiles = tools.tsse.write_TSS_bed(gtf, tss_file, temp_dir=temp_dir)
    # Add tss_file to tempfiles
    tempfiles.append(tss_file)

    # Check if file exists
    assert os.path.exists(tss_file)
    # Check if file is not empty
    assert os.path.getsize(tss_file) > 0
    # Check if file has 3 columns
    assert np.loadtxt(tss_file, dtype=str).shape[1] == 3
    # Check if the tss_list columns are in the expected format
    assert tss_list[0][0] == 'chr1'
    assert type(tss_list[0][1]) is int
    assert type(tss_list[0][2]) is int

    # Remove temporary files
    for tempfile in tempfiles:
        os.remove(tempfile)


def test_overlap_and_aggregate(gtf, fragments):
    """Test overlap_and_aggregate function."""
    # Build temporary TSS file path
    temp_dir = tss_file = os.path.join(os.path.dirname(__file__), '../data', 'atac')
    tss_file = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'mm10_genes_tss.bed')
    overlap = os.path.join(os.path.dirname(__file__), '../data', 'atac', 'overlap.bed')

    tempfiles = [tss_file, overlap]
    # Write TSS file
    tss_list, temp = tools.tsse.write_TSS_bed(gtf, tss_file, temp_dir=temp_dir)
    tempfiles.extend(temp)

    # overlap_and_aggregate
    agg, temp = tools.tsse.overlap_and_aggregate(fragments, tss_file, overlap, tss_list)
    tempfiles.extend(temp)

    # Check if agg is a dictionary
    assert type(agg) is dict
    # check if overlap file exists
    assert os.path.exists(overlap)
    # Check if file is not empty
    # assert os.path.getsize(overlap) > 0
    # Check if file has 3 columns
    assert np.loadtxt(overlap, dtype=str).shape[1] == 5

    # Remove temporary files
    for tempfile in tempfiles:
        os.remove(tempfile)


def test_add_tsse_score(adata, fragments, gtf):
    """Test add_tsse_score function."""
    adata = tools.tsse.add_tsse_score(adata,
                                      fragments,
                                      gtf,
                                      negativ_shift=2000,
                                      positiv_shift=2000,
                                      edge_size_total=100,
                                      edge_size_per_base=50,
                                      min_bias=0.01,
                                      keep_tmp=False,
                                      temp_dir="")
    assert 'tsse_score' in adata.obs.columns
