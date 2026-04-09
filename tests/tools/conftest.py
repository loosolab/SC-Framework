"""Fixtures available to all tests within the tools directory."""

import pytest
import scanpy as sc
import os


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ATAC_DATA_DIR = os.path.join(DATA_DIR, 'atac')


# ------------------------------ FIXTURES --------------------------------- #


@pytest.fixture
def adata_atac():
    """Load and return an ATAC-seq AnnData object.

    Returns
    -------
    anndata.AnnData
        ATAC-seq AnnData object from mm10_atac.h5ad.
    """
    return sc.read_h5ad(os.path.join(ATAC_DATA_DIR, 'mm10_atac.h5ad'))


@pytest.fixture
def adata_rna():
    """Load and return an RNA-seq AnnData object.

    Returns
    -------
    anndata.AnnData
        RNA-seq AnnData object from adata.h5ad.
    """
    return sc.read_h5ad(os.path.join(DATA_DIR, 'adata.h5ad'))


@pytest.fixture
def atac_bam_file():
    """Path to test BAM file.

    Returns
    -------
    str
        Path to mm10_atac.bam.
    """
    return os.path.join(ATAC_DATA_DIR, 'mm10_atac.bam')


@pytest.fixture
def atac_fragments():
    """Path to ATAC-seq fragments BED file.

    Returns
    -------
    str
        Path to mm10_atac_fragments.bed.
    """
    return os.path.join(ATAC_DATA_DIR, 'mm10_atac_fragments.bed')


@pytest.fixture
def sorted_fragments():
    """Path to sorted ATAC-seq fragments BED file.

    Returns
    -------
    str
        Path to mm10_sorted_fragments.bed.
    """
    return os.path.join(ATAC_DATA_DIR, 'mm10_sorted_fragments.bed')


@pytest.fixture
def atac_gtf():
    """Path to GTF annotation file.

    Returns
    -------
    str
        Path to mm10_genes.gtf.
    """
    return os.path.join(ATAC_DATA_DIR, 'mm10_genes.gtf')
