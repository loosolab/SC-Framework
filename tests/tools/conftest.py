"""Fixtures available to all tests within the tools directory."""

import pytest
import os

from tests.conftest import ATAC_DATA_DIR


# ------------------------------ FIXTURES --------------------------------- #


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
def atac_gtf():
    """Path to GTF annotation file.

    Returns
    -------
    str
        Path to mm10_genes.gtf.
    """
    return os.path.join(ATAC_DATA_DIR, 'mm10_genes.gtf')
