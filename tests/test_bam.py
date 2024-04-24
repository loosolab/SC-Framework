"""Test bam related functions."""

import os
import shutil
import pytest
import sctoolbox.bam
import sctoolbox.tools.bam as stb
import glob
import scanpy as sc
import logging
import random
import re


@pytest.fixture
def bam_file():
    """Fixture pointing to test bam."""
    return os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')


@pytest.fixture
def bam_handle(bam_file):
    """Fixture for a bam file handle."""
    handle = sctoolbox.bam.open_bam(bam_file, "rb")

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
    obj = sc.read_h5ad(os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad'))

    return obj
