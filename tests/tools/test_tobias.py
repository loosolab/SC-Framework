"""Tests for usage of TOBIAS within the sc framework."""
import pytest
import scanpy as sc
import os
import sctoolbox.tools.tobias as tobias
import yaml


# ----------------------------- FIXTURES ------------------------------- #


@pytest.fixture
def bam_file():
    """Fixture pointing to test bam."""
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'mm10_atac.bam')


@pytest.fixture(scope="session")
def adata():
    """Load and returns an anndata object."""

    # has .X of type numpy.array
    obj = sc.read_h5ad(os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'mm10_atac.h5ad'))

    return obj


# ------------------------------ TESTS --------------------------------- #


def test_write_TOBIAS_config():
    """Test write_TOBIAS_config success."""

    tobias.write_TOBIAS_config("tobias.yml", bams=["bam1.bam", "bam2.bam"])
    yml = yaml.full_load(open("tobias.yml"))

    assert yml["data"]["1"] == "bam1.bam"

def test_prepare_tobias(adata, bam_file):
    """Test prepare_tobias success."""

    input_dir, output_dir, yml = tobias.prepare_tobias(adata,
                                                    groupby='sample',
                                                    output='./tobias',
                                                    path_bam=bam_file,
                                                    barcode_column=None,
                                                    barcode_tag='CB',
                                                    fasta='some.fa',
                                                    motifs=None,
                                                    gtf='genes.gtf',
                                                    blacklist=None,
                                                    organism='human',
                                                    yml="TOBIAS_config.yml",
                                                    plot_comparison=True,
                                                    plot_correction=True,
                                                    plot_venn=True,
                                                    coverage=False,
                                                    wilson=False,
                                                    threads=4)

    with open(yml, 'r') as file:
        tobias_yaml = yaml.safe_load(file)

    # Check if entries of the yaml are valid
    # first order
    keys = ['data', 'run_info', 'flags', 'macs', 'atacorrect', 'footprinting', 'bindetect']
    assert all(k in tobias_yaml for k in keys)

    # second order
    run_info = ['organism', 'blacklist', 'fasta', 'gtf', 'motifs', 'output']
    assert all(k in tobias_yaml['run_info'] for k in run_info)

    flags = ['plot_correction', 'plot_venn', 'coverage', 'wilson']
    assert all(k in tobias_yaml['flags'] for k in flags)

    # check if directories exist
    assert os.path.isdir(input_dir)
    assert os.path.isdir(output_dir)

    # cleanup
    os.rmdir('./tobias')

