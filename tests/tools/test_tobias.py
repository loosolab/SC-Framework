"""Tests for usage of TOBIAS within the sc framework."""
import os
import sctoolbox.tools.tobias as tobias
import yaml


# ------------------------------ TESTS --------------------------------- #


def test_write_TOBIAS_config(tmp_path):
    """Test write_TOBIAS_config success."""

    yml_path = str(tmp_path / "tobias.yml")
    tobias.write_TOBIAS_config(yml_path, bams=["bam1.bam", "bam2.bam"])
    yml = yaml.full_load(open(yml_path))

    assert yml["data"]["1"] == "bam1.bam"


def test_prepare_tobias(adata_atac, atac_bam_file, tmp_path):
    """Test prepare_tobias success."""

    output_base = str(tmp_path / "tobias")
    yml_path = str(tmp_path / "TOBIAS_config.yml")

    input_dir, output_dir, yml = tobias.prepare_tobias(adata_atac,
                                                       groupby='Sample',
                                                       output=output_base,
                                                       path_bam=atac_bam_file,
                                                       barcode_column=None,
                                                       barcode_tag='CB',
                                                       fasta='some.fa',
                                                       motifs=None,
                                                       gtf='genes.gtf',
                                                       blacklist=None,
                                                       organism='human',
                                                       yml=yml_path,
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
